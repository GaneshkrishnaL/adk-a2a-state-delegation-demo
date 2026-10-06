# ADK + A2A `session.state` Delegation & `output_key` Chaining Reference Implementation

This reference package demonstrates how to build a multi-agent system with the **Google Agent Development Kit (ADK)** and **Agent2Agent (A2A) Protocol** on **Vertex AI (`gemini-2.5-flash`)** where:

1. **Custom `session.state` Delegation (Orchestrator $\rightarrow$ Remote Agent 1)**: The Root Orchestrator seeds custom `session.state` (`member_id`, `facility_id`, `clinical_flags`, `plan_tier`) and forwards selected state keys across an **A2A HTTP boundary** to **Remote A2A Agent 1 (`triage_specialist_agent`)**.
2. **Capturing Remote A2A Output via `output_key` (`triage_output`)**: Remote Agent 1's response is automatically captured into the Orchestrator's `session.state["triage_output"]` via a lightweight `RequestInterceptor`.
3. **Chaining State Across Remote Agents (Orchestrator $\rightarrow$ Remote Agent 2)**: The Orchestrator forwards `triage_output` (produced by Remote Agent 1) plus `plan_tier` from its `session.state` across A2A to **Remote A2A Agent 2 (`scheduling_specialist_agent`)**, whose tool (`book_appointment_and_copay`) reads `tool_context.state["triage_output"]` to book the appropriate specialist slot and calculate the copay.
4. **Orchestrator Consolidation (`final_care_plan`)**: Remote Agent 2's response is saved to `session.state["scheduling_output"]`, and the Orchestrator's Consolidator `LlmAgent` reads both `{triage_output}` and `{scheduling_output}` from `session.state` to return the consolidated final answer.
5. **Parallel Single-Turn Fan-Out (`AgentTool`)**: Demonstrates invoking multiple `RemoteA2aAgent` instances concurrently in a single turn using `AgentTool`.

---

## Architecture Flow

```
Orchestrator Session State (Initial):
  member_id      = "MEM-99412"
  facility_id    = "FACILITY-CENTRAL-01"
  clinical_flags = "History of hypertension; active cardiac watch"
  plan_tier      = "PREMIER_EPO"
        │
        ▼ (1) a2a_request_meta_provider forwards [member_id, facility_id, clinical_flags]
┌─────────────────────────────────────────────────────────────┐
│ Remote A2A Agent 1: triage_specialist_service (Port 8001)   │
│  • state_aware_request_converter injects state_delta        │
│  • Tool `evaluate_clinical_triage` reads tool_context.state │
└─────────────────────────────────────────────────────────────┘
        │
        ▼ (2) make_output_key_interceptor("triage_output") writes response to:
              Orchestrator session.state["triage_output"]
        │
        ▼ (3) a2a_request_meta_provider forwards [member_id, facility_id, plan_tier, triage_output]
┌───────────────────────────────────────────────────────────────┐
│ Remote A2A Agent 2: scheduling_specialist_service (Port 8002) │
│  • state_aware_request_converter injects triage_output        │
│  • Tool `book_appointment_and_copay` consumes                 │
│    tool_context.state["triage_output"] & plan_tier            │
└───────────────────────────────────────────────────────────────┘
        │
        ▼ (4) make_output_key_interceptor("scheduling_output") writes response to:
              Orchestrator session.state["scheduling_output"]
        │
        ▼ (5) Orchestrator Consolidator (`LlmAgent` with `output_key="final_care_plan"`)
              reads `{triage_output}` + `{scheduling_output}` from `session.state`
              and returns the consolidated final care plan!
```

---

## Files Included

| File | Purpose |
|---|---|
| `a2a_state_bridge.py` | Reusable 3-function A2A state bridge: `make_state_forwarding_meta_provider`, `make_output_key_interceptor`, and `state_aware_request_converter` (including automatic Vertex AI `part_metadata` sanitization). |
| `remote_agents.py` | Defines **Remote Agent 1** (`triage_specialist_service`) and **Remote Agent 2** (`scheduling_specialist_service`) and exposes them as A2A Starlette applications via `to_a2a(...)`. |
| `run_demo.py` | Executes **Demo 1** (Sequential A2A State Delegation & `output_key` Chaining) and **Demo 2** (Parallel Single-Turn `RemoteA2aAgent` Fan-Out via `AgentTool`). |
| `mock_llm.py` | Model factory returning Vertex AI `gemini-2.5-flash` by default, or an optional offline deterministic `BaseLlm` when `--offline-mock` is passed. |
| `run_live_servers.sh` | Starts Remote Agent 1 on `http://localhost:8001` and Remote Agent 2 on `http://localhost:8002` via `uvicorn` and runs `run_demo.py --live-http`. |
| `requirements.txt` | Python package dependencies (`google-adk[a2a]`, `a2a-sdk`, `httpx`, `starlette`, `uvicorn`). |

---

## How to Run (Vertex AI `gemini-2.5-flash`)

### 1. Install Dependencies & Configure Vertex AI Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Authenticate with Google Cloud Application Default Credentials (ADC):
gcloud auth application-default login

# Configure ADK / Google GenAI SDK to use Vertex AI:
export GOOGLE_GENAI_USE_VERTEXAI=TRUE
export GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
export GOOGLE_CLOUD_LOCATION="us-central1"
```

### 2. Option A: Run in Single-Command Mode (In-Process ASGI Transport)
Runs the Root Orchestrator and both Remote A2A Starlette apps over in-memory ASGI HTTP transport (no separate server terminals required):

```bash
python3 run_demo.py
```

### 3. Option B: Run Against Separate Live HTTP Uvicorn Servers (Ports 8001 & 8002)
Starts two background Uvicorn A2A servers (`http://localhost:8001` and `http://localhost:8002`) and runs the Orchestrator over TCP sockets:

```bash
chmod +x run_live_servers.sh
./run_live_servers.sh
```

### 4. Option C: Offline Dry-Run Mode (No GCP Credentials Required)
To verify the A2A HTTP transport, `session.state` delegation, tool execution, and `output_key` chaining without calling the Vertex AI API:

```bash
python3 run_demo.py --offline-mock
```

---

## Note on Vertex AI & A2A `part_metadata`

When `RemoteA2aAgent` serializes a user message into an A2A `Part`, it attaches A2A part-level metadata (`{"is_user_input": True}`). When converted back into a `google.genai.types.Part`, `part_metadata` can be populated. Because the Vertex AI `generateContent` serializer (`_Part_to_vertex` in `google-genai`) rejects `part_metadata` (which is only supported in the Gemini Developer API / AI Studio mode), `a2a_state_bridge.py` automatically strips `part.part_metadata = None` at the A2A boundary via `_strip_vertex_unsupported_part_metadata`.

---

## Public Documentation References

* **Agent Development Kit (ADK) Documentation**: [https://google.github.io/adk-docs/](https://google.github.io/adk-docs/)
* **ADK Session State & `output_key`**: [https://google.github.io/adk-docs/sessions/state/](https://google.github.io/adk-docs/sessions/state/)
* **ADK Multi-Agent Systems (`SequentialAgent`, `ParallelAgent`, `AgentTool`)**: [https://google.github.io/adk-docs/agents/multi-agents/](https://google.github.io/adk-docs/agents/multi-agents/)
* **ADK A2A Protocol Integration (`RemoteA2aAgent`, `to_a2a`)**: [https://google.github.io/adk-docs/a2a/](https://google.github.io/adk-docs/a2a/)
* **Agent2Agent (A2A) Protocol Specification**: [https://a2a-protocol.org/latest/specification/](https://a2a-protocol.org/latest/specification/)
* **Vertex AI Agent Engine Overview**: [https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview)
* **Vertex AI Agent Engine Memory Bank Overview**: [https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/overview](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/overview)
* **Vertex AI Agent Engine Sessions Overview**: [https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/sessions/overview](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/sessions/overview)
* **Vertex AI Agent Engine Tracing & Observability**: [https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/manage/tracing](https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/manage/tracing)
