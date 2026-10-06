# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# End-to-End Reference Implementation:
#   ADK + A2A Custom `session.state` Delegation & `output_key` Chaining
#
# Public Documentation References:
#   - ADK Multi-Agent Systems (`SequentialAgent`, `AgentTool`, `sub_agents`):
#     https://google.github.io/adk-docs/agents/multi-agents/
#   - ADK Session State & `output_key`:
#     https://google.github.io/adk-docs/sessions/state/
#   - ADK A2A Integration (`RemoteA2aAgent`, `to_a2a`):
#     https://google.github.io/adk-docs/a2a/
#   - Vertex AI Agent Engine Overview:
#     https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview

from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
import json
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore", message=".*EXPERIMENTAL.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from a2a_state_bridge import make_output_key_interceptor
from a2a_state_bridge import make_state_forwarding_meta_provider
from google.adk.a2a.agent.config import A2aRemoteAgentConfig
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.remote_a2a_agent import AGENT_CARD_WELL_KNOWN_PATH
from google.adk.agents.remote_a2a_agent import RemoteA2aAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.agent_tool import AgentTool
from google.genai import types
import httpx
from mock_llm import get_model
from remote_agents import create_remote_agent_1_app
from remote_agents import create_remote_agent_2_app


def build_chained_orchestrator(
    client_1: httpx.AsyncClient | None,
    client_2: httpx.AsyncClient | None,
    offline_mock: bool = False,
) -> SequentialAgent:
  """Builds the Root Orchestrator chaining Remote Agent 1 -> `output_key` -> Remote Agent 2 -> Consolidator.

  Reference:
    https://google.github.io/adk-docs/agents/multi-agents/#sequential-agent
  """

  # Step 1: Remote A2A Agent 1 (Triage Specialist)
  #   - Forwards ["member_id", "facility_id", "clinical_flags"] from Orchestrator session.state
  #   - Saves its output into Orchestrator session.state["triage_output"]
  remote_triage_agent = RemoteA2aAgent(
      name="triage_specialist_agent",
      description="Remote A2A Clinical Triage Specialist (Server 1)",
      agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
      httpx_client=client_1,
      a2a_request_meta_provider=make_state_forwarding_meta_provider(
          ["member_id", "facility_id", "clinical_flags"]
      ),
      config=A2aRemoteAgentConfig(
          forward_session_id_as_context_id=True,
          request_interceptors=[make_output_key_interceptor("triage_output")],
      ),
  )

  # Step 2: Remote A2A Agent 2 (Scheduling & Benefits Specialist)
  #   - Forwards ["member_id", "facility_id", "plan_tier", "triage_output"]
  #     from Orchestrator session.state (where `triage_output` came from Remote Agent 1!)
  #   - Saves its output into Orchestrator session.state["scheduling_output"]
  remote_scheduling_agent = RemoteA2aAgent(
      name="scheduling_specialist_agent",
      description="Remote A2A Scheduling & Benefits Specialist (Server 2)",
      agent_card=f"http://localhost:8002{AGENT_CARD_WELL_KNOWN_PATH}",
      httpx_client=client_2,
      a2a_request_meta_provider=make_state_forwarding_meta_provider(
          ["member_id", "facility_id", "plan_tier", "triage_output"]
      ),
      config=A2aRemoteAgentConfig(
          forward_session_id_as_context_id=True,
          request_interceptors=[
              make_output_key_interceptor("scheduling_output")
          ],
      ),
  )

  # Step 3: Orchestrator Consolidator LlmAgent
  #   - Reads `{triage_output}` and `{scheduling_output}` directly from Orchestrator
  #     `session.state` via ADK's native `{key}` state templating:
  #     https://google.github.io/adk-docs/sessions/state/
  #   - Writes consolidated response to `session.state["final_care_plan"]`
  consolidator_agent = LlmAgent(
      name="care_plan_consolidator_agent",
      model=get_model("orchestrator_consolidator", offline_mock=offline_mock),
      description="Consolidates Remote Agent 1 and Remote Agent 2 outputs.",
      instruction=(
          "You are the Root Care Orchestrator Consolidator.\n"
          "Consolidate the following state values from `session.state` into a"
          " clear, member-facing care plan:\n"
          "  • Member ID: {member_id}\n"
          "  • Facility ID: {facility_id}\n"
          "  • Plan Tier: {plan_tier}\n"
          "  • Remote Agent 1 Output (triage_output): {triage_output}\n"
          "  • Remote Agent 2 Output (scheduling_output): {scheduling_output}\n"
      ),
      output_key="final_care_plan",
  )

  return SequentialAgent(
      name="root_care_orchestrator",
      description=(
          "Orchestrates Remote Agent 1 -> triage_output -> Remote Agent 2 ->"
          " scheduling_output -> Final Consolidated Answer."
      ),
      sub_agents=[
          remote_triage_agent,
          remote_scheduling_agent,
          consolidator_agent,
      ],
  )


def build_parallel_single_turn_orchestrator(
    client_1: httpx.AsyncClient | None,
    client_2: httpx.AsyncClient | None,
    offline_mock: bool = False,
) -> LlmAgent:
  """Builds an LlmAgent Orchestrator that runs Remote Agent 1 & 2 in parallel via `AgentTool`.

  Why `AgentTool` is the `mode="single_turn"` equivalent for `RemoteA2aAgent`:
    Wrapping a `RemoteA2aAgent` in `AgentTool` allows the Root `LlmAgent` to
    emit multiple tool calls in a single LLM turn, which ADK executes
    concurrently via `asyncio.gather` while keeping control in the Root
    Orchestrator.
    See: https://google.github.io/adk-docs/agents/multi-agents/#agent-as-a-tool
  """
  remote_1 = RemoteA2aAgent(
      name="triage_specialist_agent",
      description="Remote A2A Clinical Triage Specialist",
      agent_card=f"http://localhost:8001{AGENT_CARD_WELL_KNOWN_PATH}",
      httpx_client=client_1,
      a2a_request_meta_provider=make_state_forwarding_meta_provider(
          ["member_id", "facility_id", "clinical_flags"]
      ),
      config=A2aRemoteAgentConfig(
          request_interceptors=[make_output_key_interceptor("triage_output")],
      ),
  )
  remote_2 = RemoteA2aAgent(
      name="scheduling_specialist_agent",
      description="Remote A2A Scheduling & Copay Specialist",
      agent_card=f"http://localhost:8002{AGENT_CARD_WELL_KNOWN_PATH}",
      httpx_client=client_2,
      a2a_request_meta_provider=make_state_forwarding_meta_provider(
          ["member_id", "facility_id", "plan_tier", "triage_output"]
      ),
      config=A2aRemoteAgentConfig(
          request_interceptors=[
              make_output_key_interceptor("scheduling_output")
          ],
      ),
  )
  return LlmAgent(
      name="parallel_care_orchestrator",
      model=get_model("parallel_orchestrator", offline_mock=offline_mock),
      instruction=(
          "Call both `triage_specialist_agent` and"
          " `scheduling_specialist_agent` in parallel and summarize their"
          " responses."
      ),
      tools=[
          AgentTool(agent=remote_1),
          AgentTool(agent=remote_2),
      ],
  )


async def run_demos(
    live_http: bool = False,
    offline_mock: bool = False,
) -> None:
  async with AsyncExitStack() as stack:
    if live_http:
      print(
          "Connecting to external A2A HTTP servers at http://localhost:8001 and"
          " http://localhost:8002"
      )
      client_1 = await stack.enter_async_context(httpx.AsyncClient(timeout=60.0))
      client_2 = await stack.enter_async_context(httpx.AsyncClient(timeout=60.0))
    else:
      print(
          "Starting Remote A2A Agent 1 (port 8001) and Remote A2A Agent 2 (port"
          " 8002) via in-process ASGI transport (zero external ports needed)"
      )
      app_1 = create_remote_agent_1_app(offline_mock=offline_mock)
      app_2 = create_remote_agent_2_app(offline_mock=offline_mock)
      # Trigger Starlette lifespan so A2A AgentCards & JSON-RPC routes are mounted
      await stack.enter_async_context(app_1.router.lifespan_context(app_1))
      await stack.enter_async_context(app_2.router.lifespan_context(app_2))
      client_1 = await stack.enter_async_context(
          httpx.AsyncClient(
              transport=httpx.ASGITransport(app=app_1),
              base_url="http://localhost:8001",
              timeout=60.0,
          )
      )
      client_2 = await stack.enter_async_context(
          httpx.AsyncClient(
              transport=httpx.ASGITransport(app=app_2),
              base_url="http://localhost:8002",
              timeout=60.0,
          )
      )

    # ========================================================================
    # DEMO 1: Custom State -> Remote Agent 1 -> output_key -> Remote Agent 2 -> Consolidate
    # ========================================================================
    print("\n" + "=" * 88)
    print(
        "DEMO 1: A2A SESSION.STATE DELEGATION & OUTPUT_KEY CHAINING"
        " (ORCHESTRATOR -> A1 -> A2)"
    )
    print("=" * 88)

    orchestrator = build_chained_orchestrator(
        client_1=client_1, client_2=client_2, offline_mock=offline_mock
    )
    runner = InMemoryRunner(
        agent=orchestrator, app_name="care_coordination_orchestrator"
    )

    user_id = "member_user_48291"
    session_id = "sess_cardio_triage_001"
    initial_custom_state = {
        "member_id": "MEM-99412",
        "facility_id": "FACILITY-CENTRAL-01",
        "clinical_flags": "History of hypertension; active cardiac watch",
        "plan_tier": "PREMIER_EPO",
    }

    await runner.session_service.create_session(
        app_name="care_coordination_orchestrator",
        user_id=user_id,
        session_id=session_id,
        state=initial_custom_state,
    )
    print("\n[Step 0] Seeded Orchestrator initial `session.state`:")
    print(json.dumps(initial_custom_state, indent=2))

    user_msg = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=(
                    "I have been experiencing exertional chest tightness and"
                    " shortness of breath today. Please triage my symptoms and"
                    " book the appropriate specialist appointment with my copay"
                    " estimate."
                )
            )
        ],
    )

    final_answer_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=user_msg,
    ):
      if event.content and event.content.parts:
        texts = [p.text for p in event.content.parts if p.text and not p.thought]
        if texts:
          joined = "".join(texts)
          print(f"\n[Event Emitted by '{event.author}']:\n{joined}")
          if event.author == "care_plan_consolidator_agent":
            final_answer_text = joined

    updated_session = await runner.session_service.get_session(
        app_name="care_coordination_orchestrator",
        user_id=user_id,
        session_id=session_id,
    )
    print("\n" + "-" * 88)
    print("FINAL ORCHESTRATOR `session.state` SNAPSHOT AFTER CHAIN COMPLETION:")
    print("-" * 88)
    print(json.dumps(updated_session.state, indent=2))

    print("\n" + "-" * 88)
    print("FINAL CONSOLIDATED ANSWER RETURNED TO USER:")
    print("-" * 88)
    print(final_answer_text)

    # ========================================================================
    # DEMO 2: Parallel Single-Turn Execution of Remote A2A Agents via AgentTool
    # ========================================================================
    print("\n" + "=" * 88)
    print(
        "DEMO 2 (BONUS): PARALLEL SINGLE-TURN EXECUTION OF REMOTE A2A AGENTS"
        " VIA AgentTool"
    )
    print("=" * 88)

    parallel_orch = build_parallel_single_turn_orchestrator(
        client_1=client_1, client_2=client_2, offline_mock=offline_mock
    )
    parallel_runner = InMemoryRunner(
        agent=parallel_orch, app_name="care_coordination_parallel"
    )
    await parallel_runner.session_service.create_session(
        app_name="care_coordination_parallel",
        user_id=user_id,
        session_id="sess_parallel_002",
        state={
            **initial_custom_state,
            "triage_output": "Pre-authorized Interventional Cardiology referral",
        },
    )
    async for event in parallel_runner.run_async(
        user_id=user_id,
        session_id="sess_parallel_002",
        new_message=types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="Run triage and scheduling check in parallel."
                )
            ],
        ),
    ):
      if event.content and event.content.parts:
        fn_calls = [
            p.function_call.name
            for p in event.content.parts
            if p.function_call
        ]
        if fn_calls:
          print(
              f"\n[Parallel Fan-Out by '{event.author}'] Invoking A2A"
              f" tools: {fn_calls}"
          )
        texts = [p.text for p in event.content.parts if p.text and not p.thought]
        if texts:
          print(f"\n[Parallel Consolidated Response]:\n{''.join(texts)}")


def test_run_demos() -> None:
  """Pytest entrypoint (`pytest -s run_demo.py`)."""
  asyncio.run(run_demos(live_http=False, offline_mock=True))


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Run ADK + A2A State Delegation & Output-Key Chaining Demo."
  )
  parser.add_argument(
      "--live-http",
      action="store_true",
      help=(
          "Connect over TCP sockets to http://localhost:8001 and"
          " http://localhost:8002 (start servers with ./run_live_servers.sh)."
      ),
  )
  parser.add_argument(
      "--offline-mock",
      action="store_true",
      help=(
          "Use the deterministic offline BaseLlm instead of Vertex AI"
          " gemini-2.5-flash."
      ),
  )
  args = parser.parse_args()
  asyncio.run(
      run_demos(live_http=args.live_http, offline_mock=args.offline_mock)
  )


if __name__ == "__main__":
  main()
