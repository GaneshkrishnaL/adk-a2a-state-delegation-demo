# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Remote A2A Agent 1 (Triage Service) & Remote A2A Agent 2 (Scheduling Service)
#
# Public Documentation References:
#   - Exposing ADK Agents as A2A Servers (`to_a2a`, `A2aAgentExecutor`):
#     https://google.github.io/adk-docs/a2a/
#   - Accessing Session State in Tools (`ToolContext.state`) & Instructions (`{key}`):
#     https://google.github.io/adk-docs/sessions/state/
#   - ADK Function Tools:
#     https://google.github.io/adk-docs/tools/function-tools/

from __future__ import annotations

from typing import Any

from a2a_state_bridge import state_aware_request_converter
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutorConfig
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.tools.tool_context import ToolContext
from mock_llm import get_model
from starlette.applications import Starlette


# ============================================================================
# REMOTE AGENT 1: Clinical Triage Specialist (Port 8001)
# ============================================================================


def evaluate_clinical_triage(
    reported_symptom: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
  """Evaluates symptoms using custom state delegated from the Root Orchestrator.

  How state arrives here:
    `tool_context.state` on Remote Agent 1 contains `member_id`, `facility_id`,
    and `clinical_flags` that were forwarded in A2A request metadata and
    injected into Remote Agent 1's session by `state_aware_request_converter`.
    See: https://google.github.io/adk-docs/sessions/state/
  """
  member_id = tool_context.state.get("member_id", "UNKNOWN_MEMBER")
  facility_id = tool_context.state.get("facility_id", "UNKNOWN_FACILITY")
  clinical_flags = tool_context.state.get("clinical_flags", "NONE")

  print(
      "\n  [Remote Agent 1 Tool] `evaluate_clinical_triage` reading local"
      " session.state:"
  )
  print(f"    • tool_context.state['member_id']      = {member_id!r}")
  print(f"    • tool_context.state['facility_id']    = {facility_id!r}")
  print(f"    • tool_context.state['clinical_flags'] = {clinical_flags!r}")

  is_cardiac = (
      "chest" in reported_symptom.lower()
      or "cardiac" in str(clinical_flags).lower()
  )
  urgency = "PRIORITY_24H" if is_cardiac else "ROUTINE_7D"
  specialty = "Interventional Cardiology" if is_cardiac else "Internal Medicine"

  result = {
      "member_id": member_id,
      "facility_id": facility_id,
      "reported_symptom": reported_symptom,
      "urgency_level": urgency,
      "recommended_specialty": specialty,
      "clinical_protocol": "ACS-OUTPATIENT-PATHWAY-01",
  }
  tool_context.state["local_triage_decision"] = result
  return result


def build_remote_triage_agent(offline_mock: bool = False) -> LlmAgent:
  """Constructs the ADK LlmAgent for Remote Agent 1 (Triage Service)."""
  return LlmAgent(
      name="triage_specialist_service",
      model=get_model("remote_agent_1_triage", offline_mock=offline_mock),
      description=(
          "Remote A2A Triage specialist that evaluates symptoms and determines"
          " urgency and recommended specialty."
      ),
      instruction=(
          "You are the Clinical Triage Remote Agent.\n"
          "Delegated Session Context from Orchestrator:\n"
          "- Member ID: {member_id}\n"
          "- Facility ID: {facility_id}\n"
          "- Clinical Flags: {clinical_flags}\n\n"
          "Always call `evaluate_clinical_triage` first and return a concise"
          " summary of the triage urgency, recommended specialty, facility,"
          " and clinical protocol."
      ),
      tools=[evaluate_clinical_triage],
  )


def create_remote_agent_1_app(
    host: str = "localhost",
    port: int = 8001,
    offline_mock: bool = False,
) -> Starlette:
  """Exposes Remote Agent 1 as an A2A Starlette application via `to_a2a`."""
  agent = build_remote_triage_agent(offline_mock=offline_mock)

  def _executor_factory(runner: Runner) -> A2aAgentExecutor:
    return A2aAgentExecutor(
        runner=runner,
        config=A2aAgentExecutorConfig(
            request_converter=state_aware_request_converter
        ),
    )

  return to_a2a(
      agent,
      host=host,
      port=port,
      protocol="http",
      agent_executor_factory=_executor_factory,
  )


# ============================================================================
# REMOTE AGENT 2: Scheduling & Benefits Specialist (Port 8002)
# ============================================================================


def book_appointment_and_copay(
    preferred_window: str,
    tool_context: ToolContext,
) -> dict[str, Any]:
  """Books an appointment using `triage_output` produced by Remote Agent 1.

  How chained state arrives here:
    1. Remote Agent 1 returned its triage assessment to the Orchestrator.
    2. The Orchestrator saved that output into `session.state['triage_output']`
       via `make_output_key_interceptor("triage_output")`.
    3. The Orchestrator forwarded `triage_output` (plus `plan_tier`) to Remote
       Agent 2 via `make_state_forwarding_meta_provider`, which injected it
       into Remote Agent 2's `tool_context.state['triage_output']`.
  """
  member_id = tool_context.state.get("member_id", "UNKNOWN_MEMBER")
  facility_id = tool_context.state.get("facility_id", "UNKNOWN_FACILITY")
  plan_tier = tool_context.state.get("plan_tier", "STANDARD")
  triage_output = tool_context.state.get("triage_output", "NO_TRIAGE_OUTPUT")

  print(
      "\n  [Remote Agent 2 Tool] `book_appointment_and_copay` reading local"
      " session.state (including Remote Agent 1's `triage_output`!):"
  )
  print(f"    • tool_context.state['member_id']     = {member_id!r}")
  print(f"    • tool_context.state['plan_tier']     = {plan_tier!r}")
  print(f"    • tool_context.state['triage_output'] = {triage_output!r}")

  is_priority_cardio = "cardiology" in str(triage_output).lower()
  copay = "$25.00" if plan_tier == "PREMIER_EPO" else "$60.00"

  slot_info = {
      "member_id": member_id,
      "preferred_window": preferred_window,
      "consumed_triage_specialty": (
          "Interventional Cardiology (Priority 24h)"
          if is_priority_cardio
          else "General Internal Medicine"
      ),
      "booked_slot": "Tomorrow 09:30 AM CT",
      "provider_name": "Dr. Sarah Jenkins, MD (FACC)",
      "location": f"{facility_id} - Heart & Vascular Pavilion, Suite 400",
      "plan_tier": plan_tier,
      "estimated_copay": copay,
  }
  tool_context.state["local_booking_record"] = slot_info
  return slot_info


def build_remote_scheduling_agent(offline_mock: bool = False) -> LlmAgent:
  """Constructs the ADK LlmAgent for Remote Agent 2 (Scheduling Service)."""
  return LlmAgent(
      name="scheduling_specialist_service",
      model=get_model("remote_agent_2_scheduling", offline_mock=offline_mock),
      description=(
          "Remote A2A Scheduling & Benefits specialist that books appointments"
          " based on upstream triage output and insurance plan tier."
      ),
      instruction=(
          "You are the Scheduling & Benefits Remote Agent.\n"
          "Delegated Session Context from Orchestrator:\n"
          "- Member ID: {member_id}\n"
          "- Facility ID: {facility_id}\n"
          "- Plan Tier: {plan_tier}\n"
          "- Upstream Triage Output (from Remote Agent 1): {triage_output}\n\n"
          "Call `book_appointment_and_copay` and return the booked slot,"
          " provider, location, and estimated copay."
      ),
      tools=[book_appointment_and_copay],
  )


def create_remote_agent_2_app(
    host: str = "localhost",
    port: int = 8002,
    offline_mock: bool = False,
) -> Starlette:
  """Exposes Remote Agent 2 as an A2A Starlette application via `to_a2a`."""
  agent = build_remote_scheduling_agent(offline_mock=offline_mock)

  def _executor_factory(runner: Runner) -> A2aAgentExecutor:
    return A2aAgentExecutor(
        runner=runner,
        config=A2aAgentExecutorConfig(
            request_converter=state_aware_request_converter
        ),
    )

  return to_a2a(
      agent,
      host=host,
      port=port,
      protocol="http",
      agent_executor_factory=_executor_factory,
  )


# ASGI entrypoints for running standalone with uvicorn:
#   uvicorn remote_agents:agent1_app --host localhost --port 8001
#   uvicorn remote_agents:agent2_app --host localhost --port 8002
agent1_app = create_remote_agent_1_app()
agent2_app = create_remote_agent_2_app()
