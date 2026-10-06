# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Model helper for running with Vertex AI Gemini (`gemini-2.5-flash`) by default
# or with an optional offline deterministic BaseLlm (`--offline-mock`).
#
# Public Documentation References:
#   - ADK Gemini Model Configuration (Vertex AI):
#     https://google.github.io/adk-docs/agents/models/

from __future__ import annotations

import json
import os
from typing import AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types


def _extract_last_function_responses(
    contents: list[types.Content],
) -> list[types.FunctionResponse]:
  """Returns function_response parts from the most recent turn if present."""
  if not contents:
    return []
  last = contents[-1]
  if not last.parts:
    return []
  return [p.function_response for p in last.parts if p.function_response]


class OfflineDeterministicLlm(BaseLlm):
  """Optional deterministic ADK BaseLlm for offline unit testing (`--offline-mock`)."""

  model: str = "offline-deterministic-llm"
  role_name: str = "generic"

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    del stream
    sys_instruction = ""
    if llm_request.config and llm_request.config.system_instruction:
      sys_instruction = str(llm_request.config.system_instruction)

    fn_responses = _extract_last_function_responses(llm_request.contents)

    # 1. Remote Agent 1: Triage Specialist
    if self.role_name == "remote_agent_1_triage":
      if not fn_responses:
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="evaluate_clinical_triage",
                        args={
                            "reported_symptom": (
                                "exertional chest tightness and shortness of"
                                " breath"
                            )
                        },
                    )
                ],
            )
        )
        return

      tool_result = fn_responses[0].response or {}
      summary = (
          f"[TRIAGE_RESULT] Urgency={tool_result.get('urgency_level')} | "
          f"Specialty={tool_result.get('recommended_specialty')} | "
          f"Facility={tool_result.get('facility_id')} | "
          f"Member={tool_result.get('member_id')} | "
          f"Protocol={tool_result.get('clinical_protocol')}"
      )
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[types.Part.from_text(text=summary)],
          )
      )
      return

    # 2. Remote Agent 2: Scheduling & Benefits Specialist
    if self.role_name == "remote_agent_2_scheduling":
      if not fn_responses:
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="book_appointment_and_copay",
                        args={"preferred_window": "Next Available Priority"},
                    )
                ],
            )
        )
        return

      tool_result = fn_responses[0].response or {}
      summary = (
          f"[SCHEDULING_RESULT] Slot={tool_result.get('booked_slot')} | "
          f"Provider={tool_result.get('provider_name')} | "
          f"Location={tool_result.get('location')} | "
          f"Copay={tool_result.get('estimated_copay')} "
          f"({tool_result.get('plan_tier')}) | "
          f"BasedOnTriage={tool_result.get('consumed_triage_specialty')}"
      )
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[types.Part.from_text(text=summary)],
          )
      )
      return

    # 3. Orchestrator Consolidator Agent
    if self.role_name == "orchestrator_consolidator":
      final_text = (
          "=== CONSOLIDATED CARE PLAN ===\n"
          "Orchestrator chained custom state across Remote A2A Agents 1 and 2:\n"
          f"{sys_instruction.strip()}"
      )
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[types.Part.from_text(text=final_text)],
          )
      )
      return

    # 4. Bonus Demo: Parallel Root Orchestrator (emits 2 FunctionCalls in 1 turn)
    if self.role_name == "parallel_orchestrator":
      if not fn_responses:
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_function_call(
                        name="triage_specialist_agent",
                        args={
                            "request": (
                                "Triage member symptoms using delegated"
                                " session state."
                            )
                        },
                    ),
                    types.Part.from_function_call(
                        name="scheduling_specialist_agent",
                        args={
                            "request": (
                                "Check priority cardiology slot and copay"
                                " using delegated session state."
                            )
                        },
                    ),
                ],
            )
        )
        return

      combined = {fr.name: fr.response for fr in fn_responses}
      yield LlmResponse(
          content=types.Content(
              role="model",
              parts=[
                  types.Part.from_text(
                      text=(
                          "Parallel A2A execution completed in a single turn!\n"
                          + json.dumps(combined, indent=2)
                      )
                  )
              ],
          )
      )
      return

    yield LlmResponse(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=f"Processed: {sys_instruction}")],
        )
    )


def get_model(
    role_name: str,
    offline_mock: bool = False,
) -> str | BaseLlm:
  """Returns Vertex AI `gemini-2.5-flash` by default, or `OfflineDeterministicLlm` if requested."""
  if offline_mock or os.environ.get("OFFLINE_MOCK", "").lower() in (
      "1",
      "true",
  ):
    return OfflineDeterministicLlm(role_name=role_name)
  return os.environ.get("ADK_GEMINI_MODEL", "gemini-2.5-flash")
