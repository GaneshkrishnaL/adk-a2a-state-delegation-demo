# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Reusable A2A State Delegation & `output_key` Bridge for Google ADK
#
# Public Documentation References:
#   - ADK A2A Integration (`RemoteA2aAgent`, `to_a2a`, `A2aAgentExecutor`):
#     https://google.github.io/adk-docs/a2a/
#   - ADK Session State & `output_key`:
#     https://google.github.io/adk-docs/sessions/state/
#   - A2A Protocol Specification (Message & Request Metadata):
#     https://a2a-protocol.org/latest/specification/
#   - Vertex AI Agent Engine Memory Bank & Scoping (`user_id`):
#     https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/overview

from __future__ import annotations

from typing import Any, Callable

from a2a.server.agent_execution import RequestContext
from a2a.types import Message as A2AMessage
from google.adk.a2a.agent.config import RequestInterceptor
from google.adk.a2a.converters.part_converter import A2APartToGenAIPartConverter
from google.adk.a2a.converters.part_converter import convert_a2a_part_to_genai_part
from google.adk.a2a.converters.request_converter import AgentRunRequest
from google.adk.a2a.converters.request_converter import convert_a2a_request_to_agent_run_request
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event


def _strip_vertex_unsupported_part_metadata(parts: list[Any] | None) -> None:
  """Clears `Part.part_metadata` for Vertex AI (`GOOGLE_GENAI_USE_VERTEXAI=TRUE`).

  Why this is needed when using Gemini via Vertex AI:
    1. When `RemoteA2aAgent` serializes a user message into an A2A `Part`, it
       attaches A2A part-level metadata (`{"is_user_input": True}`).
    2. When converted back to `google.genai.types.Part`, `part_metadata` may be
       populated on the `Part`.
    3. Unlike the Gemini Developer API (AI Studio), the Vertex AI
       `generateContent` serializer (`_Part_to_vertex` in `google-genai`)
       rejects `part_metadata` with:
         `ValueError: part_metadata parameter is only supported in Gemini
          Developer API mode, not in Gemini Enterprise Agent Platform mode.`
    Clearing `part.part_metadata = None` at the A2A boundary ensures seamless
    compatibility with Vertex AI across all `google-adk` and `google-genai`
    versions.
  """
  if not parts:
    return
  for part in parts:
    if part is not None and getattr(part, "part_metadata", None) is not None:
      part.part_metadata = None


def _extract_metadata_dict(
    request: RequestContext,
    run_req: AgentRunRequest,
) -> dict[str, Any]:
  """Extracts A2A request metadata as a plain dictionary using public ADK fields."""
  if (
      run_req.run_config
      and isinstance(run_req.run_config.custom_metadata, dict)
      and isinstance(
          run_req.run_config.custom_metadata.get("a2a_metadata"), dict
      )
  ):
    return run_req.run_config.custom_metadata["a2a_metadata"]
  raw_meta = getattr(request, "metadata", None)
  if isinstance(raw_meta, dict):
    return raw_meta
  return {}


def make_state_forwarding_meta_provider(
    state_keys: list[str],
) -> Callable[[InvocationContext, A2AMessage], dict[str, Any]]:
  """Creates an `a2a_request_meta_provider` for `RemoteA2aAgent`.

  Why this is needed:
    In ADK, `session.state` is shared automatically among in-process sub-agents
    (`sub_agents=[...]`), but does NOT automatically cross an A2A HTTP boundary
    because each remote A2A server manages its own isolated `SessionService`.
    See: https://google.github.io/adk-docs/sessions/state/

    This provider selectively extracts `state_keys` from the Orchestrator's
    `ctx.session.state` and attaches them to the outgoing A2A request metadata
    under `"state_delta"`, along with the authenticated `user_id`.
  """

  def _provider(
      ctx: InvocationContext, a2a_request: A2AMessage
  ) -> dict[str, Any]:
    del a2a_request
    forwarded_state = {
        key: ctx.session.state[key]
        for key in state_keys
        if key in ctx.session.state
    }
    payload = {
        "authenticated_user_id": ctx.user_id,
        "state_delta": forwarded_state,
    }
    print(
        f"\n  [A2A Client -> Wire] Forwarding user_id='{ctx.user_id}' and"
        f" state keys {list(forwarded_state.keys())} to remote agent:"
    )
    for k, v in forwarded_state.items():
      print(f"    • {k} = {v!r}")
    return payload

  return _provider


def make_output_key_interceptor(output_key: str) -> RequestInterceptor:
  """Gives `RemoteA2aAgent` native `output_key` behavior on the Orchestrator.

  Why this is needed:
    `LlmAgent` supports `output_key="..."`, which automatically saves the
    agent's final text response into `session.state[output_key]` so downstream
    agents can reference `{output_key}` in their instructions or via
    `tool_context.state[output_key]`.
    See: https://google.github.io/adk-docs/sessions/state/#output-key

    Because `RemoteA2aAgent` inherits from `BaseAgent` (not `LlmAgent`), it
    does not expose an `output_key` constructor parameter directly. Attaching
    this `RequestInterceptor(after_request=...)` via `A2aRemoteAgentConfig`
    captures the remote agent's completed response and writes it into
    `event.actions.state_delta[output_key]` and `ctx.session.state[output_key]`
    on the Orchestrator.
  """

  async def _after_request(
      ctx: InvocationContext,
      a2a_response: Any,
      event: Event,
  ) -> Event | None:
    del a2a_response
    if not event or event.partial or not event.content or not event.content.parts:
      return event

    # Strip `part_metadata` so downstream Vertex AI LlmAgents on the
    # Orchestrator never fail on A2A part metadata
    _strip_vertex_unsupported_part_metadata(event.content.parts)

    # Ignore intermediate streaming chunks or failed/canceled task states
    response_meta = (event.custom_metadata or {}).get("a2a:response")
    if isinstance(response_meta, dict):
      status = response_meta.get("status")
      if isinstance(status, dict) and status.get("state") in {
          "submitted",
          "working",
          "input-required",
          "auth-required",
          "failed",
          "rejected",
          "canceled",
          "TASK_STATE_SUBMITTED",
          "TASK_STATE_WORKING",
          "TASK_STATE_INPUT_REQUIRED",
          "TASK_STATE_AUTH_REQUIRED",
          "TASK_STATE_FAILED",
          "TASK_STATE_REJECTED",
          "TASK_STATE_CANCELED",
      }:
        return event

    text_chunks = [
        part.text
        for part in event.content.parts
        if part.text
        and not part.thought
        and not part.function_call
        and not part.function_response
    ]
    if text_chunks:
      full_text = "".join(text_chunks).strip()
      event.actions.state_delta[output_key] = full_text
      ctx.session.state[output_key] = full_text
      print(
          f"\n  [A2A Wire -> Orchestrator State] Saved RemoteA2aAgent"
          f" '{event.author}' response to session.state['{output_key}']:"
      )
      print(f"    • session.state['{output_key}'] = {full_text!r}")
    return event

  return RequestInterceptor(after_request=_after_request)


def state_aware_request_converter(
    request: RequestContext,
    part_converter: A2APartToGenAIPartConverter = convert_a2a_part_to_genai_part,
) -> AgentRunRequest:
  """Server-side `request_converter` for `A2aAgentExecutorConfig`.

  Responsibilities:
    1. Strips `part_metadata` from converted `genai.types.Part` objects so
       Vertex AI (`GOOGLE_GENAI_USE_VERTEXAI=TRUE`) never raises a `ValueError`.
    2. Propagates `authenticated_user_id` from A2A metadata into
       `AgentRunRequest.user_id` instead of falling back to the synthetic
       `A2A_USER_<contextId>` ID. This ensures Vertex AI Agent Engine Memory
       Bank (`scope={"user_id": ...}`) resolves the caller's true user scope:
       https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/memory-bank/overview
    3. Extracts `state_delta` from A2A request metadata and assigns it to
       `AgentRunRequest.state_delta`, which `A2aAgentExecutor` passes directly
       to `runner.run_async(..., state_delta=...)` on the remote server so the
       remote agent's `session.state` is populated before execution begins.
  """
  run_req = convert_a2a_request_to_agent_run_request(request, part_converter)

  if run_req.new_message and run_req.new_message.parts:
    _strip_vertex_unsupported_part_metadata(run_req.new_message.parts)

  metadata = _extract_metadata_dict(request, run_req)

  if "authenticated_user_id" in metadata and metadata["authenticated_user_id"]:
    run_req.user_id = str(metadata["authenticated_user_id"])

  incoming_state = metadata.get("state_delta")
  if isinstance(incoming_state, dict) and incoming_state:
    run_req.state_delta = dict(incoming_state)
    print(
        f"\n  [Remote A2A Server] Injected incoming state_delta into remote"
        f" session (user_id='{run_req.user_id}',"
        f" session_id='{run_req.session_id}'):"
    )
    for k, v in run_req.state_delta.items():
      print(f"    • remote session.state['{k}'] = {v!r}")

  return run_req
