# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small helpers for forwarding agent lifecycle metadata to backends."""

from __future__ import annotations

from typing import Any, Mapping, Optional


def extract_program_id(request: Mapping[str, Any]) -> Optional[str]:
    """Return the stable agent program id, preferring the canonical field.

    ``trajectory_id`` is retained as an aginfer compatibility alias for older
    replay producers.  The current Dynamo wire contract uses ``session_id``.
    """

    context = request.get("agent_context")
    if not isinstance(context, Mapping):
        return None
    for field in ("session_id", "trajectory_id"):
        value = context.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def is_program_final(request: Mapping[str, Any]) -> bool:
    """Whether a request explicitly ends the program's KV lifetime."""

    context = request.get("agent_context")
    if not isinstance(context, Mapping):
        return False
    kv_hints = context.get("kv_hints")
    evict_session = (
        bool(kv_hints.get("evict_session")) if isinstance(kv_hints, Mapping) else False
    )
    return bool(
        context.get("session_final") or evict_session or context.get("trajectory_final")
    )


def canonical_agent_context(request: Mapping[str, Any]) -> Any:
    """Normalize legacy trajectory fields to Dynamo's current wire schema.

    The Rust ``PreprocessedRequest`` deserializer intentionally rejects unknown
    AgentContext fields.  Removing the two legacy aliases here lets old replay
    producers keep working while every downstream worker sees ``session_id`` /
    ``session_final``.
    """

    context = request.get("agent_context")
    if not isinstance(context, Mapping):
        return context

    normalized = dict(context)
    program_id = extract_program_id(request)
    if program_id is not None:
        normalized["session_id"] = program_id
    if is_program_final(request):
        normalized["session_final"] = True
    normalized.pop("trajectory_id", None)
    normalized.pop("trajectory_final", None)
    return normalized


__all__ = ["canonical_agent_context", "extract_program_id", "is_program_final"]
