# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.aginfer_router.__main__ import ThunderAgentRouterHandler

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


async def _stream(*items):
    for item in items:
        yield item


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def data(self):
        return self._payload


class _ControlClient:
    def __init__(self, calls, failures=()):
        self.calls = calls
        self.failures = set(failures)

    async def direct(self, body, worker_id):
        self.calls.append(("direct", worker_id, body))
        status = "error" if worker_id in self.failures else "success"
        return _stream(_Response({"status": status, "message": "failed"}))


class _FinalScheduler:
    def __init__(self, worker_ids, calls):
        self.worker_ids = worker_ids
        self.calls = calls

    async def begin_end_program(self, program_id):
        self.calls.append(("lookup", program_id))
        return self.worker_ids

    async def end_program(self, program_id):
        self.calls.append(("release", program_id))
        return True


def _handler_for_final(worker_ids, *, failures=()):
    calls = []
    handler = ThunderAgentRouterHandler.__new__(ThunderAgentRouterHandler)
    handler._scheduler = _FinalScheduler(worker_ids, calls)
    handler._kv_router = object()
    handler._end_program_client = _ControlClient(calls, failures)
    return handler, calls


def test_worker_zero_is_not_lost_to_truthiness_fallback():
    handler, _ = _handler_for_final(())

    assert (
        handler._extract_worker_id(
            {
                "routing_data": {
                    "worker_id": {
                        "decode_worker_id": 0,
                        "prefill_worker_id": 7,
                    }
                }
            }
        )
        == 0
    )


@pytest.mark.asyncio
async def test_final_notifies_all_kv_workers_before_releasing_mapping():
    handler, calls = _handler_for_final((11, 22))

    chunks = [
        chunk
        async for chunk in handler.generate(
            {
                "token_ids": [],
                "agent_context": {
                    "session_id": "session-1",
                    "kv_hints": {"evict_session": True},
                },
            }
        )
    ]

    assert chunks == []
    assert {call[1] for call in calls if call[0] == "direct"} == {11, 22}
    assert calls[-1] == ("release", "session-1")


@pytest.mark.asyncio
async def test_final_keeps_mapping_when_worker_rejects_cleanup():
    handler, calls = _handler_for_final((11, 22), failures=(22,))

    with pytest.raises(RuntimeError, match="failed on 1/2 workers"):
        async for _ in handler.generate(
            {
                "token_ids": [],
                "agent_context": {"session_id": "session-1", "session_final": True},
            }
        ):
            pass

    assert {call[1] for call in calls if call[0] == "direct"} == {11, 22}
    assert not any(call[0] == "release" for call in calls)


@pytest.mark.asyncio
async def test_duplicate_final_after_release_is_idempotent():
    handler, calls = _handler_for_final(None)

    async for _ in handler.generate(
        {
            "token_ids": [],
            "agent_context": {"trajectory_id": "old-id", "trajectory_final": True},
        }
    ):
        pass

    assert calls == [("lookup", "old-id")]


@pytest.mark.asyncio
async def test_final_does_not_release_unattributed_program():
    handler, calls = _handler_for_final(())

    with pytest.raises(RuntimeError, match="original worker is unknown"):
        async for _ in handler.generate(
            {
                "token_ids": [],
                "agent_context": {"session_id": "session-1", "session_final": True},
            }
        ):
            pass

    assert calls == [("lookup", "session-1")]


@pytest.mark.asyncio
async def test_normal_request_normalizes_legacy_context_and_records_pd_workers():
    calls = []

    class Scheduler:
        async def before_request(
            self, program_id, estimated_prompt_tokens=0, parent_program_id=None
        ):
            calls.append(("before", program_id, estimated_prompt_tokens))
            return SimpleNamespace(priority_jump=0.0, assigned_worker_hint=None)

        async def assign_workers(self, program_id, **worker_ids):
            calls.append(("assign", program_id, worker_ids))

        def record_output_tokens(self, program_id, count):
            calls.append(("tokens", program_id, count))

        async def after_request(self, program_id, prompt_tokens, completion_tokens):
            calls.append(("after", program_id, prompt_tokens, completion_tokens))

    class KvRouter:
        async def generate_from_request(self, request):
            calls.append(("request", request))
            return _stream(
                {
                    "token_ids": [9],
                    "completion_usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                    },
                    "routing_data": {
                        "worker_id": {
                            "decode_worker_id": 11,
                            "prefill_worker_id": 22,
                        }
                    },
                }
            )

    handler = ThunderAgentRouterHandler.__new__(ThunderAgentRouterHandler)
    handler._scheduler = Scheduler()
    handler._kv_router = KvRouter()
    handler._end_program_client = object()
    handler._worker_id_extract_warned = False

    chunks = [
        chunk
        async for chunk in handler.generate(
            {
                "model": "m",
                "token_ids": [1, 2, 3],
                "agent_context": {
                    "trajectory_id": "trajectory-1",
                    "trajectory_final": False,
                },
            }
        )
    ]

    assert len(chunks) == 1
    forwarded = next(call[1] for call in calls if call[0] == "request")
    assert forwarded["agent_context"] == {"session_id": "trajectory-1"}
    assert (
        "assign",
        "trajectory-1",
        {"decode_worker_id": 11, "prefill_worker_id": 22},
    ) in calls
