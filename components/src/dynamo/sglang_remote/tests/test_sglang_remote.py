# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline unit tests for dynamo.sglang_remote (no GPU / no live SGLang)."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest

from dynamo.sglang_remote.args import RemoteConfig, parse_args
from dynamo.sglang_remote.client import SglangHttpClient
from dynamo.sglang_remote.handler import (
    RemoteWorkerHandler,
    _delta_token_ids,
    build_generate_body,
    build_sampling_params,
)
from dynamo.sglang_remote.register import (
    build_runtime_config,
    resolve_kv_zmq_plan,
)


class _FakeContext:
    def __init__(self) -> None:
        self._stopped = False

    def is_stopped(self) -> bool:
        return self._stopped

    def stop(self) -> None:
        self._stopped = True


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json_body: Any = None,
        text: str = "",
        lines: Optional[List[str]] = None,
    ) -> None:
        self.status_code = status_code
        self._json_body = json_body
        self.text = text
        self._lines = lines or []

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                f"status {self.status_code}",
                request=httpx.Request("GET", "http://x"),
                response=httpx.Response(self.status_code),
            )

    def json(self) -> Any:
        return self._json_body

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class _FakeHttpx:
    """Minimal AsyncClient stand-in for SglangHttpClient tests."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.server_info = {
            "page_size": 64,
            "max_total_num_tokens": 128000,
            "context_length": 98304,
            "max_running_requests": 64,
            "dp_size": 1,
            "incremental_streaming_output": True,
            "kv_events": {
                "publisher": "zmq",
                "endpoint_host": "*",
                "endpoint_port_base": 5557,
                "block_size": 64,
                "dp_size": 1,
            },
        }
        self.stream_lines: List[str] = [
            'data: {"output_ids":[1,2],"meta_info":{"id":"rid-1"}}',
            'data: {"output_ids":[3],"meta_info":{"id":"rid-1","finish_reason":{"type":"stop"},"prompt_tokens":10,"completion_tokens":3}}',
            "data: [DONE]",
        ]
        self.session_end_body = {
            "ok": True,
            "program_id": "p1",
            "per_rank": [
                {
                    "ok": True,
                    "deferred": False,
                    "dp_rank": 0,
                }
            ],
        }
        self.events_body = {"ok": True, "applied": 1, "skipped": 0, "ranks": 1}
        self.abort_calls: List[Dict[str, Any]] = []

    async def get(self, path: str, **kwargs):
        self.calls.append({"method": "GET", "path": path, **kwargs})
        return _FakeResponse(json_body=self.server_info)

    async def put(self, path: str, **kwargs):
        self.calls.append({"method": "PUT", "path": path, **kwargs})
        return _FakeResponse(json_body=self.events_body)

    async def post(self, path: str, **kwargs):
        self.calls.append({"method": "POST", "path": path, **kwargs})
        if path == "/abort_request":
            self.abort_calls.append(kwargs.get("json") or {})
            return _FakeResponse(status_code=200, json_body={})
        if path == "/aginfer/session_end":
            return _FakeResponse(json_body=self.session_end_body)
        return _FakeResponse(status_code=404, text="nope")

    def stream(self, method: str, path: str, **kwargs):
        self.calls.append({"method": method, "path": path, "stream": True, **kwargs})
        return _FakeResponse(lines=self.stream_lines)

    async def aclose(self) -> None:
        return None


def test_parse_args_requires_url(monkeypatch):
    with pytest.raises(SystemExit):
        parse_args([])


def test_remote_config_validate():
    cfg = RemoteConfig(
        sglang_url="http://10.0.0.5:30000/",
        model_path="/models/x",
    )
    # constructor doesn't strip; parse_args does. validate only checks scheme.
    cfg.validate()
    assert cfg.generate_endpoint == "dynamo.backend.generate"
    assert cfg.resolved_kv_events_host() == "10.0.0.5"


def test_build_sampling_params_forced_ids():
    req = {
        "sampling_options": {"temperature": 0.0, "top_p": 1.0},
        "stop_conditions": {"max_tokens": 5, "ignore_eos": True},
        "extra_args": {"forced_output_ids": [9, 8, 7]},
    }
    sp = build_sampling_params(req)
    assert sp["temperature"] == 0.0
    assert sp["max_new_tokens"] == 5
    assert sp["ignore_eos"] is True
    assert sp["custom_params"]["forced_output_ids"] == [9, 8, 7]


def test_build_generate_body_program_id():
    req = {
        "token_ids": [1, 2, 3],
        "sampling_options": {},
        "stop_conditions": {"max_tokens": 2},
        "agent_context": {"session_id": "prog#salt"},
        "extra_args": {"forced_output_ids": [4]},
    }
    body = build_generate_body(req)
    assert body["input_ids"] == [1, 2, 3]
    assert body["program_id"] == "prog#salt"
    assert body["sampling_params"]["custom_params"]["forced_output_ids"] == [4]
    assert body["stream"] is True


def test_build_generate_body_rejects_disagg():
    with pytest.raises(RuntimeError, match="disaggregated"):
        build_generate_body(
            {
                "token_ids": [1],
                "bootstrap_info": {"bootstrap_host": "x"},
            }
        )


def test_delta_token_ids_cumulative_and_incremental():
    d0, n0 = _delta_token_ids([1, 2, 3], incremental=False, prev_len=0)
    assert d0 == [1, 2, 3] and n0 == 3
    d1, n1 = _delta_token_ids([1, 2, 3, 4], incremental=False, prev_len=n0)
    assert d1 == [4] and n1 == 4
    d2, n2 = _delta_token_ids([5], incremental=True, prev_len=0)
    assert d2 == [5] and n2 == 1


@pytest.mark.asyncio
async def test_client_server_info_and_events():
    fake = _FakeHttpx()
    client = SglangHttpClient("http://remote:30000", client=fake)  # type: ignore[arg-type]
    info = await client.get_server_info()
    assert info["page_size"] == 64
    ev = await client.put_aginfer_events(
        [{"kind": "tool_call_start", "session": "p1", "payload": {}}]
    )
    assert ev["ok"] is True
    put_calls = [c for c in fake.calls if c["method"] == "PUT"]
    assert put_calls[0]["path"] == "/aginfer/events"
    assert put_calls[0]["json"]["events"][0]["kind"] == "tool_call_start"


@pytest.mark.asyncio
async def test_handler_generate_incremental_stream():
    fake = _FakeHttpx()
    client = SglangHttpClient("http://remote:30000", client=fake)  # type: ignore[arg-type]
    handler = RemoteWorkerHandler(client, incremental_streaming_output=True)
    req = {
        "token_ids": [10, 11],
        "sampling_options": {"temperature": 0.0},
        "stop_conditions": {"max_tokens": 3, "ignore_eos": True},
        "extra_args": {
            "forced_output_ids": [1, 2, 3],
            "aginfer_events": [
                {"kind": "llm_prefill", "session": "s1", "payload": {}}
            ],
        },
        "agent_context": {"session_id": "s1"},
    }
    outs = []
    async for chunk in handler.generate(req, _FakeContext()):
        outs.append(chunk)
    # Let the fire-and-forget events task run.
    await asyncio.sleep(0.05)
    assert len(outs) == 2
    assert outs[0]["token_ids"] == [1, 2]
    assert outs[1]["token_ids"] == [3]
    assert outs[1]["finish_reason"] is not None
    assert outs[1]["completion_usage"]["completion_tokens"] == 3
    # generate body used stream=true and carried forced ids + program_id
    stream_calls = [c for c in fake.calls if c.get("stream")]
    assert stream_calls
    body = stream_calls[0]["json"]
    assert body["program_id"] == "s1"
    assert body["sampling_params"]["custom_params"]["forced_output_ids"] == [1, 2, 3]
    # events were pushed
    assert any(c["method"] == "PUT" for c in fake.calls)


@pytest.mark.asyncio
async def test_handler_generate_cumulative_stream():
    fake = _FakeHttpx()
    fake.stream_lines = [
        'data: {"output_ids":[1,2],"meta_info":{"id":"r"}}',
        'data: {"output_ids":[1,2,3],"meta_info":{"id":"r","finish_reason":{"type":"length"},"prompt_tokens":1,"completion_tokens":3}}',
        "data: [DONE]",
    ]
    client = SglangHttpClient("http://remote:30000", client=fake)  # type: ignore[arg-type]
    handler = RemoteWorkerHandler(client, incremental_streaming_output=False)
    outs = []
    async for chunk in handler.generate(
        {"token_ids": [1], "sampling_options": {}, "stop_conditions": {}},
        _FakeContext(),
    ):
        outs.append(chunk)
    assert outs[0]["token_ids"] == [1, 2]
    assert outs[1]["token_ids"] == [3]


@pytest.mark.asyncio
async def test_handler_end_program_success():
    fake = _FakeHttpx()
    client = SglangHttpClient("http://remote:30000", client=fake)  # type: ignore[arg-type]
    handler = RemoteWorkerHandler(client)
    outs = []
    async for chunk in handler.end_program({"program_id": "p1"}):
        outs.append(chunk)
    assert outs[0]["status"] == "success"
    post = [c for c in fake.calls if c["path"] == "/aginfer/session_end"][0]
    assert post["json"]["program_id"] == "p1"


@pytest.mark.asyncio
async def test_handler_end_program_incomplete():
    fake = _FakeHttpx()
    fake.session_end_body = {
        "ok": False,
        "per_rank": [{"ok": True, "deferred": True, "dp_rank": 0}],
    }
    client = SglangHttpClient("http://remote:30000", client=fake)  # type: ignore[arg-type]
    handler = RemoteWorkerHandler(client)
    outs = []
    async for chunk in handler.end_program({"program_id": "p1"}):
        outs.append(chunk)
    assert outs[0]["status"] == "error"


def test_build_runtime_config_and_kv_plan():
    info = {
        "page_size": 64,
        "max_total_num_tokens": 128000,
        "context_length": 98304,
        "max_running_requests": 32,
        "dp_size": 2,
        "kv_events": {
            "publisher": "zmq",
            "endpoint_host": "*",
            "endpoint_port_base": 5557,
            "block_size": 64,
            "dp_size": 2,
        },
    }
    cfg = RemoteConfig(
        sglang_url="http://10.1.2.3:30000",
        model_path="/m",
    )
    rt = build_runtime_config(info, cfg, kv_event_publishing_enabled=True)
    assert rt.total_kv_blocks == 128000 // 64
    assert rt.context_length == 98304
    # data_parallel_size is write-only on this binding; assignment is covered
    # by build_runtime_config not raising.
    assert rt.kv_event_publishing_enabled is True
    assert rt.max_num_seqs == 32

    plan = resolve_kv_zmq_plan(info, cfg)
    assert plan == ("10.1.2.3", 5557, 2, 64)

    cfg2 = RemoteConfig(
        sglang_url="http://10.1.2.3:30000",
        model_path="/m",
        disable_kv_events=True,
    )
    assert resolve_kv_zmq_plan(info, cfg2) is None
