# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for EngineStateClient (#251 step 5) — server-free, no Dynamo runtime, no GPU.

Tests the fetch/degrade/guard ORCHESTRATION with a fake async http client + a monkeypatched
in-engine build_paper_state (the real builder is covered by the sglang verify suite). The key
contract: ANY failure path returns None so the router degrades to its token-size proxy (do-no-harm),
and a healthy fetch builds via the in-engine build_paper_state with a synthetic MEMORY_PRESSURE event.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from dynamo.aginfer_router.engine_state import EngineStateClient

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class _Resp:
    def __init__(self, status_code, payload=None, raise_json=False):
        self.status_code = status_code
        self._payload = payload
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("bad json")
        return self._payload


class _FakeHTTP:
    """Records the call; returns a canned response (or raises). ``verbs`` makes the
    GET-vs-POST transport choice observable, which is the whole point of the two
    routes: the same dump arrives by different verbs."""
    def __init__(self, resp=None, raise_get=False):
        self._resp = resp
        self._raise_get = raise_get
        self.calls = []
        self.verbs = []
        self.posted = None

    async def get(self, url, timeout=None):
        self.calls.append((url, timeout))
        self.verbs.append("get")
        if self._raise_get:
            raise ConnectionError("unreachable")
        return self._resp

    async def post(self, url, json=None, timeout=None):  # noqa: A002 — httpx kwarg name
        self.calls.append((url, timeout))
        self.verbs.append("post")
        self.posted = json
        if self._raise_get:
            raise ConnectionError("unreachable")
        return self._resp


def _stub_sglang_build(monkeypatch, capture):
    """Install a fake sglang.srt.mem_cache.aginfer.{state_builder,events,program_tracker} so
    build() resolves without the real sglang package — captures the event kind passed in."""
    pkg = "sglang.srt.mem_cache.aginfer"
    sb = types.ModuleType(pkg + ".state_builder")
    ev = types.ModuleType(pkg + ".events")
    pt = types.ModuleType(pkg + ".program_tracker")

    class EventKind:
        MEMORY_PRESSURE = "memory_pressure"

    class Event:
        def __init__(self, kind=None):
            self.kind = kind

    class ProgramTracker:
        pass

    def build_paper_state(dump, *, event, tracker, unknown_tier_log):
        capture["event_kind"] = event.kind
        capture["dump"] = dump
        return "SCHED_STATE_SENTINEL"

    ev.EventKind = EventKind
    ev.Event = Event
    pt.ProgramTracker = ProgramTracker
    sb.build_paper_state = build_paper_state
    # ensure parent packages exist so the dotted import resolves
    for name in ["sglang", "sglang.srt", "sglang.srt.mem_cache", pkg]:
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, pkg + ".state_builder", sb)
    monkeypatch.setitem(sys.modules, pkg + ".events", ev)
    monkeypatch.setitem(sys.modules, pkg + ".program_tracker", pt)


def test_disabled_when_no_url():
    c = EngineStateClient(None)
    assert c.enabled is False


@pytest.mark.asyncio
async def test_fetch_disabled_returns_none():
    c = EngineStateClient(None)
    http = _FakeHTTP()
    assert await c.fetch(http) is None
    assert http.calls == []  # never even hits the network when disabled


@pytest.mark.asyncio
async def test_fetch_non_200_degrades():
    c = EngineStateClient("http://x/aginfer/state")
    assert await c.fetch(_FakeHTTP(_Resp(503))) is None


@pytest.mark.asyncio
async def test_fetch_network_error_degrades():
    c = EngineStateClient("http://x/aginfer/state")
    assert await c.fetch(_FakeHTTP(raise_get=True)) is None


@pytest.mark.asyncio
async def test_fetch_bad_json_degrades():
    c = EngineStateClient("http://x/aginfer/state")
    assert await c.fetch(_FakeHTTP(_Resp(200, raise_json=True))) is None


def test_build_non_dict_degrades():
    c = EngineStateClient("http://x/aginfer/state")
    assert c.build(["not", "a", "dict"]) is None
    assert c.build(None) is None


def test_build_unsupported_cache_degrades():
    c = EngineStateClient("http://x/aginfer/state")
    assert c.build({"unsupported_tree_cache": "VLLM"}) is None


def test_build_valid_dump_uses_in_engine_builder(monkeypatch):
    capture = {}
    _stub_sglang_build(monkeypatch, capture)
    c = EngineStateClient("http://x/aginfer/state")
    out = c.build({"units": {}, "pool_usage": {}, "time_counter": 0})
    assert out == "SCHED_STATE_SENTINEL"
    assert capture["event_kind"] == "memory_pressure"  # synthetic MEMORY_PRESSURE event


@pytest.mark.asyncio
async def test_fetch_200_builds(monkeypatch):
    capture = {}
    _stub_sglang_build(monkeypatch, capture)
    c = EngineStateClient("http://x/aginfer/state")
    out = await c.fetch(_FakeHTTP(_Resp(200, payload={"units": {}, "time_counter": 0})))
    assert out == "SCHED_STATE_SENTINEL"
    assert capture["event_kind"] == "memory_pressure"


def test_build_swallows_builder_exception(monkeypatch):
    # if the in-engine builder raises (e.g. fatal on a malformed-but-dict dump), degrade to None
    pkg = "sglang.srt.mem_cache.aginfer"
    sb = types.ModuleType(pkg + ".state_builder")
    ev = types.ModuleType(pkg + ".events")
    pt = types.ModuleType(pkg + ".program_tracker")
    ev.EventKind = type("EK", (), {"MEMORY_PRESSURE": "memory_pressure"})
    ev.Event = lambda kind=None: types.SimpleNamespace(kind=kind)
    pt.ProgramTracker = type("PT", (), {})
    def boom(*a, **k):
        raise RuntimeError("missing_state_field")
    sb.build_paper_state = boom
    for name in ["sglang", "sglang.srt", "sglang.srt.mem_cache", pkg]:
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name) or types.ModuleType(name))
    monkeypatch.setitem(sys.modules, pkg + ".state_builder", sb)
    monkeypatch.setitem(sys.modules, pkg + ".events", ev)
    monkeypatch.setitem(sys.modules, pkg + ".program_tracker", pt)
    c = EngineStateClient("http://x/aginfer/state")
    assert c.build({"units": {}}) is None


# --- transports and envelopes -------------------------------------------------
#
# ``python -m dynamo.sglang`` does not serve sglang's HTTP surface, so in a Dynamo
# stack the dump comes from the worker's system-status passthrough instead of
# ``GET /aginfer/state``. One configured URL has to pick the right verb, and the
# three envelopes the dump arrives in have to collapse to the same dict — when they
# do not, the value gate degrades to the size proxy without saying so, which is
# indistinguishable from the arm simply not working.

ENGINE_ROUTE = "http://x:9090/engine/call_tokenizer_manager"
DUMP = {"units": {}, "time_counter": 0}


def _engine_envelope(*dumps):
    """What ``call_tokenizer_manager`` returns: one dataclass-asdict'd
    ``GetAginferStateReqOutput`` per DP rank, the dump pre-serialised to JSON text."""
    return {"result": [{"state": None, "state_bytes": json.dumps(d)} for d in dumps]}


def test_transport_is_inferred_from_the_path():
    """One flag, not two: the operator configures a URL and the verb follows."""
    assert EngineStateClient(ENGINE_ROUTE).via_engine_route is True
    assert EngineStateClient("http://x/aginfer/state").via_engine_route is False
    assert EngineStateClient(None).via_engine_route is False


@pytest.mark.asyncio
async def test_engine_route_posts_the_method_name(monkeypatch):
    capture = {}
    _stub_sglang_build(monkeypatch, capture)
    c = EngineStateClient(ENGINE_ROUTE)
    http = _FakeHTTP(_Resp(200, payload=_engine_envelope(DUMP)))

    assert await c.fetch(http) == "SCHED_STATE_SENTINEL"
    assert http.verbs == ["post"]
    assert http.posted == {"method": "get_aginfer_state"}
    assert capture["dump"] == DUMP  # the builder sees the dump, not the envelope


@pytest.mark.asyncio
async def test_native_route_still_gets_the_bare_dump(monkeypatch):
    capture = {}
    _stub_sglang_build(monkeypatch, capture)
    c = EngineStateClient("http://x/aginfer/state")
    http = _FakeHTTP(_Resp(200, payload=DUMP))

    assert await c.fetch(http) == "SCHED_STATE_SENTINEL"
    assert http.verbs == ["get"]
    assert capture["dump"] == DUMP


@pytest.mark.asyncio
async def test_engine_route_falls_back_to_the_dict_form(monkeypatch):
    """``state_bytes`` is None whenever the cache could not pre-serialise."""
    capture = {}
    _stub_sglang_build(monkeypatch, capture)
    c = EngineStateClient(ENGINE_ROUTE)
    payload = {"result": [{"state": DUMP, "state_bytes": None}]}

    assert await c.fetch(_FakeHTTP(_Resp(200, payload=payload))) == "SCHED_STATE_SENTINEL"
    assert capture["dump"] == DUMP


@pytest.mark.asyncio
async def test_per_rank_envelope_is_unwrapped(monkeypatch):
    """sglang's own route wraps multi-rank dumps differently from the passthrough."""
    capture = {}
    _stub_sglang_build(monkeypatch, capture)
    c = EngineStateClient("http://x/aginfer/state")
    other = {"units": {"b": 1}, "time_counter": 9}

    await c.fetch(_FakeHTTP(_Resp(200, payload={"per_rank": [DUMP, other]})))
    assert capture["dump"] == DUMP


@pytest.mark.asyncio
async def test_multi_rank_narrowing_is_warned_about_exactly_once(monkeypatch, caplog):
    """Scoring rank 0 of several is a real narrowing; it must not be silent, and it
    must not spam once per scheduler tick either."""
    _stub_sglang_build(monkeypatch, {})
    c = EngineStateClient(ENGINE_ROUTE)
    http = _FakeHTTP(_Resp(200, payload=_engine_envelope(DUMP, DUMP)))

    with caplog.at_level("WARNING"):
        await c.fetch(http)
        await c.fetch(http)

    assert len([r for r in caplog.records if "DP ranks" in r.getMessage()]) == 1


@pytest.mark.asyncio
async def test_single_rank_is_not_warned_about(monkeypatch, caplog):
    _stub_sglang_build(monkeypatch, {})
    c = EngineStateClient(ENGINE_ROUTE)

    with caplog.at_level("WARNING"):
        await c.fetch(_FakeHTTP(_Resp(200, payload=_engine_envelope(DUMP))))

    assert not [r for r in caplog.records if "DP ranks" in r.getMessage()]


@pytest.mark.asyncio
async def test_unparseable_state_bytes_degrades():
    """Unwrapping runs on the fetch path, so its failures degrade like any other."""
    c = EngineStateClient(ENGINE_ROUTE)
    payload = {"result": [{"state": None, "state_bytes": "{ broken"}]}
    assert await c.fetch(_FakeHTTP(_Resp(200, payload=payload))) is None


@pytest.mark.asyncio
async def test_engine_route_failures_degrade_like_the_native_one():
    """The passthrough answers 404 when the worker was started without --enable-rl."""
    c = EngineStateClient(ENGINE_ROUTE)
    assert await c.fetch(_FakeHTTP(_Resp(404))) is None
    assert await c.fetch(_FakeHTTP(raise_get=True)) is None
