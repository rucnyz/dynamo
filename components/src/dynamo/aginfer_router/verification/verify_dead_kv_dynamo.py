#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end verifier for Dynamo -> SGLang Dead-KV reclamation.

The script intentionally uses only the Python standard library.  It expects:

* a Dynamo OpenAI frontend (default: http://127.0.0.1:8000), and
* the corresponding SGLang worker system server with ``--enable-rl``
  (default: http://127.0.0.1:8081).

It flushes the worker cache at the beginning, so run it against a dedicated
test deployment.  Every HTTP response and parsed state snapshot is written to
a per-run JSON artifact directory, including failures.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Mapping, Sequence

# These endpoints are normally loopback or cluster-local.  Ignore ambient
# corporate HTTP_PROXY settings so a local verification request cannot escape
# through a forward proxy.
_HTTP_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class VerificationError(RuntimeError):
    """Raised when an observed pipeline invariant does not hold."""


@dataclasses.dataclass
class HttpResult:
    method: str
    url: str
    status: int | None
    elapsed_seconds: float
    headers: dict[str, str]
    body_text: str
    body_json: Any = None
    transport_error: str | None = None
    body_size_bytes: int = 0
    body_sha256: str = ""

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class Artifacts:
    def __init__(self, root: pathlib.Path, run_id: str) -> None:
        self.run_dir = root / run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.counter = 0
        self.summary: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "started_at": utc_now(),
            "status": "running",
            "checks": [],
            "files": [],
        }
        self.save_summary()

    def _next_path(self, kind: str, label: str) -> pathlib.Path:
        self.counter += 1
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "event"
        return self.run_dir / f"{self.counter:03d}_{kind}_{safe}.json"

    @staticmethod
    def _write(path: pathlib.Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
        temporary.replace(path)

    def write(self, kind: str, label: str, value: Any) -> pathlib.Path:
        path = self._next_path(kind, label)
        self._write(path, value)
        self.summary["files"].append(path.name)
        self.save_summary()
        return path

    def record_http(self, label: str, result: HttpResult) -> pathlib.Path:
        return self.write("http", label, result.as_dict())

    def record_state(
        self,
        label: str,
        result: HttpResult,
        states: Sequence[Mapping[str, Any]],
        analysis: Mapping[str, Any],
    ) -> pathlib.Path:
        # The raw HTTP body contains state_bytes and would duplicate the parsed
        # snapshot.  Keep its hash/metadata while preserving the parsed states.
        http_metadata = result.as_dict()
        http_metadata["body_text"] = "<stored as parsed states>"
        http_metadata["body_json"] = None
        return self.write(
            "state",
            label,
            {
                "http": http_metadata,
                "analysis": analysis,
                "states": list(states),
            },
        )

    def check(self, name: str, passed: bool, **details: Any) -> None:
        entry = {"name": name, "passed": bool(passed), **details}
        self.summary["checks"].append(entry)
        self.save_summary()
        if not passed:
            raise VerificationError(f"check failed: {name}: {details}")

    def save_summary(self) -> None:
        self._write(self.run_dir / "summary.json", self.summary)

    def finish(self, status: str, **details: Any) -> None:
        self.summary.update(details)
        self.summary["status"] = status
        self.summary["finished_at"] = utc_now()
        self.save_summary()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def compact_text(value: str, limit: int = 8000) -> str:
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    return value[:limit] + f"\n... <{omitted} characters omitted from artifact>"


def http_json(
    method: str,
    url: str,
    *,
    payload: Any = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 180.0,
    tolerate_transport_error: bool = False,
) -> HttpResult:
    request_headers = {
        "Accept": "application/json",
        "User-Agent": "deadkv-dynamo-e2e/1",
    }
    if headers:
        request_headers.update(headers)
    data = None
    if payload is not None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(
        url=url,
        data=data,
        headers=request_headers,
        method=method.upper(),
    )
    started = time.monotonic()
    status: int | None = None
    response_headers: dict[str, str] = {}
    body = b""
    transport_error: str | None = None
    try:
        with _HTTP_OPENER.open(request, timeout=timeout) as response:
            status = response.status
            response_headers = dict(response.headers.items())
            body = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_headers = dict(exc.headers.items()) if exc.headers else {}
        body = exc.read()
    except Exception as exc:  # final-control requests may close an empty stream
        transport_error = f"{type(exc).__name__}: {exc}"
        if not tolerate_transport_error:
            raise VerificationError(
                f"{method} {url} failed: {transport_error}"
            ) from exc
    elapsed = time.monotonic() - started
    body_text = body.decode("utf-8", errors="replace")
    body_json: Any = None
    if body_text.strip():
        try:
            body_json = json.loads(body_text)
        except json.JSONDecodeError:
            pass
    return HttpResult(
        method=method.upper(),
        url=url,
        status=status,
        elapsed_seconds=elapsed,
        headers=response_headers,
        body_text=compact_text(body_text),
        body_json=body_json,
        transport_error=transport_error,
        body_size_bytes=len(body),
        body_sha256=hashlib.sha256(body).hexdigest(),
    )


def require_http_ok(result: HttpResult, label: str) -> None:
    if not result.ok:
        raise VerificationError(
            f"{label} returned status={result.status}, "
            f"transport_error={result.transport_error!r}, body={result.body_text!r}"
        )


def require_control_ok(result: HttpResult, label: str) -> None:
    require_http_ok(result, label)
    payload = result.body_json
    if isinstance(payload, Mapping):
        if payload.get("success") is False or payload.get("status") == "error":
            raise VerificationError(f"{label} rejected the operation: {payload!r}")


def join_url(base: str, path: str) -> str:
    return base.rstrip("/") + "/" + path.lstrip("/")


def discover_model(
    frontend_url: str,
    headers: Mapping[str, str],
    timeout: float,
    artifacts: Artifacts,
) -> str:
    result = http_json(
        "GET",
        join_url(frontend_url, "/v1/models"),
        headers=headers,
        timeout=timeout,
    )
    artifacts.record_http("discover_models", result)
    require_http_ok(result, "model discovery")
    payload = result.body_json
    candidates = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(candidates, list):
        raise VerificationError(f"unexpected /v1/models payload: {payload!r}")
    for item in candidates:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            model = item["id"].strip()
            if model:
                return model
    raise VerificationError("/v1/models returned no usable model id")


def call_tokenizer_manager(
    worker_url: str,
    method: str,
    *,
    timeout: float,
) -> HttpResult:
    return http_json(
        "POST",
        join_url(worker_url, "/engine/call_tokenizer_manager"),
        payload={"method": method},
        timeout=timeout,
    )


def parse_state_response(result: HttpResult) -> list[dict[str, Any]]:
    require_http_ok(result, "get_aginfer_state")
    payload = result.body_json
    if not isinstance(payload, dict):
        raise VerificationError(
            f"state endpoint did not return a JSON object: {result.body_text!r}"
        )
    rank_items = payload.get("result")
    if isinstance(rank_items, dict):
        rank_items = [rank_items]
    if not isinstance(rank_items, list) or not rank_items:
        raise VerificationError(
            "state endpoint must return a non-empty result[]; was the SGLang "
            "worker started with --enable-rl and the aginfer-enabled source?"
        )
    states: list[dict[str, Any]] = []
    for rank, item in enumerate(rank_items):
        state: Any = None
        if isinstance(item, dict):
            raw = item.get("state_bytes")
            if isinstance(raw, str) and raw.strip():
                try:
                    state = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise VerificationError(
                        f"rank {rank} state_bytes is not valid JSON: {exc}"
                    ) from exc
            elif isinstance(item.get("state"), dict):
                state = item["state"]
        elif isinstance(item, str):
            try:
                state = json.loads(item)
            except json.JSONDecodeError as exc:
                raise VerificationError(
                    f"rank {rank} state string is not valid JSON: {exc}"
                ) from exc
        if not isinstance(state, dict):
            raise VerificationError(f"rank {rank} has no usable state: {item!r}")
        if not isinstance(state.get("units"), list):
            raise VerificationError(
                f"rank {rank} state has no units[]; aginfer state unsupported: {state!r}"
            )
        states.append(state)
    return states


def tier_bytes(unit: Mapping[str, Any]) -> dict[str, int]:
    totals: dict[str, int] = {}
    by_tier = unit.get("n_bytes")
    if not isinstance(by_tier, Mapping):
        return totals
    for tier, subpools in by_tier.items():
        if not isinstance(tier, str) or not isinstance(subpools, Mapping):
            continue
        total = 0
        for amount in subpools.values():
            if isinstance(amount, (int, float)) and not isinstance(amount, bool):
                total += int(amount)
        totals[tier] = total
    return totals


def analyze_states(
    states: Sequence[Mapping[str, Any]], program_a: str, program_b: str
) -> dict[str, Any]:
    ranks: list[dict[str, Any]] = []
    for rank_index, state in enumerate(states):
        units = state.get("units", [])
        physical: dict[str, int] = {}
        a_hashes: set[str] = set()
        b_hashes: set[str] = set()
        shared_hashes: set[str] = set()
        a_only_hashes: set[str] = set()
        b_only_hashes: set[str] = set()
        hit_counts: dict[str, int] = {}
        for unit in units:
            if not isinstance(unit, Mapping):
                continue
            unit_hash = str(unit.get("hash"))
            holders_raw = unit.get("session_ids")
            holders = (
                {str(value) for value in holders_raw}
                if isinstance(holders_raw, list)
                else set()
            )
            has_a = program_a in holders
            has_b = program_b in holders
            if has_a:
                a_hashes.add(unit_hash)
            if has_b:
                b_hashes.add(unit_hash)
            if has_a and has_b:
                shared_hashes.add(unit_hash)
            elif has_a:
                a_only_hashes.add(unit_hash)
            elif has_b:
                b_only_hashes.add(unit_hash)
            hit_count = unit.get("hit_count")
            if isinstance(hit_count, int) and not isinstance(hit_count, bool):
                hit_counts[unit_hash] = hit_count
            for tier, amount in tier_bytes(unit).items():
                physical[tier] = physical.get(tier, 0) + amount
        ranks.append(
            {
                "rank": rank_index,
                "unit_count": len(units),
                "physical_bytes": physical,
                "physical_bytes_total": sum(physical.values()),
                "all_hashes": sorted(
                    str(unit.get("hash")) for unit in units if isinstance(unit, Mapping)
                ),
                "a_hashes": sorted(a_hashes),
                "b_hashes": sorted(b_hashes),
                "shared_hashes": sorted(shared_hashes),
                "a_only_hashes": sorted(a_only_hashes),
                "b_only_hashes": sorted(b_only_hashes),
                "hit_counts": hit_counts,
                "per_program_a": (
                    (state.get("per_program_usage") or {}).get(program_a)
                    if isinstance(state.get("per_program_usage"), Mapping)
                    else None
                ),
                "per_program_b": (
                    (state.get("per_program_usage") or {}).get(program_b)
                    if isinstance(state.get("per_program_usage"), Mapping)
                    else None
                ),
            }
        )
    return {
        "rank_count": len(ranks),
        "program_a": program_a,
        "program_b": program_b,
        "ranks": ranks,
        "physical_bytes_total": sum(r["physical_bytes_total"] for r in ranks),
    }


def fetch_states(
    worker_url: str,
    *,
    program_a: str,
    program_b: str,
    timeout: float,
) -> tuple[HttpResult, list[dict[str, Any]], dict[str, Any]]:
    result = call_tokenizer_manager(worker_url, "get_aginfer_state", timeout=timeout)
    states = parse_state_response(result)
    analysis = analyze_states(states, program_a, program_b)
    return result, states, analysis


def all_ranks(
    analysis: Mapping[str, Any], predicate: Callable[[Mapping[str, Any]], bool]
) -> bool:
    ranks = analysis.get("ranks")
    return bool(ranks) and all(predicate(rank) for rank in ranks)


def same_rank_count(*analyses: Mapping[str, Any]) -> bool:
    return len({analysis.get("rank_count") for analysis in analyses}) == 1


def poll_states(
    label: str,
    worker_url: str,
    *,
    program_a: str,
    program_b: str,
    timeout: float,
    request_timeout: float,
    interval: float,
    predicate: Callable[[Mapping[str, Any]], bool],
    artifacts: Artifacts,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    deadline = time.monotonic() + timeout
    attempts: list[dict[str, Any]] = []
    last_result: HttpResult | None = None
    last_states: list[dict[str, Any]] = []
    last_analysis: dict[str, Any] = {}
    while True:
        last_result, last_states, last_analysis = fetch_states(
            worker_url,
            program_a=program_a,
            program_b=program_b,
            timeout=request_timeout,
        )
        matched = bool(predicate(last_analysis))
        attempts.append(
            {
                "at": utc_now(),
                "matched": matched,
                "physical_bytes_total": last_analysis.get("physical_bytes_total"),
                "rank_summaries": [
                    {
                        "rank": rank["rank"],
                        "a_units": len(rank["a_hashes"]),
                        "b_units": len(rank["b_hashes"]),
                        "shared_units": len(rank["shared_hashes"]),
                        "a_only_units": len(rank["a_only_hashes"]),
                        "b_only_units": len(rank["b_only_hashes"]),
                    }
                    for rank in last_analysis.get("ranks", [])
                ],
            }
        )
        if matched:
            artifacts.record_state(label, last_result, last_states, last_analysis)
            artifacts.write("poll", label, {"attempts": attempts})
            return last_states, last_analysis
        if time.monotonic() >= deadline:
            artifacts.record_state(
                label + "_timeout", last_result, last_states, last_analysis
            )
            artifacts.write("poll", label + "_timeout", {"attempts": attempts})
            raise VerificationError(
                f"timed out after {timeout:.1f}s waiting for state condition {label}"
            )
        time.sleep(interval)


def make_prompts(shared_chunks: int, tail_chunks: int) -> tuple[str, str]:
    shared = "\n".join(
        f"Common evidence block {index:04d}: cedar river cobalt lantern "
        "keeps this exact prefix stable for cache verification."
        for index in range(shared_chunks)
    )
    tail_a = "\n".join(
        f"Alpha-only branch {index:04d}: amber falcon maple quartz belongs only to A."
        for index in range(tail_chunks)
    )
    tail_b = "\n".join(
        f"Beta-only branch {index:04d}: bronze heron willow sapphire belongs only to B."
        for index in range(tail_chunks)
    )
    suffix = "\nReply with exactly the word OK."
    return shared + "\n\n" + tail_a + suffix, shared + "\n\n" + tail_b + suffix


def chat_payload(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }


def send_chat(
    frontend_url: str,
    payload: Mapping[str, Any],
    *,
    program_id: str,
    base_headers: Mapping[str, str],
    timeout: float,
    final: bool = False,
) -> HttpResult:
    headers = dict(base_headers)
    headers["x-dynamo-session-id"] = program_id
    if final:
        headers["x-dynamo-session-final"] = "true"
    return http_json(
        "POST",
        join_url(frontend_url, "/v1/chat/completions"),
        payload=payload,
        headers=headers,
        timeout=timeout,
        tolerate_transport_error=final,
    )


def cached_tokens_from_response(response: HttpResult) -> int | None:
    payload = response.body_json
    if not isinstance(payload, Mapping):
        return None
    usage = payload.get("usage")
    if not isinstance(usage, Mapping):
        return None
    details = usage.get("prompt_tokens_details")
    if isinstance(details, Mapping):
        cached = details.get("cached_tokens")
        if isinstance(cached, int) and not isinstance(cached, bool):
            return cached
    cached = usage.get("cached_tokens")
    if isinstance(cached, int) and not isinstance(cached, bool):
        return cached
    return None


def b_hit_total(analysis: Mapping[str, Any]) -> int:
    total = 0
    for rank in analysis.get("ranks", []):
        hits = rank.get("hit_counts", {})
        for unit_hash in rank.get("b_hashes", []):
            value = hits.get(unit_hash)
            if isinstance(value, int):
                total += value
    return total


def b_hashes_preserved(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    if not same_rank_count(before, after):
        return False
    for before_rank, after_rank in zip(before["ranks"], after["ranks"]):
        if not set(before_rank["b_hashes"]).issubset(set(after_rank["b_hashes"])):
            return False
    return True


def program_hashes_physically_absent(
    before: Mapping[str, Any], after: Mapping[str, Any], field: str
) -> bool:
    if not same_rank_count(before, after):
        return False
    for before_rank, after_rank in zip(before["ranks"], after["ranks"]):
        if set(before_rank[field]) & set(after_rank["all_hashes"]):
            return False
    return True


def prompt_metadata(prompt: str) -> dict[str, Any]:
    encoded = prompt.encode("utf-8")
    return {
        "characters": len(prompt),
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--worker-url", default="http://127.0.0.1:8081")
    parser.add_argument(
        "--model",
        default=os.environ.get("DYNAMO_MODEL"),
        help="served model id; defaults to the first item from /v1/models",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DYNAMO_API_KEY", "dummy"),
        help="frontend bearer token (default: DYNAMO_API_KEY or dummy)",
    )
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--poll-timeout", type=float, default=40.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--shared-chunks", type=int, default=96)
    parser.add_argument("--tail-chunks", type=int, default=48)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument(
        "--artifact-dir",
        type=pathlib.Path,
        default=pathlib.Path("/tmp/deadkv-dynamo-e2e/artifacts"),
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    if args.request_timeout <= 0 or args.poll_timeout <= 0 or args.poll_interval <= 0:
        parser.error("timeouts and poll interval must be positive")
    if args.shared_chunks < 8 or args.tail_chunks < 8:
        parser.error("shared-chunks and tail-chunks must each be at least 8")
    if args.max_tokens < 1:
        parser.error("max-tokens must be positive")
    return args


def run(args: argparse.Namespace, artifacts: Artifacts) -> None:
    base_headers = {"Authorization": f"Bearer {args.api_key}"}
    model = args.model or discover_model(
        args.frontend_url, base_headers, args.request_timeout, artifacts
    )
    program_a = f"deadkv-{artifacts.summary['run_id']}-A"
    program_b = f"deadkv-{artifacts.summary['run_id']}-B"
    prompt_a, prompt_b = make_prompts(args.shared_chunks, args.tail_chunks)
    payload_a = chat_payload(model, prompt_a, args.max_tokens)
    payload_b = chat_payload(model, prompt_b, args.max_tokens)
    final_payload = chat_payload(
        model, "Finalize this session without retaining any KV cache.", 1
    )
    artifacts.summary["configuration"] = {
        "frontend_url": args.frontend_url,
        "worker_url": args.worker_url,
        "model": model,
        "program_a": program_a,
        "program_b": program_b,
        "shared_chunks": args.shared_chunks,
        "tail_chunks": args.tail_chunks,
        "max_tokens": args.max_tokens,
        "prompt_a": prompt_metadata(prompt_a),
        "prompt_b": prompt_metadata(prompt_b),
    }
    artifacts.save_summary()

    flush = call_tokenizer_manager(
        args.worker_url, "flush_cache", timeout=args.request_timeout
    )
    artifacts.record_http("flush_cache", flush)
    require_control_ok(flush, "flush_cache")

    initial_http, initial_states, initial = fetch_states(
        args.worker_url,
        program_a=program_a,
        program_b=program_b,
        timeout=args.request_timeout,
    )
    artifacts.record_state("after_flush", initial_http, initial_states, initial)
    artifacts.check(
        "fresh_program_ids_absent_after_flush",
        all_ranks(initial, lambda rank: not rank["a_hashes"] and not rank["b_hashes"]),
    )

    response_a = send_chat(
        args.frontend_url,
        payload_a,
        program_id=program_a,
        base_headers=base_headers,
        timeout=args.request_timeout,
    )
    artifacts.record_http("ordinary_response_a", response_a)
    require_http_ok(response_a, "ordinary A inference")

    _, state_after_a = poll_states(
        "after_ordinary_a",
        args.worker_url,
        program_a=program_a,
        program_b=program_b,
        timeout=args.poll_timeout,
        request_timeout=args.request_timeout,
        interval=args.poll_interval,
        predicate=lambda analysis: all_ranks(
            analysis, lambda rank: bool(rank["a_hashes"]) and not rank["b_hashes"]
        ),
        artifacts=artifacts,
    )
    artifacts.check(
        "ordinary_response_does_not_reclaim_a",
        all_ranks(state_after_a, lambda rank: bool(rank["a_hashes"])),
        completed_http_status=response_a.status,
    )

    response_b = send_chat(
        args.frontend_url,
        payload_b,
        program_id=program_b,
        base_headers=base_headers,
        timeout=args.request_timeout,
    )
    artifacts.record_http("ordinary_response_b", response_b)
    require_http_ok(response_b, "ordinary B inference")

    _, before_a_end = poll_states(
        "before_a_end_shared_and_exclusive",
        args.worker_url,
        program_a=program_a,
        program_b=program_b,
        timeout=args.poll_timeout,
        request_timeout=args.request_timeout,
        interval=args.poll_interval,
        predicate=lambda analysis: all_ranks(
            analysis,
            lambda rank: bool(rank["shared_hashes"])
            and bool(rank["a_only_hashes"])
            and bool(rank["b_only_hashes"]),
        ),
        artifacts=artifacts,
    )
    artifacts.check(
        "shared_a_only_b_only_units_exist",
        all_ranks(
            before_a_end,
            lambda rank: bool(rank["shared_hashes"])
            and bool(rank["a_only_hashes"])
            and bool(rank["b_only_hashes"]),
        ),
    )

    final_a = send_chat(
        args.frontend_url,
        final_payload,
        program_id=program_a,
        base_headers=base_headers,
        timeout=args.request_timeout,
        final=True,
    )
    artifacts.record_http("final_a", final_a)
    artifacts.summary["final_a_response_shape"] = {
        "status": final_a.status,
        "content_type": final_a.headers.get("Content-Type"),
        "body_is_json": final_a.body_json is not None,
        "body_size_bytes": final_a.body_size_bytes,
        "transport_error": final_a.transport_error,
    }
    artifacts.save_summary()

    before_total = int(before_a_end["physical_bytes_total"])

    def a_reclaimed(analysis: Mapping[str, Any]) -> bool:
        return (
            all_ranks(
                analysis,
                lambda rank: not rank["a_hashes"] and bool(rank["b_hashes"]),
            )
            and b_hashes_preserved(before_a_end, analysis)
            and program_hashes_physically_absent(
                before_a_end, analysis, "a_only_hashes"
            )
            and int(analysis.get("physical_bytes_total", before_total)) < before_total
        )

    _, after_a_end = poll_states(
        "after_a_end",
        args.worker_url,
        program_a=program_a,
        program_b=program_b,
        timeout=args.poll_timeout,
        request_timeout=args.request_timeout,
        interval=args.poll_interval,
        predicate=a_reclaimed,
        artifacts=artifacts,
    )
    artifacts.check(
        "a_holders_removed_on_every_rank",
        all_ranks(after_a_end, lambda rank: not rank["a_hashes"]),
    )
    artifacts.check(
        "a_exclusive_units_physically_removed",
        program_hashes_physically_absent(before_a_end, after_a_end, "a_only_hashes"),
    )
    artifacts.check(
        "b_units_preserved_when_a_ends",
        b_hashes_preserved(before_a_end, after_a_end),
    )
    artifacts.check(
        "physical_bytes_drop_when_a_ends",
        int(after_a_end["physical_bytes_total"]) < before_total,
        before=before_total,
        after=after_a_end["physical_bytes_total"],
    )

    duplicate_a = send_chat(
        args.frontend_url,
        final_payload,
        program_id=program_a,
        base_headers=base_headers,
        timeout=args.request_timeout,
        final=True,
    )
    artifacts.record_http("duplicate_final_a", duplicate_a)
    duplicate_http, duplicate_states, after_duplicate_a = fetch_states(
        args.worker_url,
        program_a=program_a,
        program_b=program_b,
        timeout=args.request_timeout,
    )
    artifacts.record_state(
        "after_duplicate_final_a", duplicate_http, duplicate_states, after_duplicate_a
    )
    artifacts.check(
        "duplicate_final_a_is_idempotent",
        all_ranks(after_duplicate_a, lambda rank: not rank["a_hashes"])
        and b_hashes_preserved(after_a_end, after_duplicate_a),
        first_final_http_status=final_a.status,
        duplicate_final_http_status=duplicate_a.status,
        physical_bytes_before=after_a_end["physical_bytes_total"],
        physical_bytes_after=after_duplicate_a["physical_bytes_total"],
    )

    pre_repeat_hits = b_hit_total(after_duplicate_a)
    response_b_repeat = send_chat(
        args.frontend_url,
        payload_b,
        program_id=program_b,
        base_headers=base_headers,
        timeout=args.request_timeout,
    )
    artifacts.record_http("repeat_b_for_cache_hit", response_b_repeat)
    require_http_ok(response_b_repeat, "repeated B inference")
    repeat_http, repeat_states, after_b_repeat = fetch_states(
        args.worker_url,
        program_a=program_a,
        program_b=program_b,
        timeout=args.request_timeout,
    )
    artifacts.record_state(
        "after_b_cache_hit", repeat_http, repeat_states, after_b_repeat
    )
    post_repeat_hits = b_hit_total(after_b_repeat)
    cached_tokens = cached_tokens_from_response(response_b_repeat)
    latency_ratio = (
        response_b_repeat.elapsed_seconds / response_b.elapsed_seconds
        if response_b.elapsed_seconds > 0
        else None
    )
    if cached_tokens is not None:
        artifacts.check(
            "frontend_reports_b_cache_hit",
            cached_tokens > 0,
            cached_tokens=cached_tokens,
            repeated_latency_seconds=response_b_repeat.elapsed_seconds,
            first_b_latency_seconds=response_b.elapsed_seconds,
        )
        cache_evidence = "frontend_usage.cached_tokens"
    else:
        # State hit_count is authoritative engine-side evidence.  Latency is
        # retained as supporting evidence, but is intentionally not the sole
        # pass/fail signal because a local request can be scheduler-noisy.
        artifacts.check(
            "engine_state_records_b_cache_hit",
            post_repeat_hits > pre_repeat_hits,
            hit_count_before=pre_repeat_hits,
            hit_count_after=post_repeat_hits,
            first_b_latency_seconds=response_b.elapsed_seconds,
            repeated_latency_seconds=response_b_repeat.elapsed_seconds,
            latency_ratio=latency_ratio,
        )
        cache_evidence = "aginfer_state.hit_count"
    artifacts.summary["b_cache_hit_evidence"] = {
        "source": cache_evidence,
        "cached_tokens": cached_tokens,
        "hit_count_before": pre_repeat_hits,
        "hit_count_after": post_repeat_hits,
        "first_b_latency_seconds": response_b.elapsed_seconds,
        "repeated_latency_seconds": response_b_repeat.elapsed_seconds,
        "latency_ratio": latency_ratio,
    }
    artifacts.save_summary()

    final_b = send_chat(
        args.frontend_url,
        final_payload,
        program_id=program_b,
        base_headers=base_headers,
        timeout=args.request_timeout,
        final=True,
    )
    artifacts.record_http("final_b", final_b)

    def b_reclaimed(analysis: Mapping[str, Any]) -> bool:
        return all_ranks(
            analysis, lambda rank: not rank["a_hashes"] and not rank["b_hashes"]
        ) and program_hashes_physically_absent(after_b_repeat, analysis, "b_hashes")

    _, after_b_end = poll_states(
        "after_b_end",
        args.worker_url,
        program_a=program_a,
        program_b=program_b,
        timeout=args.poll_timeout,
        request_timeout=args.request_timeout,
        interval=args.poll_interval,
        predicate=b_reclaimed,
        artifacts=artifacts,
    )
    artifacts.check(
        "b_holders_removed_on_every_rank",
        b_reclaimed(after_b_end),
    )
    artifacts.check(
        "b_units_physically_removed",
        program_hashes_physically_absent(after_b_repeat, after_b_end, "b_hashes"),
    )

    health_results: dict[str, dict[str, Any]] = {}
    for label, base in (("frontend", args.frontend_url), ("worker", args.worker_url)):
        for endpoint in ("/live", "/health"):
            result = http_json(
                "GET",
                join_url(base, endpoint),
                headers=base_headers if label == "frontend" else None,
                timeout=min(args.request_timeout, 30.0),
                tolerate_transport_error=True,
            )
            artifacts.record_http(f"health_{label}_{endpoint.strip('/')}", result)
            health_results[f"{label}{endpoint}"] = {
                "status": result.status,
                "ok": result.ok,
                "transport_error": result.transport_error,
            }
    artifacts.check(
        "frontend_live_after_cleanup",
        health_results["frontend/live"]["ok"],
        observed=health_results["frontend/live"],
    )
    artifacts.check(
        "worker_live_after_cleanup",
        health_results["worker/live"]["ok"],
        observed=health_results["worker/live"],
    )
    artifacts.summary["health"] = health_results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_id = args.run_id or (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    artifacts = Artifacts(args.artifact_dir, run_id)
    try:
        run(args, artifacts)
    except Exception as exc:
        artifacts.finish(
            "failed",
            error={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        print(f"FAIL: {exc}", file=sys.stderr)
        print(f"Artifacts: {artifacts.run_dir}", file=sys.stderr)
        return 1
    artifacts.finish("passed")
    print("PASS: Dynamo -> SGLang Dead-KV full pipeline verified")
    print(f"Artifacts: {artifacts.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
