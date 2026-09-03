# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request/response mapping between Dynamo request plane and SGLang HTTP."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from dynamo.common.agent_lifecycle import extract_program_id
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.sglang_remote.client import SglangHttpClient

logger = logging.getLogger(__name__)

_SAMPLING_OPTION_FIELDS = (
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "min_tokens",
)


def build_sampling_params(request: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror ``DecodeWorkerHandler._build_sampling_params`` (token mode).

    ``forced_output_ids`` rides ``extra_args`` because Dynamo's request plane
    strips unknown nested keys from ``sampling_options``.
    """
    sampling_opts = request.get("sampling_options", {}) or {}
    stop_conditions = request.get("stop_conditions", {}) or {}

    hidden = stop_conditions.get("stop_token_ids_hidden") or []
    plain = stop_conditions.get("stop_token_ids") or []
    merged = list(set(hidden).union(plain))
    stop_token_ids = merged if merged else None

    param_mapping: Dict[str, Any] = {
        "n": sampling_opts.get("n"),
        "max_new_tokens": stop_conditions.get("max_tokens"),
        "ignore_eos": stop_conditions.get("ignore_eos"),
        "stop_token_ids": stop_token_ids,
    }
    for field in _SAMPLING_OPTION_FIELDS:
        if sampling_opts.get(field) is not None:
            param_mapping[field] = sampling_opts.get(field)
    if sampling_opts.get("seed") is not None:
        param_mapping["sampling_seed"] = sampling_opts.get("seed")

    extra_args = request.get("extra_args") or {}
    forced = extra_args.get("forced_output_ids") if isinstance(extra_args, dict) else None
    if forced and not param_mapping.get("custom_params"):
        param_mapping["custom_params"] = {"forced_output_ids": list(forced)}

    keep_if_none = {"max_new_tokens"}
    return {
        k: v for k, v in param_mapping.items() if v is not None or k in keep_if_none
    }


def build_generate_body(request: Dict[str, Any]) -> Dict[str, Any]:
    """Map a Dynamo preprocessed request to an SGLang ``/generate`` JSON body."""
    token_ids = request.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError("request.token_ids must be a non-empty list of ints")
    if request.get("bootstrap_info") is not None:
        raise RuntimeError(
            "sglang_remote does not support disaggregated decode "
            "(bootstrap_info present); use in-process dynamo.sglang"
        )
    if request.get("prefill_result") is not None:
        raise RuntimeError(
            "sglang_remote does not support disaggregated prefill_result; "
            "use in-process dynamo.sglang"
        )

    body: Dict[str, Any] = {
        "input_ids": list(token_ids),
        "sampling_params": build_sampling_params(request),
        "stream": True,
    }
    program_id = extract_program_id(request)
    if program_id is not None:
        body["program_id"] = program_id

    routing = request.get("routing") or {}
    dp_rank = routing.get("dp_rank")
    if dp_rank is not None:
        # SGLang HTTP accepts data_parallel_rank on GenerateReqInput in newer
        # builds; ignore silently if the remote rejects unknown fields by
        # stripping — we only set it when present.
        body["data_parallel_rank"] = dp_rank
    return body


def _delta_token_ids(
    output_ids: Sequence[int],
    *,
    incremental: bool,
    prev_len: int,
) -> tuple[List[int], int]:
    """Convert one SSE chunk's ``output_ids`` into a disjoint delta.

    Default SGLang streaming returns *cumulative* ``output_ids``; with
    ``--incremental-streaming-output`` each chunk is already a delta.
    """
    ids = [int(x) for x in output_ids]
    if incremental:
        return ids, prev_len + len(ids)
    if len(ids) < prev_len:
        # Server restarted the cumulative list (shouldn't happen mid-stream);
        # treat the whole list as new tokens.
        return ids, len(ids)
    delta = ids[prev_len:]
    return delta, len(ids)


def _finish_reason_type(meta_info: Dict[str, Any]) -> Optional[str]:
    fr = meta_info.get("finish_reason")
    if isinstance(fr, dict):
        return fr.get("type")
    if isinstance(fr, str):
        return fr
    return None


class RemoteWorkerHandler:
    """Dynamo endpoint handlers backed by :class:`SglangHttpClient`."""

    def __init__(
        self,
        client: SglangHttpClient,
        *,
        incremental_streaming_output: bool = False,
    ) -> None:
        self.client = client
        self.incremental_streaming_output = bool(incremental_streaming_output)

    def _forward_aginfer_events(self, request: Dict[str, Any]) -> None:
        extra_args = request.get("extra_args")
        events = (
            extra_args.get("aginfer_events") if isinstance(extra_args, dict) else None
        )
        if not events:
            return

        async def _push() -> None:
            try:
                await self.client.put_aginfer_events(events)
            except Exception:  # noqa: BLE001 — best-effort
                logger.debug(
                    "aginfer: PUT /aginfer/events failed (ignored)",
                    exc_info=True,
                )

        asyncio.create_task(_push())

    async def generate(
        self, request: Dict[str, Any], context: Any
    ) -> AsyncIterator[Dict[str, Any]]:
        self._forward_aginfer_events(request)
        body = build_generate_body(request)

        request_id_future: asyncio.Future[str] = asyncio.Future()
        abort_task = asyncio.create_task(
            self._abort_on_cancel(request_id_future, context)
        )
        prev_len = 0
        try:
            async for chunk in self.client.generate_stream(body):
                if getattr(context, "is_stopped", lambda: False)():
                    break
                meta_info = chunk.get("meta_info") or {}
                if not isinstance(meta_info, dict):
                    meta_info = {}
                rid = meta_info.get("id")
                if (
                    isinstance(rid, str)
                    and rid
                    and not request_id_future.done()
                ):
                    request_id_future.set_result(rid)

                output_ids = chunk.get("output_ids") or []
                if not isinstance(output_ids, list):
                    output_ids = []
                finish_type = _finish_reason_type(meta_info)
                delta, prev_len = _delta_token_ids(
                    output_ids,
                    incremental=self.incremental_streaming_output,
                    prev_len=prev_len,
                )
                if not delta and not finish_type:
                    continue

                out: Dict[str, Any] = {
                    "index": chunk.get("index") or 0,
                    "token_ids": delta,
                }
                if finish_type:
                    out["finish_reason"] = normalize_finish_reason(finish_type)
                    prompt_tokens = meta_info.get("prompt_tokens")
                    completion_tokens = meta_info.get("completion_tokens")
                    cached_tokens = meta_info.get("cached_tokens")
                    if prompt_tokens is not None and completion_tokens is not None:
                        usage: Dict[str, Any] = {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        }
                        if isinstance(cached_tokens, int) and cached_tokens > 0:
                            usage["prompt_tokens_details"] = {
                                "cached_tokens": cached_tokens
                            }
                        out["completion_usage"] = usage
                yield out
        finally:
            abort_task.cancel()
            try:
                await abort_task
            except asyncio.CancelledError:
                pass
            if not request_id_future.done():
                request_id_future.cancel()

    async def _abort_on_cancel(
        self, request_id_future: asyncio.Future[str], context: Any
    ) -> None:
        """Watch ``context`` for cancellation and POST /abort_request."""
        try:
            while True:
                if getattr(context, "is_stopped", lambda: False)():
                    rid: Optional[str] = None
                    if request_id_future.done() and not request_id_future.cancelled():
                        try:
                            rid = request_id_future.result()
                        except Exception:  # noqa: BLE001
                            rid = None
                    await self.client.abort_request(rid=rid)
                    return
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            return

    async def end_program(
        self, body: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        body = body or {}
        program_id = body.get("program_id")
        if not isinstance(program_id, str) or not program_id.strip():
            yield {"status": "error", "message": "program_id is required"}
            return
        program_id = program_id.strip()
        session_id = body.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            session_id = program_id
        try:
            result = await self.client.session_end(program_id, session_id=session_id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to end aginfer program %s", program_id)
            yield {
                "status": "error",
                "program_id": program_id,
                "message": str(exc),
            }
            return

        per_rank = result.get("per_rank") or []
        complete = bool(result.get("ok")) and bool(per_rank) and all(
            isinstance(r, dict) and r.get("ok") and not r.get("deferred")
            for r in per_rank
        )
        if not complete:
            yield {
                "status": "error",
                "program_id": program_id,
                "message": "engine did not complete Dead-KV reclamation on every rank",
                "engine_result": {"result": per_rank, "raw": result},
            }
            return
        yield {
            "status": "success",
            "program_id": program_id,
            "engine_result": {"result": per_rank},
        }

    async def clear_kv_blocks(
        self, body: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[Dict[str, Any]]:
        # Aggregated remote proxy does not expose flush yet; refuse loudly so
        # callers do not think the remote tree was cleared.
        yield {
            "status": "error",
            "message": "sglang_remote.clear_kv_blocks is not implemented",
        }
