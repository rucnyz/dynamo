# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone aginfer program-aware router service.

Usage:
    python -m dynamo.aginfer_router \\
        --endpoint dynamo.vllm.generate \\
        --router-block-size 64

Serves ``{namespace}.aginfer_router.generate``. Pause/resume is
opt-in per-request via ``agent_context.session_id``. The legacy
``trajectory_id`` alias is accepted and normalized at this boundary.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

import uvloop

from dynamo.aginfer_router.args import (
    ThunderAgentRouterConfig,
    build_aic_perf_config,
    build_kv_router_config,
    parse_args,
)
from dynamo.aginfer_router.capacity import WorkerCapacityProvider
from dynamo.aginfer_router.router import ThunderAgentScheduler
from dynamo.common.agent_lifecycle import (
    canonical_agent_context,
    extract_program_id,
    is_program_final,
)
from dynamo.llm import (
    KvRouter,
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    register_model,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)
_END_PROGRAM_TIMEOUT_SECONDS = 30.0


def _extract_program_id(request: dict[str, Any]) -> Optional[str]:
    return extract_program_id(request)


def _extract_parent_program_id(request: dict[str, Any]) -> Optional[str]:
    """``agent_context.parent_session_id`` -- the session that spawned this one
    as a sub-agent. Fed to the scheduler's value gate, which scores a program
    with live sub-agents higher (it will be re-entered when they return).
    Unused by the request plane otherwise.
    """
    ctx = request.get("agent_context")
    if not isinstance(ctx, dict):
        return None
    pid = ctx.get("parent_session_id")
    return pid if isinstance(pid, str) and pid else None


def _is_trajectory_final(request: dict[str, Any]) -> bool:
    """Return true for canonical or legacy terminal lifecycle metadata."""
    return is_program_final(request)


def _wrap_preprocessed_request(request: dict[str, Any]) -> dict[str, Any]:
    # Duplicated from dynamo.router/__main__.py since neither package exports
    # it. TODO(idhanani): file follow-up to lift this into dynamo.router as a
    # shared helper before the field list drifts.
    routing = request.get("routing")
    dp_rank = request.get("dp_rank")
    if routing is None and dp_rank is not None:
        routing = {"dp_rank": dp_rank}

    return {
        "model": request.get("model", "unknown"),
        "token_ids": request["token_ids"],
        "stop_conditions": request.get("stop_conditions", {}),
        "sampling_options": request.get("sampling_options", {}),
        "output_options": request.get("output_options", {}),
        "eos_token_ids": request.get("eos_token_ids", []),
        "annotations": request.get("annotations", []),
        "routing": routing,
        "router_config_override": request.get("router_config_override"),
        "prefill_result": request.get("prefill_result"),
        "bootstrap_info": request.get("bootstrap_info"),
        "extra_args": request.get("extra_args"),
        "mm_processor_kwargs": request.get("mm_processor_kwargs"),
        "agent_context": canonical_agent_context(request),
        "request_timestamp_ms": request.get("request_timestamp_ms"),
    }


class ThunderAgentRouterHandler:
    def __init__(
        self,
        runtime: DistributedRuntime,
        config: ThunderAgentRouterConfig,
    ) -> None:
        self._runtime = runtime
        self._config = config
        self._kv_router: Optional[KvRouter] = None
        self._capacity: Optional[WorkerCapacityProvider] = None
        self._scheduler: Optional[ThunderAgentScheduler] = None
        self._end_program_client: Any = None
        self._worker_id_extract_warned = False

    async def initialize(self) -> None:
        # Endpoint shape was validated by ThunderAgentRouterConfig.validate()
        # in parse_args; it also populates ``config.namespace``.
        worker_endpoint = self._runtime.endpoint(self._config.endpoint)

        self._kv_router = KvRouter(
            endpoint=worker_endpoint,
            block_size=self._config.router_block_size,
            kv_router_config=build_kv_router_config(self._config),
            aic_perf_config=build_aic_perf_config(self._config),
        )

        # Lifecycle messages are control-plane requests, not fake inference
        # turns. Every SGLang worker exposes this sibling endpoint under the
        # same component instance id, allowing an exact direct dispatch.
        endpoint_prefix, separator, _ = self._config.endpoint.rpartition(".")
        if not separator:
            raise ValueError(
                f"worker endpoint must end in an endpoint name: {self._config.endpoint!r}"
            )
        end_program_endpoint = self._runtime.endpoint(f"{endpoint_prefix}.end_program")
        self._end_program_client = await end_program_endpoint.client()

        self._capacity = WorkerCapacityProvider(worker_endpoint)
        self._capacity.start()

        self._scheduler = ThunderAgentScheduler(
            capacity=self._capacity,
            config=self._config.to_thunderagent_config(),
        )
        self._scheduler.start()

    async def shutdown(self) -> None:
        if self._scheduler is not None:
            await self._scheduler.stop()
        if self._capacity is not None:
            self._capacity.stop()

    async def generate(self, request: dict[str, Any]):
        if (
            self._scheduler is None
            or self._kv_router is None
            or self._end_program_client is None
        ):
            raise RuntimeError(
                "ThunderAgentRouterHandler used before initialize() was called"
            )
        program_id = _extract_program_id(request)
        terminal = _is_trajectory_final(request)
        if terminal and program_id is None:
            raise ValueError("terminal agent_context requires a non-empty session_id")

        # Notify every worker that may retain this program's KV and wait for
        # all acknowledgements before releasing the sticky route. If a subset
        # fails, keep the mapping so the caller can retry; engine end_program
        # is required to be idempotent for exactly this case.
        if program_id is not None and terminal:
            # Atomically close admission, wait for all already-admitted work to
            # drain, and only then snapshot every worker that may hold KV.
            worker_ids = await self._scheduler.begin_end_program(program_id)
            if worker_ids is None:
                # A duplicate final after a successful release is a no-op.
                return
            if not worker_ids:
                raise RuntimeError(
                    f"cannot end program {program_id!r}: original worker is unknown"
                )
            results = await asyncio.gather(
                *(
                    asyncio.wait_for(
                        self._end_program_on_worker(program_id, worker_id),
                        timeout=_END_PROGRAM_TIMEOUT_SECONDS,
                    )
                    for worker_id in worker_ids
                ),
                return_exceptions=True,
            )
            failures = [result for result in results if isinstance(result, Exception)]
            if failures:
                raise RuntimeError(
                    f"end_program({program_id!r}) failed on "
                    f"{len(failures)}/{len(worker_ids)} workers"
                ) from failures[0]
            await self._scheduler.end_program(program_id)
            return

        # Path A: no program_id -> behave like the standalone router.
        # Backward compat for clients that don't send agent_context.
        if program_id is None:
            preprocessed = _wrap_preprocessed_request(request)
            async for chunk in await self._kv_router.generate_from_request(
                preprocessed  # type: ignore[arg-type]
            ):
                yield chunk
            return

        # Path B: program lifecycle.
        token_ids = request["token_ids"]
        estimated_prompt_tokens = len(token_ids) if isinstance(token_ids, list) else 0

        prompt_tokens_seen = 0
        completion_tokens_seen = 0
        usage_completion_seen = False
        worker_attribution_recorded = False
        decision = await self._scheduler.before_request(
            program_id,
            estimated_prompt_tokens=estimated_prompt_tokens,
            parent_program_id=_extract_parent_program_id(request),
        )
        try:
            worker_pin = decision.assigned_worker_hint

            preprocessed = _wrap_preprocessed_request(request)
            if decision.priority_jump != 0.0:
                routing = preprocessed.get("routing") or {}
                existing = routing.get("priority_jump") or 0.0
                routing["priority_jump"] = float(existing) + decision.priority_jump
                preprocessed["routing"] = routing

            if worker_pin is not None:
                routing = preprocessed.get("routing") or {}
                routing["backend_instance_id"] = worker_pin
                preprocessed["routing"] = routing

            async for chunk in await self._kv_router.generate_from_request(
                preprocessed  # type: ignore[arg-type]
            ):
                if not worker_attribution_recorded:
                    decode_worker, prefill_worker = self._extract_worker_ids(chunk)
                    if decode_worker is not None or prefill_worker is not None:
                        worker_attribution_recorded = True
                        active_worker = decode_worker
                        if active_worker is None:
                            active_worker = (
                                worker_pin if worker_pin is not None else prefill_worker
                            )
                        await self._scheduler.assign_workers(
                            program_id,
                            decode_worker_id=active_worker,
                            prefill_worker_id=prefill_worker,
                        )

                usage = (
                    chunk.get("completion_usage") if isinstance(chunk, dict) else None
                )
                if isinstance(usage, dict):
                    prompt_tokens_seen = int(
                        usage.get("prompt_tokens", prompt_tokens_seen)
                    )
                    if isinstance(usage.get("completion_tokens"), int):
                        completion_tokens_seen = int(usage["completion_tokens"])
                        usage_completion_seen = True
                token_ids_out = (
                    chunk.get("token_ids", []) if isinstance(chunk, dict) else []
                )
                if isinstance(token_ids_out, list) and token_ids_out:
                    # Engine usage is authoritative if present; only the
                    # token-id fallback path increments completion_tokens_seen.
                    if not usage_completion_seen:
                        completion_tokens_seen += len(token_ids_out)
                    self._scheduler.record_output_tokens(program_id, len(token_ids_out))

                yield chunk
        finally:
            # Fall back to len(token_ids) if the engine didn't report usage --
            # still better than upstream's chars/5 estimator.
            if prompt_tokens_seen == 0 and isinstance(token_ids, list):
                prompt_tokens_seen = len(token_ids)
            await self._scheduler.after_request(
                program_id,
                prompt_tokens_seen,
                completion_tokens_seen,
            )

    async def _end_program_on_worker(self, program_id: str, worker_id: int) -> None:
        response_stream = await self._end_program_client.direct(
            {"type": "end_program", "program_id": program_id},
            worker_id,
        )
        acknowledged = False
        async for response in response_stream:
            data = getattr(response, "data", None)
            payload = data() if callable(data) else response
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"worker {worker_id} returned malformed end_program response"
                )
            if payload.get("status") not in {"ok", "success"}:
                raise RuntimeError(
                    f"worker {worker_id} rejected end_program({program_id!r}): "
                    f"{payload.get('message', payload)!r}"
                )
            acknowledged = True
        if not acknowledged:
            raise RuntimeError(
                f"worker {worker_id} returned no end_program acknowledgement"
            )

    def _extract_worker_ids(self, chunk: Any) -> tuple[Optional[int], Optional[int]]:
        # Expects the shape set by ``inject_worker_id_from_tracker`` in the Python
        # bindings: worker attribution rides ``routing_data.worker_id``. Log once if the
        # shape no longer matches; silent extraction failure here means we lose
        # worker-affinity on pin.
        if not isinstance(chunk, dict):
            self._warn_unexpected_chunk_shape("not a dict")
            return None, None
        routing_data = chunk.get("routing_data")
        if not isinstance(routing_data, dict):
            self._warn_unexpected_chunk_shape("no routing_data dict")
            return None, None
        info = routing_data.get("worker_id")
        if isinstance(info, dict):
            decode_worker_id = info.get("decode_worker_id")
            prefill_worker_id = info.get("prefill_worker_id")
            decode_worker_id = (
                decode_worker_id if isinstance(decode_worker_id, int) else None
            )
            prefill_worker_id = (
                prefill_worker_id if isinstance(prefill_worker_id, int) else None
            )
            if decode_worker_id is not None or prefill_worker_id is not None:
                # Aggregated mode reports the same instance for both fields;
                # worker_ids_for_program de-duplicates it at finalization.
                return decode_worker_id, prefill_worker_id
        self._warn_unexpected_chunk_shape("worker_id payload shape changed")
        return None, None

    def _extract_worker_id(self, chunk: Any) -> Optional[int]:
        """Compatibility helper used by older tests/callers."""
        decode_worker, prefill_worker = self._extract_worker_ids(chunk)
        return decode_worker if decode_worker is not None else prefill_worker

    def _warn_unexpected_chunk_shape(self, reason: str) -> None:
        if self._worker_id_extract_warned:
            return
        self._worker_id_extract_warned = True
        logger.warning(
            "aginfer worker-id extraction failed (%s); subsequent "
            "requests will lose sticky pinning until the binding shape is "
            "fixed.",
            reason,
        )


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    config = parse_args()
    logger.info(
        "aginfer Router starting (endpoint=%s, namespace=%s)",
        config.endpoint,
        config.namespace,
    )

    handler = ThunderAgentRouterHandler(runtime, config)
    await handler.initialize()

    generate_endpoint = runtime.endpoint(f"{config.namespace}.aginfer_router.generate")

    if config.model_name:
        model_path = config.model_path or config.model_name
        # Thread the tool_call/reasoning parsers into register_model so the
        # frontend's response path can translate model-native tool calls (e.g.
        # MiniMax's <minimax:tool_call> XML, Qwen's hermes) into OpenAI
        # tool_calls before pi / openhands / other agents see them. These use
        # the same --dyn-tool-call-parser / --dyn-reasoning-parser flag names
        # (and DYN_TOOL_CALL_PARSER / DYN_REASONING_PARSER env vars) as the
        # standalone dynamo.vllm worker.
        runtime_cfg = ModelRuntimeConfig()
        if config.tool_call_parser:
            runtime_cfg.tool_call_parser = config.tool_call_parser
        if config.reasoning_parser:
            runtime_cfg.reasoning_parser = config.reasoning_parser
        await register_model(
            model_input=ModelInput.Tokens,
            model_type=ModelType.Chat | ModelType.Completions,
            endpoint=generate_endpoint,
            model_path=model_path,
            model_name=config.model_name,
            runtime_config=runtime_cfg,
            # The router is the serving entry point (front door) exposing the
            # OpenAI surface; it has no mandatory peer-role dependency.
            worker_type=WorkerType.Aggregated,
        )

    try:
        await generate_endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
            metrics_labels=[("service", "aginfer_router")],
        )
    finally:
        await handler.shutdown()


def main() -> None:
    uvloop.run(worker())


if __name__ == "__main__":
    main()
