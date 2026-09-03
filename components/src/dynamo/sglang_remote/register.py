# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build ModelRuntimeConfig from remote ``GET /server_info`` and register."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple

from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    register_model,
)
from dynamo.runtime import Endpoint
from dynamo.sglang_remote.args import RemoteConfig

logger = logging.getLogger(__name__)


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return None


def build_runtime_config(
    server_info: Dict[str, Any],
    config: RemoteConfig,
    *,
    kv_event_publishing_enabled: bool,
) -> ModelRuntimeConfig:
    """Populate the MDC fields routers need from a remote /server_info dump."""
    runtime = ModelRuntimeConfig()
    runtime.context_length = _as_int(server_info.get("context_length"))
    page_size = _as_int(server_info.get("page_size")) or 1
    max_total_tokens = _as_int(server_info.get("max_total_num_tokens"))
    if max_total_tokens and page_size > 0:
        runtime.total_kv_blocks = max_total_tokens // page_size
        logger.info(
            "remote /server_info: total_kv_blocks=%s "
            "(max_total_num_tokens=%s page_size=%s)",
            runtime.total_kv_blocks,
            max_total_tokens,
            page_size,
        )
    max_running = _as_int(server_info.get("max_running_requests"))
    if max_running is not None:
        runtime.max_num_seqs = max_running
    max_prefill = _as_int(server_info.get("max_prefill_tokens"))
    if max_prefill is not None:
        runtime.max_num_batched_tokens = max_prefill
    elif max_total_tokens is not None:
        runtime.max_num_batched_tokens = max_total_tokens

    dp_size = _as_int(server_info.get("dp_size")) or 1
    runtime.data_parallel_size = max(1, dp_size)
    runtime.data_parallel_start_rank = 0
    runtime.kv_event_publishing_enabled = bool(kv_event_publishing_enabled)
    # Remote proxy is always aggregated; never advertise local indexer /
    # disagg bootstrap (those need in-process engine state).
    runtime.enable_local_indexer = False
    return runtime


def parse_kv_events_descriptor(
    server_info: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return the structured ``kv_events`` descriptor, or None if disabled."""
    kv = server_info.get("kv_events")
    if not isinstance(kv, dict):
        return None
    if kv.get("publisher") in (None, "null", "none", ""):
        return None
    return kv


def resolve_kv_zmq_plan(
    server_info: Dict[str, Any],
    config: RemoteConfig,
) -> Optional[Tuple[str, int, int, int]]:
    """Return ``(host, port_base, dp_size, page_size)`` for ZMQ subscribers.

    ``/server_info``'s ``kv_events.endpoint_host`` is typically ``*``; we
    substitute :meth:`RemoteConfig.resolved_kv_events_host`.
    """
    if config.disable_kv_events:
        return None
    desc = parse_kv_events_descriptor(server_info)
    if desc is None:
        # Fall back to raw CLI string if the structured descriptor is absent.
        raw = server_info.get("kv_events_config")
        if isinstance(raw, str) and raw.strip():
            import json

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict) and parsed.get("publisher") not in (
                None,
                "null",
                "none",
                "",
            ):
                host = config.resolved_kv_events_host()
                endpoint = parsed.get("endpoint") or "tcp://*:5557"
                # endpoint like tcp://*:5557 or tcp://0.0.0.0:5557
                try:
                    port = int(str(endpoint).rsplit(":", 1)[-1])
                except ValueError:
                    port = 5557
                if config.kv_events_port_base is not None:
                    port = config.kv_events_port_base
                page_size = _as_int(server_info.get("page_size")) or 1
                dp_size = _as_int(server_info.get("dp_size")) or 1
                return host, port, max(1, dp_size), page_size
        logger.warning(
            "remote SGLang has no kv_events publisher; "
            "prefix-aware KvRouter routing will be blind for this worker"
        )
        return None

    host = config.resolved_kv_events_host()
    port = config.kv_events_port_base
    if port is None:
        port = _as_int(desc.get("endpoint_port_base"))
    if port is None:
        logger.warning("kv_events descriptor missing endpoint_port_base")
        return None
    dp_size = _as_int(desc.get("dp_size")) or _as_int(server_info.get("dp_size")) or 1
    page_size = (
        _as_int(desc.get("block_size"))
        or _as_int(server_info.get("page_size"))
        or 1
    )
    return host, int(port), max(1, int(dp_size)), int(page_size)


async def register_remote_model(
    generate_endpoint: Endpoint,
    config: RemoteConfig,
    server_info: Dict[str, Any],
    *,
    kv_event_publishing_enabled: bool,
) -> ModelRuntimeConfig:
    runtime = build_runtime_config(
        server_info,
        config,
        kv_event_publishing_enabled=kv_event_publishing_enabled,
    )
    await register_model(
        ModelInput.Tokens,
        ModelType.Chat | ModelType.Completions,
        generate_endpoint,
        config.model_path,
        config.served_model_name,
        runtime_config=runtime,
        worker_type=WorkerType.Aggregated,
    )
    logger.info(
        "registered remote model %s as %s (kv_events=%s)",
        config.model_path,
        config.generate_endpoint,
        kv_event_publishing_enabled,
    )
    return runtime
