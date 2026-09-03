# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point: ``python -m dynamo.sglang_remote --sglang-url ... --model-path ...``."""

from __future__ import annotations

import asyncio
import logging

import uvloop

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sglang_remote.args import RemoteConfig, parse_args
from dynamo.sglang_remote.client import SglangHttpClient
from dynamo.sglang_remote.handler import RemoteWorkerHandler
from dynamo.sglang_remote.kv_events import start_kv_event_publishers
from dynamo.sglang_remote.register import (
    register_remote_model,
    resolve_kv_zmq_plan,
)

configure_dynamo_logging()
logger = logging.getLogger(__name__)


async def _serve(runtime: DistributedRuntime, config: RemoteConfig) -> None:
    client = SglangHttpClient(
        config.sglang_url,
        generate_timeout_s=config.generate_timeout_s,
        control_timeout_s=config.control_timeout_s,
    )
    try:
        logger.info("probing remote SGLang at %s", config.sglang_url)
        server_info = await client.get_server_info()
    except Exception:
        await client.aclose()
        raise

    incremental = bool(server_info.get("incremental_streaming_output"))
    logger.info(
        "remote server ready (incremental_streaming_output=%s, "
        "page_size=%s, max_total_num_tokens=%s)",
        incremental,
        server_info.get("page_size"),
        server_info.get("max_total_num_tokens"),
    )

    handler = RemoteWorkerHandler(
        client, incremental_streaming_output=incremental
    )

    generate_endpoint = runtime.endpoint(config.generate_endpoint)
    end_program_endpoint = runtime.endpoint(config.end_program_endpoint)
    clear_endpoint = runtime.endpoint(config.clear_kv_blocks_endpoint)

    plan = resolve_kv_zmq_plan(server_info, config)
    kv_publishers = start_kv_event_publishers(generate_endpoint, plan)
    kv_enabled = bool(kv_publishers)

    await register_remote_model(
        generate_endpoint,
        config,
        server_info,
        kv_event_publishing_enabled=kv_enabled,
    )

    logger.info(
        "serving %s / %s (proxy → %s)",
        config.generate_endpoint,
        config.end_program_endpoint,
        config.sglang_url,
    )
    try:
        await asyncio.gather(
            generate_endpoint.serve_endpoint(handler.generate),
            end_program_endpoint.serve_endpoint(handler.end_program),
            clear_endpoint.serve_endpoint(handler.clear_kv_blocks),
        )
    finally:
        for pub in kv_publishers:
            close = getattr(pub, "close", None) or getattr(pub, "shutdown", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    logger.debug("KvEventPublisher close failed", exc_info=True)
        await client.aclose()


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    config = parse_args()
    await _serve(runtime, config)


def main() -> None:
    # Fail-fast on bad CLI before DistributedRuntime spins up.
    parse_args()
    uvloop.run(worker())


if __name__ == "__main__":
    main()
