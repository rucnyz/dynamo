# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subscribe to a remote SGLang ZMQ KV-event publisher and re-publish into Dynamo."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from dynamo.runtime import Endpoint

logger = logging.getLogger(__name__)


def start_kv_event_publishers(
    generate_endpoint: Endpoint,
    plan: Optional[Tuple[str, int, int, int]],
    *,
    worker_id: Optional[int] = None,
) -> List[Any]:
    """Create one ``KvEventPublisher`` per DP rank connecting to ``host:port+rank``.

    ``plan`` is ``(host, port_base, dp_size, page_size)`` from
    :func:`dynamo.sglang_remote.register.resolve_kv_zmq_plan`. Returns an empty
    list when KV events are disabled / unavailable.
    """
    if plan is None:
        return []
    host, port_base, dp_size, page_size = plan
    try:
        from dynamo.llm import KvEventPublisher
    except ImportError:
        logger.warning(
            "KvEventPublisher unavailable; skipping remote KV event subscription"
        )
        return []

    publishers: List[Any] = []
    for dp_rank in range(dp_size):
        port = port_base + dp_rank
        # Bracket IPv6 hosts the same way SGLang's NetworkAddress does.
        if ":" in host and not host.startswith("["):
            zmq_ep = f"tcp://[{host}]:{port}"
        else:
            zmq_ep = f"tcp://{host}:{port}"
        logger.info(
            "remote KV events: subscribing dp_rank=%s at %s (block_size=%s)",
            dp_rank,
            zmq_ep,
            page_size,
        )
        kwargs = {
            "endpoint": generate_endpoint,
            "kv_block_size": page_size,
            "zmq_endpoint": zmq_ep,
            "zmq_topic": "",
            "dp_rank": dp_rank,
        }
        if worker_id is not None:
            kwargs["worker_id"] = worker_id
        try:
            publishers.append(KvEventPublisher(**kwargs))
        except TypeError:
            # Older bindings may not accept worker_id / dp_rank; retry minimal.
            kwargs.pop("worker_id", None)
            try:
                publishers.append(KvEventPublisher(**kwargs))
            except Exception:  # noqa: BLE001
                logger.exception(
                    "failed to create KvEventPublisher for %s", zmq_ep
                )
        except Exception:  # noqa: BLE001
            logger.exception("failed to create KvEventPublisher for %s", zmq_ep)
    return publishers
