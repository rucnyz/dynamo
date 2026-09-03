# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for ``python -m dynamo.sglang_remote``."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse


@dataclass
class RemoteConfig:
    """Configuration for the remote-SGLang proxy worker."""

    sglang_url: str
    model_path: str
    namespace: str = "dynamo"
    component: str = "backend"
    model_name: Optional[str] = None
    kv_events_host: Optional[str] = None
    kv_events_port_base: Optional[int] = None
    disable_kv_events: bool = False
    generate_timeout_s: float = 3600.0
    control_timeout_s: float = 30.0
    discovery_backend: str = "etcd"
    request_plane: str = "tcp"

    def validate(self) -> None:
        parsed = urlparse(self.sglang_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(
                f"--sglang-url must be an absolute http(s) URL, got {self.sglang_url!r}"
            )
        if not self.model_path:
            raise ValueError("--model-path is required")
        if not self.namespace or not self.component:
            raise ValueError("--namespace and --component must be non-empty")
        if self.kv_events_port_base is not None and not (
            1 <= self.kv_events_port_base <= 65535
        ):
            raise ValueError("--kv-events-port-base must be in 1..65535")

    @property
    def generate_endpoint(self) -> str:
        return f"{self.namespace}.{self.component}.generate"

    @property
    def end_program_endpoint(self) -> str:
        return f"{self.namespace}.{self.component}.end_program"

    @property
    def clear_kv_blocks_endpoint(self) -> str:
        return f"{self.namespace}.{self.component}.clear_kv_blocks"

    @property
    def served_model_name(self) -> str:
        return self.model_name or self.model_path

    def resolved_kv_events_host(self) -> str:
        if self.kv_events_host:
            return self.kv_events_host
        return urlparse(self.sglang_url).hostname or "127.0.0.1"


def parse_args(argv: Optional[list[str]] = None) -> RemoteConfig:
    parser = argparse.ArgumentParser(
        prog="python -m dynamo.sglang_remote",
        description=(
            "Dynamo proxy worker for a remote standalone "
            "``sglang.launch_server`` instance."
        ),
    )
    parser.add_argument(
        "--sglang-url",
        required=True,
        help="Base URL of the remote SGLang HTTP server "
        "(e.g. http://10.0.0.5:30000).",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Model path / HF id used for MDC registration. Must match the "
        "remote server's --model-path so routers and agentreplay agree.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional served model name (defaults to --model-path).",
    )
    parser.add_argument(
        "--namespace",
        default="dynamo",
        help="Dynamo discovery namespace (default: dynamo).",
    )
    parser.add_argument(
        "--component",
        default="backend",
        help="Dynamo component name under the namespace "
        "(default: backend → {ns}.backend.generate).",
    )
    parser.add_argument(
        "--kv-events-host",
        default=None,
        help="Host used when dialing the remote ZMQ KV-event publisher. "
        "Defaults to the host of --sglang-url (required when the remote "
        "server binds tcp://*:port).",
    )
    parser.add_argument(
        "--kv-events-port-base",
        type=int,
        default=None,
        help="Override the ZMQ port base advertised by /server_info. "
        "Leave unset to trust the remote descriptor.",
    )
    parser.add_argument(
        "--disable-kv-events",
        action="store_true",
        help="Do not subscribe to remote ZMQ KV events "
        "(disables prefix-aware KvRouter routing for this worker).",
    )
    parser.add_argument(
        "--generate-timeout-s",
        type=float,
        default=3600.0,
        help="HTTP timeout for /generate streams (seconds).",
    )
    parser.add_argument(
        "--control-timeout-s",
        type=float,
        default=30.0,
        help="HTTP timeout for control-plane calls "
        "(session_end / events / abort / server_info).",
    )
    parser.add_argument(
        "--discovery-backend",
        default="etcd",
        help="Dynamo discovery backend (default: etcd).",
    )
    parser.add_argument(
        "--request-plane",
        default="tcp",
        help="Dynamo request plane (default: tcp).",
    )
    ns = parser.parse_args(argv)
    cfg = RemoteConfig(
        sglang_url=ns.sglang_url.rstrip("/"),
        model_path=ns.model_path,
        namespace=ns.namespace,
        component=ns.component,
        model_name=ns.model_name,
        kv_events_host=ns.kv_events_host,
        kv_events_port_base=ns.kv_events_port_base,
        disable_kv_events=bool(ns.disable_kv_events),
        generate_timeout_s=float(ns.generate_timeout_s),
        control_timeout_s=float(ns.control_timeout_s),
        discovery_backend=ns.discovery_backend,
        request_plane=ns.request_plane,
    )
    cfg.validate()
    return cfg
