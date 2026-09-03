# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin Dynamo worker that proxies to a remote standalone SGLang HTTP server.

Registers the same ``{namespace}.backend.generate`` / ``.end_program``
endpoints as ``dynamo.sglang``, so existing routers
(``aginfer_router`` / ``thunderagent_router``) and ``agentreplay
replay-dynamo`` keep working without code changes. The actual engine
lives on another host/container and is reached over HTTP + ZMQ KV events.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
