# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.common.agent_lifecycle import (
    canonical_agent_context,
    extract_program_id,
    is_program_final,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_canonical_session_fields_take_precedence_over_legacy_aliases():
    request = {
        "agent_context": {
            "session_id": "session-new",
            "trajectory_id": "trajectory-old",
            "session_final": False,
            "trajectory_final": True,
        }
    }

    assert extract_program_id(request) == "session-new"
    assert is_program_final(request)
    assert canonical_agent_context(request) == {
        "session_id": "session-new",
        "session_final": True,
    }


def test_kv_evict_hint_is_terminal():
    request = {
        "agent_context": {
            "session_id": "session-1",
            "kv_hints": {"evict_session": True},
        }
    }

    assert is_program_final(request)
    assert canonical_agent_context(request) == {
        "session_id": "session-1",
        "session_final": True,
        "kv_hints": {"evict_session": True},
    }


def test_legacy_trajectory_fields_are_normalized_for_typed_wire_contract():
    request = {
        "agent_context": {
            "trajectory_id": " trajectory-1 ",
            "trajectory_final": True,
        }
    }

    assert extract_program_id(request) == "trajectory-1"
    assert canonical_agent_context(request) == {
        "session_id": "trajectory-1",
        "session_final": True,
    }


@pytest.mark.parametrize(
    "payload",
    [{}, {"agent_context": None}, {"agent_context": {}}, {"agent_context": "bad"}],
)
def test_missing_or_malformed_context_has_no_lifecycle(payload):
    assert extract_program_id(payload) is None
    assert not is_program_final(payload)
