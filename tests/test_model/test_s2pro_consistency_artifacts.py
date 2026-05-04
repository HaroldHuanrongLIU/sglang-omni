# SPDX-License-Identifier: Apache-2.0
"""JSON-only S2-Pro stage-3 checks for GitHub-hosted runners."""

from __future__ import annotations

import json
import os
from pathlib import Path

from tests.s2pro_consistency import assert_streaming_consistency

S2PRO_STAGE1_SPEED_RESULTS_DIR_ENV = "S2PRO_STAGE1_SPEED_RESULTS_DIR"
S2PRO_STAGE2_SPEED_RESULTS_DIR_ENV = "S2PRO_STAGE2_SPEED_RESULTS_DIR"
S2PRO_CONSISTENCY_CONCURRENCY_ENV = "S2PRO_CONSISTENCY_CONCURRENCY"
DEFAULT_CONSISTENCY_CONCURRENCY = 8
STREAMING_BENCHMARK_MAX_SAMPLES = 16


def _load_speed_results(results_root_env: str, output_dir_name: str) -> dict:
    results_root = os.environ.get(results_root_env)
    assert results_root, f"{results_root_env} must point to downloaded stage artifacts"

    matches = sorted(Path(results_root).rglob(f"{output_dir_name}/speed_results.json"))
    assert matches, f"Missing {output_dir_name}/speed_results.json under {results_root}"

    with open(matches[0]) as results_file:
        speed_results = json.load(results_file)
    assert "per_request" in speed_results, f"Missing 'per_request' key in {matches[0]}"
    return speed_results


def _selected_concurrency() -> int:
    option_value = os.environ.get(
        S2PRO_CONSISTENCY_CONCURRENCY_ENV,
        str(DEFAULT_CONSISTENCY_CONCURRENCY),
    )
    try:
        return int(option_value)
    except ValueError as exc:
        raise ValueError(
            f"{S2PRO_CONSISTENCY_CONCURRENCY_ENV} must be an integer"
        ) from exc


def test_s2pro_streaming_consistency_from_artifacts() -> None:
    concurrency = _selected_concurrency()
    non_stream_results = _load_speed_results(
        S2PRO_STAGE1_SPEED_RESULTS_DIR_ENV,
        f"vc_nonstream_c{concurrency}",
    )
    stream_results = _load_speed_results(
        S2PRO_STAGE2_SPEED_RESULTS_DIR_ENV,
        f"vc_stream_c{concurrency}",
    )

    assert_streaming_consistency(
        non_stream_results["per_request"],
        stream_results["per_request"],
        expected_stream_count=STREAMING_BENCHMARK_MAX_SAMPLES,
    )
