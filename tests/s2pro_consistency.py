# SPDX-License-Identifier: Apache-2.0
"""Lightweight S2-Pro consistency assertions shared by GPU and artifact tests."""

from __future__ import annotations

import statistics

DEFAULT_TOTAL_COMPLETION_TOKEN_RTOL = 0.12
DEFAULT_MEDIAN_COMPLETION_TOKEN_RTOL = 0.20
DEFAULT_TOTAL_AUDIO_DURATION_RTOL = 0.12


def _request_by_id(requests: list[dict]) -> dict:
    return {request["id"]: request for request in requests}


def _assert_request_sets(
    non_stream_by_id: dict,
    stream_by_id: dict,
    expected_stream_count: int | None,
) -> list:
    common_ids = sorted(set(non_stream_by_id) & set(stream_by_id))
    assert common_ids, "No overlapping request IDs between non-stream and stream runs"
    assert set(stream_by_id).issubset(set(non_stream_by_id)), (
        "Streaming requests must be a subset of non-streaming requests: "
        f"non_stream={sorted(non_stream_by_id)}, stream={sorted(stream_by_id)}"
    )
    if expected_stream_count is not None:
        assert len(stream_by_id) == expected_stream_count, (
            f"Expected {expected_stream_count} streaming requests, "
            f"got {len(stream_by_id)}"
        )
    return common_ids


def _assert_relative_difference(
    metric_name: str,
    non_stream_value: float,
    stream_value: float,
    relative_tolerance: float,
) -> None:
    max_value = max(non_stream_value, stream_value)
    assert abs(non_stream_value - stream_value) <= (relative_tolerance * max_value), (
        f"{metric_name} differ too much - "
        f"non_stream={non_stream_value}, stream={stream_value} "
        f"(rtol={relative_tolerance})"
    )


def assert_streaming_consistency(
    non_stream_requests: list[dict],
    stream_requests: list[dict],
    *,
    expected_stream_count: int | None = None,
    total_completion_token_rtol: float = DEFAULT_TOTAL_COMPLETION_TOKEN_RTOL,
    median_completion_token_rtol: float = DEFAULT_MEDIAN_COMPLETION_TOKEN_RTOL,
    total_audio_duration_rtol: float = DEFAULT_TOTAL_AUDIO_DURATION_RTOL,
) -> None:
    """Assert stable invariants on the shared request subset."""
    non_stream_by_id = _request_by_id(non_stream_requests)
    stream_by_id = _request_by_id(stream_requests)
    common_ids = _assert_request_sets(
        non_stream_by_id, stream_by_id, expected_stream_count
    )

    non_stream_completion_tokens: list[int] = []
    stream_completion_tokens: list[int] = []
    non_stream_audio_duration_total = 0.0
    stream_audio_duration_total = 0.0

    for request_id in common_ids:
        non_stream_request = non_stream_by_id[request_id]
        stream_request = stream_by_id[request_id]
        assert non_stream_request["prompt_tokens"] == stream_request["prompt_tokens"], (
            f"Request {request_id}: prompt_tokens mismatch - "
            f"non_stream={non_stream_request['prompt_tokens']}, "
            f"stream={stream_request['prompt_tokens']}"
        )
        non_stream_completion_tokens.append(non_stream_request["completion_tokens"])
        stream_completion_tokens.append(stream_request["completion_tokens"])
        non_stream_audio_duration_total += non_stream_request["audio_duration_s"]
        stream_audio_duration_total += stream_request["audio_duration_s"]

    _assert_relative_difference(
        "Total completion_tokens",
        sum(non_stream_completion_tokens),
        sum(stream_completion_tokens),
        total_completion_token_rtol,
    )
    _assert_relative_difference(
        "Median completion_tokens",
        statistics.median(non_stream_completion_tokens),
        statistics.median(stream_completion_tokens),
        median_completion_token_rtol,
    )
    _assert_relative_difference(
        "Total audio_duration_s",
        non_stream_audio_duration_total,
        stream_audio_duration_total,
        total_audio_duration_rtol,
    )
