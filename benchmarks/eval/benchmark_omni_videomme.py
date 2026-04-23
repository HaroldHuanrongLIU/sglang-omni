# SPDX-License-Identifier: Apache-2.0
"""Video-MME benchmark for sglang-omni models.

Evaluates video understanding accuracy and performance on the Video-MME
test set via /v1/chat/completions with video input. Each sample is a
multiple-choice question (A-D) grounded in a YouTube video clip, covering
short, medium, and long durations across six domains.

Usage:
    # Smoke test — a handful of samples at c=1
    python benchmarks/eval/benchmark_omni_videomme.py \
        --model qwen3-omni --port 8000 --max-samples 10

    # Reference-sized run at c=4 (matches the numbers below)
    python benchmarks/eval/benchmark_omni_videomme.py \
        --model qwen3-omni --port 8000 \
        --max-samples 50 --max-concurrency 4 --max-tokens 256

    # Throughput probe at higher concurrency (raise --mem-fraction-static on
    # the server if you see CUDA OOM in the vision encoder)
    python benchmarks/eval/benchmark_omni_videomme.py \
        --model qwen3-omni --port 8000 \
        --max-samples 50 --max-concurrency 16


H200 Full-Set Reference Results

Reproducibility references for the FULL eval set — NOT CI thresholds.
CI runs on a 50-sample subset (videomme-ci-50) and has its own thresholds
elsewhere (see tasks/*.py).

Benchmark: Video-MME |  Dataset: lmms-lab/Video-MME test split
                        (2520 total questions; 50-sample prefix used here
                        as an indicative reference — the full 2520-sample
                        run is ~9 h at c=4 on one H200 because each video
                        prefill is ~50 s, which exceeds a single debugging
                        budget)
Hardware:  1 x H200 (thinker-only; speech disabled)
Last verified: 2026-04-23

Accuracy (summary)

| Model      | Config                           | accuracy | correct | failed | mc_fallback | Source                                                       |
| ---------- | -------------------------------- | -------- | ------- | ------ | ----------- | ------------------------------------------------------------ |
| Qwen3-Omni | thinker-only, mem_fraction=0.65  | 62.00%   | 31/50   | 0      | 2           | PR #327 [H200, 50-sample prefix, c=4, max_tokens=256]        |

Note: full 2520 not run — at c=4 on one H200, video prefill averages
~50 s/sample (throughput ~0.08 req/s), so the full test split is ~9 h
wall-clock. The 50-sample prefix documented here matches the subset used
by the videomme-ci-50 CI job and is sufficient to establish a smoke-test
baseline. The Qwen3-Omni thinker defaults from commit a40e591
(thinker_max_seq_len=32768, encoder reserve 0.20) are tuned for
single-request prompts; they OOM in the vision encoder at c>=4 for
long-video clips on H200, so this run was taken with
--mem-fraction-static 0.65 on the server to give the encoder ~48 GB of
activation headroom. A wider-coverage 300- or 900-sample reference at
c=4 should be added once a stable multi-hour test slot is available.

Speed (speed)

| Model      | Config                           | latency_mean_s | latency_p95_s | throughput_qps | tok_per_s_mean | tok_per_s_agg | Source                                                 |
| ---------- | -------------------------------- | -------------- | ------------- | -------------- | -------------- | ------------- | ------------------------------------------------------ |
| Qwen3-Omni | thinker-only, mem_fraction=0.65  | 48.24          | 78.87         | 0.082          | 2.6            | 2.5           | PR #327 [H200, 50-sample prefix, c=4, max_tokens=256]  |
"""


from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import save_json_results, wait_for_service
from benchmarks.dataset.videomme import load_videomme_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.tts import print_speed_summary
from benchmarks.tasks.video_understanding import (
    compute_videomme_metrics,
    make_videomme_send_fn,
    print_videomme_accuracy_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class VideoMMEEvalConfig:
    model: str
    split: str = "test"
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    max_samples: int | None = None
    max_tokens: int = 256
    temperature: float = 0.0
    output_dir: str | None = None
    max_concurrency: int = 1
    warmup: int = 0
    request_rate: float = float("inf")
    disable_tqdm: bool = False
    repo_id: str | None = None


def _build_base_url(config: VideoMMEEvalConfig) -> str:
    return config.base_url or f"http://{config.host}:{config.port}"


async def run_videomme_eval(config: VideoMMEEvalConfig) -> dict:
    base_url = _build_base_url(config)
    api_url = f"{base_url}/v1/chat/completions"

    samples = load_videomme_samples(
        repo_id=config.repo_id,
        split=config.split,
        max_samples=config.max_samples,
    )
    logger.info("Prepared %d Video-MME samples", len(samples))

    send_fn = make_videomme_send_fn(
        config.model,
        api_url,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.max_concurrency,
            request_rate=config.request_rate,
            warmup=config.warmup,
            disable_tqdm=config.disable_tqdm,
        )
    )
    request_results = await runner.run(samples, send_fn)

    summary, per_sample = compute_videomme_metrics(samples, request_results)
    speed = compute_speed_metrics(request_results, wall_clock_s=runner.wall_clock_s)
    results = {
        "summary": summary,
        "speed": speed,
        "config": {
            "model": config.model,
            "base_url": base_url,
            "repo_id": config.repo_id,
            "split": config.split,
            "max_samples": config.max_samples,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "max_concurrency": config.max_concurrency,
            "warmup": config.warmup,
        },
        "per_sample": per_sample,
    }

    if config.output_dir:
        save_json_results(results, config.output_dir, "videomme_results.json")

    return results


async def benchmark(args: argparse.Namespace) -> dict:
    config = VideoMMEEvalConfig(
        model=args.model,
        repo_id=args.repo_id,
        split=args.split,
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        output_dir=args.output_dir,
        max_concurrency=args.max_concurrency,
        warmup=args.warmup,
        request_rate=args.request_rate,
        disable_tqdm=args.disable_tqdm,
    )
    results = await run_videomme_eval(config)
    print_videomme_accuracy_summary(results["summary"], config.model)
    print_speed_summary(
        results["speed"],
        config.model,
        config.max_concurrency,
        title="Video-MME Speed",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video-MME benchmark for video understanding models."
    )
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="qwen3-omni")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=(
            "HuggingFace dataset repo for Video-MME. "
            "Defaults to zhaochenyang20/Video_MME."
        ),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, default="results/videomme")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    wait_for_service(args.base_url or f"http://{args.host}:{args.port}")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
