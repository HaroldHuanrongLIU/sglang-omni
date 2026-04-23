# SPDX-License-Identifier: Apache-2.0
"""Video-MME dataset loader for local benchmarks."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from datasets import Video, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)


@dataclass
class VideoMMESample:
    sample_id: str
    video_path: str
    question: str
    options: list[str]
    answer: str
    url: str = ""
    video_id: str = ""
    question_id: str = ""
    duration: str = "short"
    domain: str = "unknown"
    task_type: str = "understanding"
    sub_category: str = ""
    prompt: str = ""
    all_choices: list[str] = field(default_factory=list)
    index2ans: dict[str, str] = field(default_factory=dict)


def _strip_option_prefix(option: str) -> str:
    return re.sub(r"^[A-D]\.\s*", "", option.strip())


def format_videomme_prompt(question: str, options: list[str]) -> str:
    prompt = f"{question.strip()}\n"
    for index, option in enumerate(options):
        letter = chr(ord("A") + index)
        prompt += f"{letter}. {option}\n"
    prompt += (
        "\nAnswer the following multiple-choice question. "
        "The last line of your response should be of the "
        "following format: 'Answer: $LETTER' (without quotes) "
        "where LETTER is one of the options. "
        "Think step by step before answering."
    )
    return prompt


def _extract_video_path(row: dict) -> str | None:
    video = row.get("video")
    if isinstance(video, dict):
        path = video.get("path")
        if path:
            return str(path)
    if isinstance(video, str) and video:
        return video
    return None


def _dataset_to_samples(dataset, *, max_samples: int | None) -> list[VideoMMESample]:
    samples: list[VideoMMESample] = []
    for row_index, row in enumerate(dataset):
        duration = str(row.get("duration", "short")).strip()
        question_id = str(row.get("question_id", f"videomme:{row_index}")).strip()

        options = [_strip_option_prefix(str(option)) for option in row["options"]]
        all_choices = [chr(ord("A") + i) for i in range(len(options))]
        index2ans = {choice: option for choice, option in zip(all_choices, options)}
        video_id = str(row["video_id"]).strip()
        url = str(row["url"]).strip()
        video_path = _extract_video_path(row)
        if not video_path:
            logger.warning(
                "Skipping Video-MME sample %s because the dataset row has no usable video path",
                question_id,
            )
            continue

        samples.append(
            VideoMMESample(
                sample_id=question_id,
                video_path=video_path,
                question=str(row["question"]).strip(),
                options=options,
                answer=str(row["answer"]).strip(),
                url=url,
                video_id=video_id,
                question_id=question_id,
                duration=duration,
                domain=str(row.get("domain", "unknown")).strip(),
                task_type=str(row.get("task_type", "understanding")).strip(),
                sub_category=str(row.get("sub_category", "")).strip(),
                prompt=format_videomme_prompt(str(row["question"]).strip(), options),
                all_choices=all_choices,
                index2ans=index2ans,
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


def load_videomme_samples(
    max_samples: int | None = None,
    *,
    repo_id: str | None = None,
    split: str = "test",
) -> list[VideoMMESample]:
    resolved_repo_id = repo_id or "zhaochenyang20/Video_MME"
    all_splits = load_dataset(resolved_repo_id)
    split_parts = sorted(
        split_name
        for split_name in all_splits.keys()
        if split_name.startswith(f"{split}_part_")
    )
    if split_parts:
        ds = concatenate_datasets(
            [all_splits[split_name] for split_name in split_parts]
        )
    elif split in all_splits:
        ds = all_splits[split]
    else:
        raise ValueError(
            f"Split '{split}' not found in {resolved_repo_id}, and no chunked splits matching "
            f"'{split}_part_*' were found. Available splits: {list(all_splits.keys())}"
        )
    ds = ds.cast_column("video", Video(decode=False))
    samples = _dataset_to_samples(ds, max_samples=max_samples)
    logger.info("Loaded %d Video-MME samples", len(samples))
    return samples
