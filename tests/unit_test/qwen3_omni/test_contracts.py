# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni_v1.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni_v1.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
)
from sglang_omni_v1.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni_v1.models.qwen3_omni.config import (
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)
from sglang_omni_v1.models.qwen3_omni.merge import decode_events, merge_for_thinker
from sglang_omni_v1.models.qwen3_omni.payload_types import PipelineState
from sglang_omni_v1.models.qwen3_omni.request_builders import (
    build_sglang_thinker_request,
    project_preprocessing_to_mm_aggregate,
)
from sglang_omni_v1.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni_v1.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler
from tests.unit_test.fixtures.qwen_fakes import (
    FakeCode2WavModel,
    FakeQwenTokenizer,
    make_qwen_payload,
    make_qwen_state,
)


def test_qwen_pipeline_config_and_state_contracts() -> None:
    text_config = Qwen3OmniPipelineConfig(model_path="model")
    speech_config = Qwen3OmniSpeechPipelineConfig(model_path="model")

    assert [stage.name for stage in text_config.stages] == [
        "preprocessing",
        "image_encoder",
        "audio_encoder",
        "mm_aggregate",
        "thinker",
        "decode",
    ]
    assert speech_config.terminal_stages == ["decode", "code2wav"]
    assert {stage.name: stage for stage in speech_config.stages}[
        "thinker"
    ].stream_to == ["talker_ar"]

    state = PipelineState.from_dict(
        {
            "prompt": {"input_ids": torch.tensor([1, 2]), "prompt_text": "hi"},
            "mm_inputs": "bad",
            "encoder_inputs": {"image_encoder": {"cache_key": "img"}},
            "thinker_out": {"output_ids": [3], "is_final": True},
        }
    )
    assert torch.equal(state.prompt["input_ids"], torch.tensor([1, 2]))
    assert state.mm_inputs == {}
    assert state.encoder_inputs["image_encoder"]["cache_key"] == "img"
    assert state.thinker_out["is_final"] is True


def test_qwen_mm_aggregate_keeps_lightweight_inputs_and_prunes_after_merge() -> None:
    state = make_qwen_state(
        mm_inputs={
            "image": {
                "pixel_values": torch.ones((2, 3)),
                "image_grid_thw": torch.tensor([[1, 1, 2]]),
            },
            "audio": {"audio_feature_lengths": torch.tensor([1])},
        },
        encoder_inputs={"image_encoder": {"cache_key": "image-cache"}},
    )

    projected = project_preprocessing_to_mm_aggregate(make_qwen_payload(state))
    projected_state = PipelineState.from_dict(projected.data)
    assert "pixel_values" not in projected_state.mm_inputs["image"]
    assert projected_state.encoder_inputs == {
        "image_encoder": {"cache_key": "image-cache"}
    }

    image_state = PipelineState(
        encoder_outs={"image_encoder": {"image_embeds": torch.ones((2, 2))}}
    )
    merged = merge_for_thinker(
        {
            "preprocessing": make_qwen_payload(state),
            "image_encoder": make_qwen_payload(image_state),
        }
    )
    merged_state = PipelineState.from_dict(merged.data)
    assert merged_state.encoder_inputs == {}
    assert merged_state.encoder_outs == {}
    assert "image_embeds" in merged_state.thinker_inputs["model_inputs"]
    assert merged_state.thinker_inputs["media_cache_keys"] == {
        "image": "image:image-cache",
        "video": "video:image-cache",
    }


def test_qwen_thinker_request_and_decode_contracts() -> None:
    stream_state = PipelineState()
    tokenizer = FakeQwenTokenizer(pieces={1: "A", 2: "\ufffd", 3: "B"})
    first = list(
        decode_events(
            thinker_out={"output_ids": [1]},
            state=stream_state,
            tokenizer=tokenizer,
            eos_token_id=99,
            step=1,
        )
    )
    dropped = list(
        decode_events(
            thinker_out={"output_ids": [2]},
            state=stream_state,
            tokenizer=tokenizer,
            eos_token_id=99,
            step=2,
        )
    )
    final = list(
        decode_events(
            thinker_out={"output_ids": [1, 3, 99], "is_final": True},
            state=stream_state,
            tokenizer=FakeQwenTokenizer(pieces={1: "A", 3: "B"}),
            eos_token_id=99,
            step=3,
        )
    )
    assert first[0].payload == {"text": "A"}
    assert dropped == []
    assert final[0].type == "text_final"
    assert final[0].payload == {"text": "AB"}


def test_qwen_sglang_request_hashes_media_tokens_without_changing_mrope_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, torch.Tensor] = {}

    def fake_mrope(input_ids, model_inputs, thinker_config):
        del model_inputs, thinker_config
        captured["input_ids"] = input_ids.clone()
        return torch.zeros((3, input_ids.numel()), dtype=torch.long), torch.tensor(0)

    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )
    monkeypatch.setattr(
        "sglang_omni_v1.models.qwen3_omni.request_builders._compute_mrope_positions",
        fake_mrope,
    )

    audio_token_id = 77
    input_ids = torch.tensor([10, audio_token_id, 11], dtype=torch.long)
    state = make_qwen_state(
        prompt={"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)},
        thinker_inputs={
            "model_inputs": {"audio_embeds": torch.ones((1, 4))},
            "media_cache_keys": {"audio": "audio:cache"},
        },
    )
    req_data = build_sglang_thinker_request(
        state,
        params={"max_new_tokens": 3, "seed": 123},
        tokenizer=FakeQwenTokenizer(),
        vocab_size=256,
        request_id="rid-1",
        thinker_config=SimpleNamespace(
            image_token_id=55,
            video_token_id=66,
            audio_token_id=audio_token_id,
        ),
    )

    pad_values = req_data.req.omni_model_inputs["pad_values"]
    assert pad_values["audio"] >= 256
    assert int(req_data.input_ids[1]) == pad_values["audio"]
    assert captured["input_ids"].tolist() == input_ids.tolist()


def test_qwen_talker_feedback_fifo_and_stream_done_contract() -> None:
    sched_req = SimpleNamespace(
        data=SimpleNamespace(
            pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
            pending_text_queue=deque(),
            tts_pad_embed=torch.tensor([7.0, 8.0]),
            thinker_chunks_done=False,
        )
    )

    assert (
        QwenTalkerModelRunner._take_next_decode_input_embed(
            sched_req=sched_req,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        is None
    )
    sched_req.data.pending_text_queue.append(torch.tensor([20.0, 20.0]))
    assert torch.equal(
        QwenTalkerModelRunner._take_next_decode_input_embed(
            sched_req=sched_req,
            device=torch.device("cpu"),
            dtype=torch.float32,
        ),
        torch.tensor([21.0, 22.0]),
    )

    scheduler = object.__new__(QwenTalkerScheduler)
    req_data = SimpleNamespace(
        pending_text_queue=deque([torch.tensor([11.0, 12.0])]),
        thinker_chunks_done=True,
    )
    payload = SimpleNamespace(
        prefetched_chunks=[SimpleNamespace(data=torch.tensor([20.0, 20.0]))],
        prefetched_stream_done=True,
    )
    assert scheduler._is_request_build_ready(payload, pending_stream_done=True)
    scheduler._initialize_request_stream_state(req_data, payload)
    assert len(req_data.pending_text_queue) == 1
    assert torch.equal(req_data.pending_text_queue[0], torch.tensor([11.0, 12.0]))


def test_qwen_code2wav_streams_incrementally_and_abort_clears_state() -> None:
    model = FakeCode2WavModel(total_upsample=2)
    scheduler = Code2WavScheduler(
        model,
        device="cpu",
        stream_chunk_size=2,
        left_context_size=1,
        sample_rate=24000,
    )
    scheduler._payloads["req-1"] = make_qwen_payload(request_id="req-1")
    scheduler._ensure_request_state("req-1")

    scheduler._on_chunk("req-1", SimpleNamespace(data=torch.tensor([1, 10])))
    scheduler._on_chunk("req-1", SimpleNamespace(data=torch.tensor([2, 20])))
    scheduler._on_chunk("req-1", SimpleNamespace(data=torch.tensor([3, 30])))
    scheduler._on_done("req-1")

    message = scheduler.outbox.get_nowait()
    audio = np.frombuffer(message.data.data["audio_waveform"], dtype=np.float32)
    assert model.calls == [(1, 2, 2), (1, 2, 2)]
    assert audio.shape == (6,)

    scheduler._payloads["req-2"] = make_qwen_payload(request_id="req-2")
    scheduler._ensure_request_state("req-2")
    scheduler._pending_done.add("req-2")
    scheduler.abort("req-2")
    assert "req-2" not in scheduler._code_chunks
    assert "req-2" not in scheduler._payloads
    assert "req-2" not in scheduler._pending_done


def test_qwen_model_runner_and_code_predictor_tensor_contracts() -> None:
    class RecordingEmbed:
        num_embeddings = 10

        def __init__(self) -> None:
            self.seen: torch.Tensor | None = None

        def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
            self.seen = input_ids.clone()
            return torch.zeros((input_ids.shape[0], 4), dtype=torch.float32)

    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    runner._embed_tokens = RecordingEmbed()
    runner._image_token_id = 5
    runner._video_token_id = 6
    runner._audio_token_id = 7
    req = SimpleNamespace(
        omni_model_inputs={
            "audio_embeds": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "pad_values": {"audio": 999},
        },
        _omni_consumed=None,
        is_chunked=0,
    )
    input_embeds, _, _ = runner._inject_multimodal_embeds(
        SimpleNamespace(input_ids=torch.tensor([1, 999, 2]), extend_seq_lens_cpu=[3]),
        SimpleNamespace(reqs=[req]),
    )

    assert (
        int(runner._embed_tokens.seen.max().item())
        < runner._embed_tokens.num_embeddings
    )
    assert torch.equal(input_embeds[1], torch.tensor([1.0, 2.0, 3.0, 4.0]))

    logits = torch.tensor([[[0.0, 1.0, 2.0]], [[2.0, 1.0, 0.0]]])
    sampled = Qwen3OmniTalker._sample_code_predictor_token(logits)
    assert sampled.shape == (2, 1)
    assert sampled[:, 0].tolist() == [2, 0]
