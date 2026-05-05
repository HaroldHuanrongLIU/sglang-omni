# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from sglang_omni_v1.models.fishaudio_s2_pro import stages
from sglang_omni_v1.models.fishaudio_s2_pro.config import S2ProPipelineConfig
from sglang_omni_v1.models.fishaudio_s2_pro.fish_scheduler import (
    FishIterationController,
    FishScheduler,
)
from sglang_omni_v1.models.fishaudio_s2_pro.model_runner import FishS2ProModelRunner
from sglang_omni_v1.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni_v1.models.fishaudio_s2_pro.request_builders import (
    apply_tts_result,
    build_sglang_tts_request,
    make_tts_scheduler_adapters,
)
from sglang_omni_v1.models.fishaudio_s2_pro.tokenizer import (
    Reference,
    S2ProTokenizerAdapter,
)
from sglang_omni_v1.scheduling.messages import IncomingMessage
from sglang_omni_v1.scheduling.types import (
    ModelRunnerOutput,
    RequestOutput,
    SchedulerRequest,
)
from tests.unit_test.fixtures.fish_fakes import (
    FakeFishCodec,
    FakeFishModel,
    FakeFishReq,
    FakeFishTokenizer,
    make_s2pro_payload,
    make_s2pro_state,
)


@pytest.fixture(autouse=True)
def fast_sampling_params(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )


def run_scheduler(
    scheduler, messages: list[IncomingMessage], output_count: int
) -> list:
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        for message in messages:
            scheduler.inbox.put(message)
        return [scheduler.outbox.get(timeout=2.0) for _ in range(output_count)]
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)


def test_fish_config_state_and_tokenizer_prompt_contracts() -> None:
    config = S2ProPipelineConfig(model_path="model")
    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "tts_engine",
        "vocoder",
    ]
    assert config.terminal_stages == ["vocoder"]
    assert config.gpu_placement == {"tts_engine": 0, "vocoder": 0}

    state = S2ProState(
        input_ids=torch.tensor([1, 2, 3]),
        vq_mask_tokens=torch.tensor([False, True, False]),
        vq_parts=[torch.tensor([[10, 11], [20, 21]])],
        output_codes=torch.tensor([[100, 101], [1, 2], [3, 4]]),
    )
    restored = S2ProState.from_dict(state.to_dict())
    assert restored.input_ids == [1, 2, 3]
    assert torch.equal(restored.vq_parts[0], torch.tensor([[10, 11], [20, 21]]))
    assert torch.equal(
        restored.output_codes, torch.tensor([[100, 101], [1, 2], [3, 4]])
    )

    tokenizer = FakeFishTokenizer()
    adapter = S2ProTokenizerAdapter(tokenizer)
    prompt = adapter.build_prompt(
        "target",
        references=[
            Reference(
                audio_bytes=b"",
                text="ref",
                vq_codes=torch.tensor([[0, 1], [10, 11]], dtype=torch.long),
            )
        ],
        num_codebooks=2,
        speaker="alice",
    )
    assert adapter.eos_token_ids == [99]
    assert prompt["vq_mask_tokens"].dtype == torch.bool
    assert prompt["vq_mask_tokens"].sum().item() == 2
    assert torch.equal(prompt["vq_parts"][0], torch.tensor([[0, 1], [10, 11]]))
    assert any("<|speaker:alice|>target" in text for text in tokenizer.encoded_texts)


def test_fish_tts_request_and_result_adapters_preserve_tensor_contracts() -> None:
    tokenizer = FakeFishTokenizer()
    state = make_s2pro_state(
        input_ids=[10, 11, 12],
        vq_mask_tokens=[False, True, True],
        vq_parts=[[[1, 2], [3, 4]]],
        max_new_tokens=6,
        temperature=0.6,
    )

    req_data = build_sglang_tts_request(state, tokenizer, request_id="req-1")
    assert torch.equal(req_data.input_ids, torch.tensor([10, 11, 12]))
    assert req_data.vq_mask_tokens.dtype == torch.bool
    assert torch.equal(req_data.vq_parts[0], torch.tensor([[1, 2], [3, 4]]))
    assert req_data.req.eos_token_ids == {99}

    req_data.output_codes = [
        torch.tensor([[100], [1], [2]], dtype=torch.long),
        torch.tensor([[101], [3], [4]], dtype=torch.long),
    ]
    apply_tts_result(state, req_data)
    assert torch.equal(
        state.output_codes,
        torch.tensor([[100, 101], [1, 3], [2, 4]], dtype=torch.long),
    )
    assert state.prompt_tokens == 3
    assert state.completion_tokens == 2

    payload = make_s2pro_payload(request_id="req-2")
    request_builder, result_adapter = make_tts_scheduler_adapters(tokenizer=tokenizer)
    adapted = request_builder(payload)
    adapted.output_codes = [torch.tensor([[100], [1], [2]], dtype=torch.long)]
    result_payload = result_adapter(adapted)
    assert adapted.stage_payload is payload
    assert result_payload.request is payload.request
    assert result_payload.data["output_codes"] == [[100], [1], [2]]


def test_fish_model_runner_vq_injection_and_code_collection_contracts() -> None:
    runner = object.__new__(FishS2ProModelRunner)
    runner.model = FakeFishModel()
    runner._semantic_begin_id = 200
    runner._semantic_end_id = 295
    prefill_request = SchedulerRequest(
        request_id="prefill",
        data=SimpleNamespace(
            req=FakeFishReq(extend_input_len=3),
            vq_mask_tokens=torch.tensor([True, False, True]),
            vq_parts=[torch.tensor([[7, 8], [9, 10]], dtype=torch.long)],
        ),
    )
    embeds = runner._build_prefill_input_embeds(
        SimpleNamespace(input_ids=torch.tensor([10, 11, 12])),
        [prefill_request],
    )
    assert torch.equal(embeds[0], torch.tensor([1007.0, 1009.0]))
    assert torch.equal(embeds[1], torch.tensor([11.0, 11.0]))

    active = SchedulerRequest(
        request_id="active",
        data=SimpleNamespace(
            req=FakeFishReq(is_chunked=0),
            output_codes=[],
            previous_semantic_tokens=[],
            last_codebook_values=None,
        ),
    )
    runner._collect_step_outputs(SimpleNamespace(next_token_ids=None), [active])
    assert len(active.data.output_codes) == 1
    assert torch.equal(active.data.last_codebook_values, torch.tensor([1, 2]))
    assert active.data.previous_semantic_tokens == [201]


class _FakePlanner:
    def __init__(self) -> None:
        self.recorded = None

    def select_requests(self, waiting, running):
        del running
        return list(waiting)

    def build_batch(self, requests):
        return SimpleNamespace(request_ids=[request.request_id for request in requests])

    def record_last_batch(self, batch_data) -> None:
        self.recorded = batch_data


class _FakeResourceManager:
    def __init__(self) -> None:
        self.freed: list[str] = []

    def free(self, request) -> None:
        self.freed.append(request.request_id)


def make_fish_scheduler() -> FishScheduler:
    def request_builder(payload):
        return SimpleNamespace(
            req=FakeFishReq(rid=payload.request_id),
            output_codes=[],
            previous_semantic_tokens=[],
            last_codebook_values=None,
            max_new_tokens=4,
            input_ids=[1, 2, 3],
        )

    def result_adapter(data):
        payload = make_s2pro_payload(request_id=data.req.rid)
        payload.data = {"output_ids": list(data.req.output_ids)}
        return payload

    scheduler = FishScheduler(
        tree_cache=SimpleNamespace(cache_unfinished_req=lambda req: None),
        req_to_token_pool=SimpleNamespace(),
        token_to_kv_pool_allocator=SimpleNamespace(),
        prefill_manager=SimpleNamespace(),
        decode_manager=SimpleNamespace(),
        server_args=SimpleNamespace(),
        model_runner=SimpleNamespace(),
        request_builder=request_builder,
        result_adapter=result_adapter,
        im_end_token_id=99,
        max_new_tokens=4,
    )
    scheduler.batch_planner = _FakePlanner()
    scheduler.resource_manager = _FakeResourceManager()
    return scheduler


def test_fish_scheduler_lifecycle_abort_and_iteration_contracts() -> None:
    request = SchedulerRequest(
        request_id="chunked",
        data=SimpleNamespace(
            req=FakeFishReq(is_chunked=2),
            output_codes=[],
            previous_semantic_tokens=[],
        ),
    )
    controller = FishIterationController(
        tree_cache=SimpleNamespace(cache_unfinished_req=lambda req: None),
        im_end_token_id=99,
        max_new_tokens=4,
    )
    controller.update_request(request, 10)
    assert request.data.req.is_chunked == 1
    assert request.data.req.output_ids == []

    scheduler = make_fish_scheduler()
    scheduler.process_input_requests([make_s2pro_payload(request_id="req-1")])
    batch = scheduler.schedule()
    finished = scheduler.update(
        batch,
        ModelRunnerOutput(outputs={"req-1": RequestOutput("req-1", data=99)}),
    )
    scheduler.emit_finished(finished)
    message = scheduler.outbox.get_nowait()
    assert batch.request_ids == ["req-1"]
    assert scheduler.resource_manager.freed == ["req-1"]
    assert message.type == "result"
    assert message.data.data["output_ids"] == [99]

    scheduler.process_input_requests([make_s2pro_payload(request_id="req-2")])
    scheduler.abort("req-2")
    scheduler.inbox.put(
        IncomingMessage("req-2", "new_request", make_s2pro_payload(request_id="req-2"))
    )
    assert scheduler.recv_requests() == []
    assert "req-2" not in scheduler._requests


def test_fish_vocoder_batches_and_trims_audio_by_code_length(monkeypatch) -> None:
    codec = FakeFishCodec(frame_length=4)
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(stages, "_load_codec", lambda checkpoint, device: codec)
    scheduler = stages.create_vocoder_executor(
        "unused",
        device="cpu",
        max_batch_size=4,
        max_batch_wait_ms=50,
    )

    def payload(request_id: str, code_len: int) -> object:
        return make_s2pro_payload(
            S2ProState(
                output_codes=torch.arange(3 * code_len).reshape(3, code_len),
                prompt_tokens=4,
                completion_tokens=code_len,
                engine_time_s=0.5,
            ),
            request_id=request_id,
        )

    first, second = run_scheduler(
        scheduler,
        [
            IncomingMessage("req-short", "new_request", payload("req-short", 2)),
            IncomingMessage("req-long", "new_request", payload("req-long", 3)),
        ],
        output_count=2,
    )
    outputs = {first.request_id: first.data, second.request_id: second.data}

    assert codec.calls == [(2, 2, 3)]
    assert outputs["req-short"].data["audio_data"] == [1.0] * 8
    assert outputs["req-long"].data["audio_data"] == [2.0] * 12
    assert outputs["req-short"].data["usage"]["total_tokens"] == 6
