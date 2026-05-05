# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
import torch

from sglang_omni_v1.config.compiler import compile_pipeline
from sglang_omni_v1.config.schema import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni_v1.pipeline import relay_io
from sglang_omni_v1.pipeline.coordinator import Coordinator
from sglang_omni_v1.pipeline.mp_runner import _build_stage_groups
from sglang_omni_v1.pipeline.stage.input import AggregatedInput
from sglang_omni_v1.pipeline.stage.runtime import Stage
from sglang_omni_v1.pipeline.stage.stream_queue import StreamItem, StreamQueue
from sglang_omni_v1.pipeline.stage_process import get_stage_process_env
from sglang_omni_v1.proto import CompleteMessage, DataReadyMessage
from sglang_omni_v1.scheduling.messages import IncomingMessage
from sglang_omni_v1.scheduling.simple_scheduler import SimpleScheduler
from tests.unit_test.fixtures.pipeline_fakes import (
    EventLog,
    FakeMpContext,
    FakeRelay,
    FakeScheduler,
    RecordingCoordinatorControlPlane,
    RecordingStageControlPlane,
    collect_event_names,
    fake_factory_path,
    make_noop_projector,
    make_result_message,
    make_stage_payload,
    make_stream_message,
    make_tensor_payload,
    tensor_equal,
)

FACTORY = fake_factory_path("make_scheduler")


def stage(name: str, **kwargs) -> StageConfig:
    kwargs.setdefault("factory", FACTORY)
    return StageConfig(name=name, **kwargs)


def make_stage(
    *,
    name: str = "stage",
    role: str = "single",
    get_next=None,
    endpoints: dict[str, str] | None = None,
    scheduler: FakeScheduler | None = None,
    relay: FakeRelay | None = None,
    control_plane: RecordingStageControlPlane | None = None,
    **kwargs,
) -> Stage:
    return Stage(
        name=name,
        role=role,
        get_next=get_next or (lambda request_id, output: None),
        gpu_id=None,
        endpoints=endpoints or {},
        control_plane=control_plane or RecordingStageControlPlane(),
        relay=relay or FakeRelay(),
        scheduler=scheduler or FakeScheduler(),
        **kwargs,
    )


def run_scheduler(
    scheduler: SimpleScheduler,
    messages: list[IncomingMessage],
    *,
    output_count: int,
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


def test_pipeline_schema_keeps_topology_and_validation_contracts() -> None:
    config = PipelineConfig(
        model_path="model",
        stages=[
            stage("preprocess", next="thinker"),
            stage("thinker", next="decode", gpu=[0, 1], tp_size=2),
            stage("decode", terminal=True),
        ],
    )

    assert config.resolved_entry_stage == "preprocess"
    assert config.terminal_stages == ["decode"]
    assert config.gpu_placement == {"thinker": [0, 1]}

    with pytest.raises(ValueError, match="unknown stages"):
        PipelineConfig(model_path="model", stages=[stage("a", next="missing")])
    with pytest.raises(ValueError, match="wait_for but no merge_fn"):
        PipelineConfig(
            model_path="model",
            stages=[
                stage("a", wait_for=["b"], terminal=True),
                stage("b", terminal=True),
            ],
        )
    with pytest.raises(ValueError, match="gpu has 1 entries"):
        PipelineConfig(
            model_path="model",
            stages=[stage("tp", gpu=[0], tp_size=2, terminal=True)],
        )


def test_compile_pipeline_wires_routes_overrides_aggregation_and_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sglang_omni_v1.pipeline.stage.runtime as runtime

    monkeypatch.setattr(
        runtime,
        "create_relay",
        lambda relay_type, **kwargs: FakeRelay(device=kwargs.get("device", "cpu")),
    )
    config = PipelineConfig(
        model_path="global-model",
        name="contract",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        runtime_overrides={"thinker": {"model_path": "runtime-model", "extra": "rt"}},
        stages=[
            stage("preprocess", next=["thinker", "aggregate"]),
            stage(
                "thinker",
                factory=fake_factory_path("make_scheduler_accepting_model_path"),
                factory_args={"extra": "factory"},
                gpu=0,
                next="aggregate",
                stream_to=["talker"],
            ),
            stage(
                "aggregate",
                wait_for=["preprocess", "thinker"],
                merge_fn=fake_factory_path("merge_payloads"),
                terminal=True,
            ),
            stage("talker", gpu=0, terminal=True),
        ],
    )

    coordinator, stages = compile_pipeline(config)
    stage_map = {compiled.name: compiled for compiled in stages}

    assert coordinator.entry_stage == "preprocess"
    assert stage_map["preprocess"].get_next("req", None) == ["thinker", "aggregate"]
    assert isinstance(stage_map["aggregate"].input_handler, AggregatedInput)
    assert isinstance(stage_map["talker"]._stream_queue, StreamQueue)
    assert stage_map["thinker"]._same_gpu_targets == {"talker"}
    assert stage_map["thinker"].scheduler.model_path == "runtime-model"
    assert stage_map["thinker"].scheduler.factory_kwargs["extra"] == "rt"


def test_coordinator_multi_terminal_failure_and_abort_contracts() -> None:
    async def _run() -> None:
        coordinator = Coordinator(
            "inproc://complete",
            "inproc://abort",
            entry_stage="preprocess",
            terminal_stages=["decode", "code2wav"],
        )
        control_plane = RecordingCoordinatorControlPlane()
        coordinator.control_plane = control_plane
        coordinator.register_stage("preprocess", "inproc://preprocess")

        await coordinator._submit_request("req-1", {"text": "hello"})
        await coordinator._handle_completion(
            CompleteMessage("req-1", "decode", True, result={"text": "hi"})
        )
        assert not coordinator._completion_futures["req-1"].done()
        await coordinator._handle_completion(
            CompleteMessage("req-1", "code2wav", True, result={"audio": "ok"})
        )
        assert coordinator._completion_futures["req-1"].result() == {
            "decode": {"text": "hi"},
            "code2wav": {"audio": "ok"},
        }

        await coordinator._submit_request("req-2", "hello")
        future = coordinator._completion_futures["req-2"]
        assert await coordinator.abort("req-2") is True
        assert control_plane.aborts[0].request_id == "req-2"
        with pytest.raises(asyncio.CancelledError):
            await future

    asyncio.run(_run())


def test_stage_routes_results_streams_and_clears_abort_state() -> None:
    async def _run() -> None:
        relay = FakeRelay()
        scheduler = FakeScheduler()
        control_plane = RecordingStageControlPlane()
        stage_obj = make_stage(
            name="thinker",
            get_next=lambda request_id, output: "decode",
            endpoints={"decode": "inproc://decode", "talker": "inproc://talker"},
            project_payload={"decode": make_noop_projector("decode-only")},
            stream_targets=["talker"],
            relay=relay,
            scheduler=scheduler,
            control_plane=control_plane,
        )
        stage_obj._active_requests.add("req-1")
        scheduler.outbox.put(make_stream_message("req-1", data=torch.tensor([7])))
        scheduler.outbox.put(make_result_message("req-1", data={"answer": 1}))

        await stage_obj._drain_outbox()

        decode_msg = next(
            msg for target, _, msg in control_plane.sent_to_stage if target == "decode"
        )
        restored = await relay_io.read_payload(relay, "req-1", decode_msg.shm_metadata)
        assert restored.data == {"marker": "decode-only", "data": {"answer": 1}}
        stream_msg = next(
            msg
            for target, _, msg in control_plane.sent_to_stage
            if target == "talker" and msg.chunk_id == 0
        )
        assert stream_msg.chunk_id == 0

        stage_obj._stream_queue = StreamQueue()
        stage_obj._stream_queue.open("req-1")
        stage_obj._pending_stream_data["req-1"] = [
            StreamItem(0, torch.tensor([1]), "t")
        ]
        stage_obj._on_abort("req-1")

        assert "req-1" in stage_obj._aborted
        assert relay.cleaned[-1] == "req-1"
        assert scheduler.aborted == ["req-1"]
        assert "req-1" not in stage_obj._pending_stream_data

    asyncio.run(_run())


def test_stage_relay_read_failure_completes_with_error() -> None:
    async def _run() -> None:
        relay = FakeRelay()
        control_plane = RecordingStageControlPlane()
        stage_obj = make_stage(relay=relay, control_plane=control_plane)
        payload = make_stage_payload(request_id="req-1")
        metadata, _ = await relay_io.write_payload(relay, "req-1", payload)
        relay.fail_get = RuntimeError("read failed")

        await stage_obj._on_data_ready(
            DataReadyMessage("req-1", "upstream", "stage", metadata)
        )

        assert control_plane.completions[0].success is False
        assert "relay read failed" in control_plane.completions[0].error
        assert relay.cleaned[-1] == "req-1"

    asyncio.run(_run())


def test_relay_payload_and_cross_gpu_stream_contracts() -> None:
    async def _run() -> None:
        relay = FakeRelay()
        payload = make_tensor_payload()
        metadata, op = await relay_io.write_payload(relay, payload.request_id, payload)
        await op.wait_for_completion()
        restored = await relay_io.read_payload(relay, payload.request_id, metadata)
        assert tensor_equal(restored.data, payload.data)

        log = EventLog()
        stream_relay = FakeRelay(log=log)
        control_plane = RecordingStageControlPlane()
        control_plane.log = log
        await relay_io.send_stream_chunk(
            stream_relay,
            control_plane,
            request_id="req-1",
            data=torch.tensor([1, 2, 3]),
            target_stage="talker",
            target_endpoint="inproc://talker",
            from_stage="thinker",
            chunk_id=0,
            metadata={"token_id": 1, "hidden": torch.tensor([4])},
        )

        names = collect_event_names(log)
        assert names.index("stage_cp_send_to_stage") < names.index("op_wait")
        msg = control_plane.sent_to_stage[0][2]
        assert msg.shm_metadata["chunk_metadata"]["token_id"] == 1
        assert "hidden" in msg.shm_metadata["chunk_metadata_tensors"]

    asyncio.run(_run())


def test_aggregated_input_waits_per_request_without_cross_talk() -> None:
    handler = AggregatedInput(
        {"preprocess", "image"},
        lambda payloads: make_stage_payload(data={"sources": sorted(payloads)}),
    )

    assert handler.receive("req-1", "preprocess", make_stage_payload()) is None
    assert handler.receive("req-2", "preprocess", make_stage_payload()) is None
    req2 = handler.receive("req-2", "image", make_stage_payload())
    req1 = handler.receive("req-1", "image", make_stage_payload())

    assert req2.data == {"sources": ["image", "preprocess"]}
    assert req1.data == {"sources": ["image", "preprocess"]}


def test_simple_scheduler_batch_and_error_contracts() -> None:
    good = SimpleScheduler(
        lambda payload: payload,
        batch_compute_fn=lambda payloads: [payload.upper() for payload in payloads],
        max_batch_size=2,
        max_batch_wait_ms=10,
    )
    outputs = run_scheduler(
        good,
        [
            IncomingMessage("req-1", "new_request", "a"),
            IncomingMessage("req-2", "new_request", "b"),
        ],
        output_count=2,
    )
    assert {out.data for out in outputs} == {"A", "B"}

    bad = SimpleScheduler(
        lambda payload: payload,
        batch_compute_fn=lambda payloads: ["only-one"],
        max_batch_size=2,
        max_batch_wait_ms=10,
    )
    outputs = run_scheduler(
        bad,
        [
            IncomingMessage("req-1", "new_request", "a"),
            IncomingMessage("req-2", "new_request", "b"),
        ],
        output_count=2,
    )
    assert {out.request_id for out in outputs} == {"req-1", "req-2"}
    assert all(
        out.type == "error" and isinstance(out.data, ValueError) for out in outputs
    )


def test_mp_runner_preserves_tp_rank_and_visible_device_contracts(
    tmp_path: Path,
) -> None:
    config = PipelineConfig(
        model_path="model",
        name="mp",
        endpoints=EndpointsConfig(base_path=str(tmp_path)),
        relay_backend="nccl",
        stages=[
            stage(
                "thinker",
                factory=fake_factory_path("make_scheduler_accepting_gpu_id"),
                gpu=[1, 3],
                tp_size=2,
                terminal=True,
            )
        ],
    )

    group = _build_stage_groups(config, ctx=FakeMpContext())[0]
    leader, follower = group.specs
    env = get_stage_process_env(follower, env={"CUDA_VISIBLE_DEVICES": "4,5,6,7"})

    assert leader.role == "leader"
    assert follower.role == "follower"
    assert leader.factory_args["tp_rank"] == 0
    assert follower.factory_args["tp_rank"] == 1
    assert leader.factory_args["nccl_port"] == follower.factory_args["nccl_port"]
    assert env["CUDA_VISIBLE_DEVICES"] == "7"
