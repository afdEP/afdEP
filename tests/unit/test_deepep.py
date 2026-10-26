# SPDX-License-Identifier: Apache-2.0
"""Host contracts for the DeepEP facade; these mocks do not validate transport."""

from __future__ import annotations

import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from afdep import deepep
from benchmarks.fastafd import FastAFDBuffer, FastAFDConfig


class Tensor:
    def __init__(self, shape, dtype="bf16", device=0, contiguous=True):
        self.shape, self.dtype = tuple(shape), dtype
        self.ndim, self.is_cuda = len(shape), True
        self.device, self.contiguous_storage = device, contiguous
        self.view_parent = None

    def get_device(self):
        return self.device

    def is_contiguous(self):
        return self.contiguous_storage

    def contiguous(self):
        assert self.contiguous_storage, "test unexpectedly requested a payload copy"
        return self

    def to(self, dtype):
        assert self.dtype == dtype, "test unexpectedly requested a dtype conversion"
        return self

    def float(self):
        return self.to("fp32")

    def __getitem__(self, index):
        result = Tensor((index.stop, *self.shape[1:]), self.dtype, self.device)
        result.view_parent = self
        return result


@pytest.fixture
def runtime(monkeypatch):
    torch, dist = ModuleType("torch"), ModuleType("torch.distributed")
    cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        capturing=False,
        is_current_stream_capturing=lambda: cuda.capturing,
        get_device_properties=lambda _: SimpleNamespace(multi_processor_count=132),
        synchronize=Mock(),
    )
    torch.cuda, torch.distributed = cuda, dist
    torch.bfloat16, torch.float32, torch.float8_e4m3fn = "bf16", "fp32", "fp8"
    torch.device = lambda _, index: index
    torch.empty = torch.zeros = lambda shape, dtype, device: Tensor(
        shape, dtype, device
    )
    dist.is_initialized, dist.get_backend = lambda: True, lambda _: "gloo"
    dist.barrier = Mock()

    def gather(values, value, group):
        values[:] = [value] * group.size()
        if value is None:
            values[0] = b"root-id"

    dist.all_gather_object = gather
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist)
    group = SimpleNamespace(rank=lambda: 2, size=lambda: 4)
    native = Mock()
    native.get_logical_domain_size.return_value = (1, 4)
    native.get_afd_count_first_config.return_value = (0, "full")
    native.get_afd_sparse_hybrid_config.return_value = (0, False, False, 0, False, 0, 0)
    native.combine.return_value = (Tensor((2, 256)), None, None)

    def dispatch(*args):
        constructor = module.ElasticBuffer.call_args.args
        rows = 0 if len(constructor) >= 14 and group.rank() >= constructor[13] else 4
        scale = (
            Tensor((rows, 2), "fp32", contiguous=False) if args[1] is not None else None
        )
        return (
            Tensor((rows, 256), args[0].dtype),
            scale,
            None,
            Tensor((rows,), "fp32"),
            args[2],
            [],
            Tensor((4,), "i32"),
            Tensor((8,), "i32"),
            Tensor((rows, 18), "i32"),
            Tensor((0,), "i32"),
            None,
            None,
            None,
        )

    native.dispatch.side_effect = dispatch
    module = SimpleNamespace(
        topk_idx_t="i64",
        get_local_nccl_unique_id=Mock(return_value=b"root-id"),
        create_nccl_comm=Mock(return_value=17),
        destroy_nccl_comm=Mock(),
        get_physical_domain_size=Mock(return_value=(1, 4)),
        calculate_elastic_buffer_size=Mock(return_value=4096),
        ElasticBuffer=Mock(return_value=native),
    )
    monkeypatch.setattr(deepep, "load_deepep", lambda: module)
    monkeypatch.setattr(deepep, "source_digest", lambda: "pinned-source")
    return SimpleNamespace(
        torch=torch, dist=dist, group=group, native=native, module=module
    )


def elastic(runtime, **kwargs):
    settings = dict(
        group=runtime.group,
        capacity_tokens_per_rank=16,
        hidden_size=256,
        num_experts=32,
        top_k=16,
        allow_hybrid_mode=True,
        rdma_traffic_class=0,
        cpu_timeout_s=120,
        gpu_timeout_s=120,
    )
    return deepep.ElasticBuffer(**(settings | kwargs))


def config(**kwargs):
    settings = dict(
        capacity_tokens_per_rank=16,
        hidden_size=256,
        num_experts=16,
        top_k=16,
        ffn_ranks=2,
        attention_ranks=2,
    )
    return deepep.DeepEPConfig(**(settings | kwargs))


def test_real_and_logical_expert_counts():
    value = config()
    assert value.experts_per_rank == 8
    assert value.logical_experts == 32
    assert value.top_k > value.experts_per_rank
    with pytest.raises(ValueError, match="divide evenly"):
        config(num_experts=15)


def test_per_domain_ffn_prefix_numbers_roles_and_experts():
    # Two NVLink domains of 5 ranks, local rank 0 of each is FFN: FFN ranks {0, 5}.
    value = config(ffn_ranks=2, attention_ranks=8, ffn_ranks_per_domain=1)
    assert [r for r in range(10) if value.is_ffn(r)] == [0, 5]
    assert [value.role_rank(r) for r in range(10)] == [0, 0, 1, 2, 3, 1, 4, 5, 6, 7]
    assert [value.ffn_rank(j) for j in range(2)] == [0, 5]
    # Real expert e of FFN index j lands in DeepEP's block of rank ffn_rank(j).
    assert value.experts_per_rank == 8
    assert [value.logical_ids(e) for e in (-1, 0, 7, 8, 15)] == [-1, 0, 7, 40, 47]
    for e in range(value.num_experts):
        index = e // value.experts_per_rank
        assert value.logical_ids(e) // value.experts_per_rank == value.ffn_rank(index)
    # The global prefix, also when one domain holds every FFN rank.
    for prefix in (config(), config(ffn_ranks_per_domain=2)):
        assert [prefix.is_ffn(r) for r in range(4)] == [True, True, False, False]
        assert [prefix.role_rank(r) for r in range(4)] == [0, 1, 0, 1]
        assert [prefix.logical_ids(e) for e in (-1, 0, 15)] == [-1, 0, 15]
    for f, message in ((3, "split ffn_ranks"), (True, "nonnegative")):
        with pytest.raises(ValueError, match=message):
            config(ffn_ranks=2, attention_ranks=8, ffn_ranks_per_domain=f)


def test_per_domain_prefix_checks_the_physical_domains(runtime, monkeypatch):
    monkeypatch.delenv("EP_DISABLE_GIN", raising=False)
    runtime.module.get_physical_domain_size.return_value = (2, 2)
    buffer = elastic(runtime, ffn_ranks=2, ffn_ranks_per_domain=1)
    assert runtime.module.ElasticBuffer.call_args.args[-2:] == (2, 1)
    assert runtime.module.calculate_elastic_buffer_size.call_args.args[-2:] == (2, 1)
    assert [buffer._is_ffn(r) for r in range(4)] == [True, False, True, False]
    buffer.close()
    # One domain holding every FFN rank is the global prefix: the 14-argument ctor.
    runtime.module.get_physical_domain_size.return_value = (1, 4)
    buffer = elastic(runtime, ffn_ranks=2, ffn_ranks_per_domain=2)
    assert len(runtime.module.ElasticBuffer.call_args.args) == 14
    assert runtime.module.calculate_elastic_buffer_size.call_args.args[-1] == 0
    buffer.close()
    runtime.module.ElasticBuffer.reset_mock()
    with pytest.raises(ValueError, match="needs 2 NVLink domains of 2 ranks"):
        elastic(runtime, ffn_ranks=2, ffn_ranks_per_domain=1)
    # f == F asks for every FFN rank in one domain, which two domains contradict.
    runtime.module.get_physical_domain_size.return_value = (2, 2)
    with pytest.raises(ValueError, match="needs 1 NVLink domains of 4 ranks"):
        elastic(runtime, ffn_ranks=2, ffn_ranks_per_domain=2)
    runtime.module.ElasticBuffer.assert_not_called()
    with pytest.raises(RuntimeError, match="need hybrid mode"):
        elastic(runtime, ffn_ranks=2, ffn_ranks_per_domain=1, allow_hybrid_mode=False)
    with pytest.raises(RuntimeError, match="requires ffn_ranks"):
        elastic(runtime, ffn_ranks_per_domain=1)


def test_per_domain_buffer_roles(runtime, monkeypatch):
    monkeypatch.delenv("EP_DISABLE_GIN", raising=False)
    runtime.module.get_physical_domain_size.return_value = (2, 2)
    value = config(ffn_ranks_per_domain=1)
    buffer = deepep.DeepEPBuffer(group=runtime.group, config=value)
    # Rank 2 is local rank 0 of the second domain: FFN index 1.
    assert (buffer.role, buffer.role_rank) == ("ffn", 1)
    assert runtime.module.ElasticBuffer.call_args.args[-2:] == (2, 1)
    buffer.close()
    runtime.group.rank = lambda: 1
    buffer = deepep.DeepEPBuffer(group=runtime.group, config=value)
    assert (buffer.role, buffer.role_rank) == ("attention", 0)
    buffer.close()


def test_receive_bounds_count_attention_sources_only():
    afdep = config(capacity_tokens_per_rank=2048)
    # FastAFD's symmetric group counts every rank as a source.
    fastafd = FastAFDConfig(**vars(afdep))
    assert fastafd.receive_rows(2048) == 66_560
    assert afdep.receive_rows(2048) == 33_792
    assert afdep.logical_experts == fastafd.logical_experts
    assert not hasattr(afdep, "role_aware")


def test_ffn_ranks_select_the_afd_transport_and_keep_the_upstream_ctor(runtime):
    upstream = elastic(runtime)
    assert len(runtime.module.ElasticBuffer.call_args.args) == 13
    # Symmetric sizing without ffn_ranks (the baseline); role-sized inboxes with it.
    assert runtime.module.calculate_elastic_buffer_size.call_args.args[-2:] == (0, 0)
    upstream.close()
    roles = elastic(runtime, ffn_ranks=2)
    assert len(runtime.module.ElasticBuffer.call_args.args) == 14
    assert runtime.module.ElasticBuffer.call_args.args[-1] == 2
    assert runtime.module.calculate_elastic_buffer_size.call_args.args[-2:] == (2, 0)
    roles.close()


def test_count_first_configuration_must_agree(runtime):
    gather = runtime.dist.all_gather_object

    def split(values, value, group):
        gather(values, value, group)
        if isinstance(value, tuple):
            values[1] = (1, "full")

    runtime.dist.all_gather_object = split
    with pytest.raises(ValueError, match="same count-first configuration"):
        elastic(runtime, ffn_ranks=2)
    runtime.native.destroy.assert_called_once()
    runtime.module.destroy_nccl_comm.assert_called_once_with(17)


def disagree_on_hybrid_configuration(runtime, other):
    gather = runtime.dist.all_gather_object

    def split(values, value, group):
        gather(values, value, group)
        if value == (0, False, False, 0, False, 0, 0):
            values[1] = other

    runtime.dist.all_gather_object = split


def test_per_domain_protocol_must_agree(runtime):
    # AFDEP_UNION_HYBRID=1 on one rank only: silent FFN rails there, union elsewhere.
    disagree_on_hybrid_configuration(runtime, (0, False, False, 0, False, 0, 1))
    with pytest.raises(ValueError, match="same hybrid configuration"):
        elastic(runtime, ffn_ranks=2)
    runtime.native.destroy.assert_called_once()


def test_sparse_hybrid_configuration_must_agree(runtime):
    disagree_on_hybrid_configuration(runtime, (0, False, False, 2, False, 0, 0))
    with pytest.raises(ValueError, match="same hybrid configuration"):
        elastic(runtime, ffn_ranks=2)
    runtime.native.destroy.assert_called_once()
    runtime.module.destroy_nccl_comm.assert_called_once_with(17)


def test_fp8_combine_configuration_must_agree(runtime):
    # A rank that latched FP8 combine replies would mis-decode its peers' BF16 rows.
    disagree_on_hybrid_configuration(runtime, (0, False, False, 0, True, 0))
    with pytest.raises(ValueError, match="same hybrid configuration"):
        elastic(runtime, ffn_ranks=2)
    runtime.native.destroy.assert_called_once()
    runtime.module.destroy_nccl_comm.assert_called_once_with(17)


def test_fp8_combine_is_reported_when_every_rank_latched_it(runtime):
    assert elastic(runtime, ffn_ranks=2).fp8_combine is False
    runtime.native.get_afd_sparse_hybrid_config.return_value = (
        0,
        True,
        True,
        0,
        True,
        64,
        0,
    )
    transport = elastic(runtime, ffn_ranks=2)
    assert transport.fp8_combine is True and transport.fp8_combine_min_bucket == 64
    buffer = deepep.DeepEPBuffer(group=runtime.group, config=config())
    assert buffer.fp8_combine is True and buffer.fp8_combine_min_bucket == 64


def test_fp8_combine_min_bucket_must_agree(runtime):
    runtime.native.get_afd_sparse_hybrid_config.return_value = (
        0,
        False,
        False,
        0,
        True,
        0,
        0,
    )
    gather = runtime.dist.all_gather_object

    def split(values, value, group):
        gather(values, value, group)
        if value == (0, False, False, 0, True, 0, 0):
            values[1] = (0, False, False, 0, True, 64, 0)

    runtime.dist.all_gather_object = split
    with pytest.raises(ValueError, match="same hybrid configuration"):
        elastic(runtime, ffn_ranks=2)
    runtime.native.destroy.assert_called_once()


def test_count_first_arm_is_rejected_collectively(runtime):
    runtime.native.get_afd_count_first_config.return_value = (1, "both")
    with pytest.raises(
        ValueError, match="AFDEP_COUNT_FIRST_ARM must be full or source"
    ):
        elastic(runtime, ffn_ranks=2)
    runtime.native.destroy.assert_called_once()
    runtime.module.destroy_nccl_comm.assert_called_once_with(17)


def test_source_owns_no_dummy_payload_arena(runtime):
    buffer = deepep.DeepEPBuffer(group=runtime.group, config=config())
    assert buffer._zero_output.shape == (0, 256)
    x, ids, weights = Tensor((2, 256)), Tensor((2, 16), "i64"), Tensor((2, 16), "fp32")
    buffer.source_dispatch(x, ids, weights, bucket=8)
    assert buffer.transport._pending.recv_shape == (0, 256)
    buffer.source_collect()
    assert runtime.native.combine.call_args.args[0].shape == (0, 256)


def test_ffn_worker_rejects_source_payload(runtime):
    runtime.group.rank = lambda: 0
    buffer = elastic(runtime, ffn_ranks=2)
    with pytest.raises(ValueError, match="cannot originate tokens"):
        buffer.dispatch(
            Tensor((2, 256)),
            None,
            Tensor((2, 16), "i64"),
            Tensor((2, 16), "fp32"),
            bucket=8,
            expert_alignment=128,
            do_cpu_sync=False,
        )
    runtime.native.dispatch.assert_not_called()


def test_fastafd_baseline_runs_upstream_deepep_with_full_source_bounds(runtime):
    buffer = FastAFDBuffer(group=runtime.group, config=FastAFDConfig(**vars(config())))
    assert len(runtime.module.ElasticBuffer.call_args.args) == 13
    assert runtime.module.calculate_elastic_buffer_size.call_args.args[-2:] == (0, 0)
    assert buffer._zero_output.shape == (buffer.config.receive_rows(16), 256)
    buffer.close()


def test_facade_accepts_topk_above_local_experts_and_owns_bootstrap(
    runtime, monkeypatch
):
    monkeypatch.delenv("EP_DISABLE_GIN", raising=False)
    buffer = elastic(runtime)
    assert buffer.top_k == 16  # Eight logical experts/rank; 16 choices is valid.
    runtime.module.create_nccl_comm.assert_called_once_with(b"root-id", 4, 2)
    assert buffer.physical_domain == (1, 4)
    assert buffer.logical_domain == (1, 4)
    assert "EP_DISABLE_GIN" not in deepep.os.environ
    assert runtime.module.calculate_elastic_buffer_size.call_args.args[1] == 17
    buffer.close()
    buffer.close()
    runtime.native.destroy.assert_called_once()
    runtime.module.destroy_nccl_comm.assert_called_once_with(17)


def test_explicit_disable_rejects_multidomain_before_buffer_creation(
    runtime, monkeypatch
):
    monkeypatch.setenv("EP_DISABLE_GIN", "1")
    runtime.module.get_physical_domain_size.return_value = (2, 2)
    with pytest.raises(ValueError, match="Multiple NVLink domains require GIN"):
        elastic(runtime)
    runtime.module.ElasticBuffer.assert_not_called()
    runtime.module.destroy_nccl_comm.assert_called_once_with(17)


def test_afd_transport_accepts_multidomain_with_gin(runtime, monkeypatch):
    monkeypatch.delenv("EP_DISABLE_GIN", raising=False)
    runtime.module.get_physical_domain_size.return_value = (2, 2)
    buffer = elastic(runtime, ffn_ranks=2)
    assert buffer.physical_domain == (2, 2)
    assert runtime.module.ElasticBuffer.call_args.args[-1] == 2
    buffer.close()


def test_opaque_handle_is_required_and_native_output_is_returned(runtime):
    buffer = elastic(runtime)
    x, ids, weights = Tensor((2, 256)), Tensor((2, 16), "i64"), Tensor((2, 16), "fp32")
    received = buffer.dispatch(
        x, None, ids, weights, bucket=8, expert_alignment=128, do_cpu_sync=False
    )
    args = runtime.native.dispatch.call_args.args
    assert args[:4] == (x, None, ids, weights)
    assert args[12:15] == (8, 32, 128)
    assert args[22:] == (False, True, True)
    assert received.receipt is received.handle
    with pytest.raises(ValueError, match="pending exchange"):
        buffer.combine(Tensor((4, 256)), replace(received.handle, _owner=object()))
    with pytest.raises(RuntimeError, match="previous exchange"):
        buffer.dispatch(
            x, None, ids, weights, bucket=8, expert_alignment=128, do_cpu_sync=False
        )
    rows = Tensor((4, 256))
    output = buffer.combine(rows, received.handle)
    assert output is runtime.native.combine.return_value[0]
    assert runtime.native.combine.call_args.args[0] is rows
    with pytest.raises(ValueError, match="pending exchange"):
        buffer.combine(rows, received.handle)


def test_capture_rejects_host_counts_and_close_with_pending_exchange(runtime):
    buffer = elastic(runtime)
    x, ids, weights = Tensor((2, 256)), Tensor((2, 16), "i64"), Tensor((2, 16), "fp32")
    runtime.torch.cuda.capturing = True
    with pytest.raises(RuntimeError, match="Host receive counts"):
        buffer.dispatch(
            x, None, ids, weights, bucket=8, expert_alignment=128, do_cpu_sync=True
        )
    runtime.native.dispatch.assert_not_called()
    runtime.torch.cuda.capturing = False
    buffer.dispatch(
        x, None, ids, weights, bucket=8, expert_alignment=128, do_cpu_sync=False
    )
    with pytest.raises(RuntimeError, match="pending exchange"):
        buffer.close()
    runtime.native.destroy.assert_not_called()


def test_role_source_returns_native_output_without_copy(runtime):
    buffer = deepep.DeepEPBuffer(group=runtime.group, config=config())
    x, ids, weights = Tensor((2, 256)), Tensor((2, 16), "i64"), Tensor((2, 16), "fp32")
    buffer.source_dispatch(x, ids, weights, bucket=8)
    output = buffer.source_collect()
    assert output is runtime.native.combine.return_value[0]
    assert runtime.native.combine.call_args.args[0].view_parent is buffer._zero_output
    assert runtime.native.combine.call_args.args[-1] is False
    buffer.source_dispatch(x, ids, weights, bucket=8)
    buffer.source_collect(view=True)
    assert runtime.native.combine.call_args.args[-1] is True
    with pytest.raises(RuntimeError, match="ffn role"):
        buffer.worker_receive(bucket=8)
    with pytest.raises(ValueError, match="dtype"):
        buffer.source_dispatch(x, ids, Tensor((2, 16)), bucket=8)


def test_role_worker_dispatches_empty_and_preserves_receipt(runtime):
    runtime.group.rank = lambda: 0
    buffer = deepep.DeepEPBuffer(group=runtime.group, config=config(payload="fp8"))
    received = buffer.worker_receive(bucket=8)
    args = runtime.native.dispatch.call_args.args
    assert args[0].shape == (0, 256)
    assert args[0].dtype == "fp8"
    assert args[1].shape == (0, 2)
    assert args[13] == 32
    assert received.recv_scale is not None
    rows = Tensor(received.recv_x.shape)
    buffer.worker_reply(rows, received.receipt)
    assert runtime.native.combine.call_args.args[0] is rows


def test_public_api_is_the_afd_transport_without_loading_torch():
    from afdep import Buffer, Config

    value = Config(2048, 4096, 128, 8, 2, 2)
    assert value.receive_rows(2048) < FastAFDConfig(**vars(value)).receive_rows(2048)
    assert Buffer is deepep.DeepEPBuffer
    assert Config is deepep.DeepEPConfig


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"capacity_tokens_per_rank": 0}, "positive int32"),
        ({"ffn_ranks": True}, "positive int32"),
        ({"attention_ranks": 0}, "positive int32"),
        ({"hidden_size": 257}, "multiple of 256"),
        ({"top_k": 17}, "top_k"),
        ({"num_sms": -1}, "nonnegative int32"),
        ({"num_qps": 1.5}, "nonnegative int32"),
        ({"allow_hybrid_mode": 1}, "boolean"),
        ({"payload": "fp16"}, "payload"),
        ({"rdma_traffic_class": 256}, "traffic_class"),
        ({"capacity_tokens_per_rank": (1 << 31) - 2}, "int32 indexing"),
    ],
)
def test_invalid_configuration_fails_before_gpu_initialization(settings, message):
    with pytest.raises(ValueError, match=message):
        config(**settings)
