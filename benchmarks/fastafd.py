# SPDX-License-Identifier: Apache-2.0
"""The baseline afdEP-comm is measured against: FastAFD's DeepEP M2N.

FastAFD (https://github.com/hao-ai-lab/FastAFD) runs AFD over DeepEP V2 with its
``DeepEPM2NAdapter`` (python/minisgl/moe/deepep_m2n_adapter.py): DeepEP sees a
symmetric expert-parallel group, every rank sends, receives and waits in both
phases, and Attention ranks hold dummy expert slots that no token routes to. The
AFD plugin's ``deepep_m2n`` connector (https://github.com/ms-llmd/afd-plugin) ships
the same snapshot and layout. This module drives the same contract over the DeepEP
V2 kernels vendored here, unspecialized, behind afdEP-comm's role API, so both arms
share one build and harness. docs/validation-fastafd-equivalence.json checks it
against the plugin's own buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from afdep import Buffer, Config
from afdep.deepep import ElasticBuffer

if TYPE_CHECKING:
    import torch.distributed as dist

# The arm override that selects this baseline in benchmarks.comm and benchmarks.screen.
BASELINE_KEY, BASELINE = "_baseline", "fastafd"


@dataclass(frozen=True)
class FastAFDConfig(Config):
    """Config whose receive bound counts every rank of the symmetric group."""

    def receive_rows(self, bucket: int) -> int:
        return self._receive_rows(self.world_size, bucket)


class FastAFDBuffer(Buffer):
    """afdEP-comm's role API over unspecialized DeepEP V2, as FastAFD runs it."""

    def _open_transport(self, group: dist.ProcessGroup) -> ElasticBuffer:
        config = self.config
        # No ffn_ranks: DeepEP V2 as upstream ships it, over a symmetric group.
        return ElasticBuffer(
            group=group,
            capacity_tokens_per_rank=config.capacity_tokens_per_rank,
            hidden_size=config.hidden_size,
            num_experts=config.logical_experts,
            top_k=config.top_k,
            allow_hybrid_mode=config.allow_hybrid_mode,
            rdma_traffic_class=config.rdma_traffic_class,
            cpu_timeout_s=config.cpu_timeout_s,
            gpu_timeout_s=config.gpu_timeout_s,
            use_fp8_dispatch=config.payload == "fp8",
            num_sms=config.num_sms,
            num_qps=config.num_qps,
        )

    def _source_output_rows(self) -> int:
        # As a member of the symmetric group, a source combines a full receive bound.
        return self.config.receive_rows(self.config.capacity_tokens_per_rank)

    def source_dispatch(self, x, ids, weights, x_scale=None, **kwargs) -> None:
        # Upstream DeepEP routes logical experts. A per-domain FFN prefix puts FFN
        # blocks off [0, num_experts), so the baseline maps real IDs as a caller would.
        if self.config.ffn_ranks_per_domain:
            ids = self.config.logical_ids(ids)
        super().source_dispatch(x, ids, weights, x_scale, **kwargs)


def make(group: dist.ProcessGroup, baseline: str | None, **config) -> Buffer:
    """Build afdEP-comm's Buffer, or the named baseline's."""
    if baseline is None:
        return Buffer(group=group, config=Config(**config))
    if baseline != BASELINE:
        raise ValueError(f"unknown baseline {baseline!r}; expected {BASELINE!r}")
    return FastAFDBuffer(group=group, config=FastAFDConfig(**config))
