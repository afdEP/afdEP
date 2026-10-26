# SPDX-License-Identifier: Apache-2.0
"""Negative control: one rank sets a wire-visible AFDEP_* control, the others do not.

For each control in afdep.deepep.PROTOCOL_CONTROLS, rank 1 opens its Buffer with the
control set; every rank must raise ValueError naming it, with no hang. A Buffer opened
afterwards with matching controls must still exchange correctly.

Example: EP_DISABLE_GIN=1 torchrun --standalone --nproc-per-node=4 \
    -m tests.gpu.protocol_agreement
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

from afdep._build import source_digest
from afdep.deepep import PROTOCOL_CONTROLS

from .deepep import verify
from .two_buffers import cases, exchange_for, open_buffer


def main() -> None:
    args = SimpleNamespace(
        payload="bf16", capacity=8, hidden_size=256, ffn_ranks=2, num_sms=8
    )
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=300))
    rank, world = dist.get_rank(), dist.get_world_size()
    rejected = []
    for control in PROTOCOL_CONTROLS:
        if rank == 1:
            os.environ[control] = "1"
        try:
            open_buffer(args, world).close()
        except ValueError as error:
            assert control in str(error), (control, str(error))
            rejected.append(control)
        else:
            raise AssertionError(f"{control} on one rank was accepted")
        finally:
            os.environ.pop(control, None)
        dist.barrier()
    buffer = open_buffer(args, world)
    for case in cases(buffer, 1):
        exchange = exchange_for(buffer, case)
        exchange.update(case)
        snapshot = exchange.run(delay=True)
        torch.cuda.synchronize()
        verify(snapshot, case, buffer, exchange.rows)
    buffer.close()
    summaries = [None] * world
    dist.all_gather_object(summaries, {"rank": rank, "rejected": rejected})
    dist.destroy_process_group()
    if rank == 0:
        print(
            json.dumps(
                {
                    "ok": True,
                    "validation": "protocol agreement",
                    "gpu": torch.cuda.get_device_name(),
                    "native_source_sha256": source_digest(),
                    "ranks": summaries,
                }
            )
        )


if __name__ == "__main__":
    main()
