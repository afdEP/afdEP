# SPDX-License-Identifier: Apache-2.0
"""Two Buffers in one process, every rank enqueuing their exchanges in one order.

Buffer A is opened under the process environment, Buffer B under it plus --env-b,
which stays set afterwards (as an in-process A/B harness leaves it). Each Buffer
must keep the launch-shape controls it was opened with. Eager exchanges alternate
A and B; then one graph per Buffer replays alternately, one graph captures
both Buffers' exchanges interleaved, and two graphs captured on Buffer A replay
alternately.

Example: EP_DISABLE_GIN=1 torchrun --standalone --nproc-per-node=4 \
    -m tests.gpu.two_buffers --env-b AFDEP_UNION_BUFFERS=1
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from afdep import Buffer, Config
from afdep._build import source_digest

from .deepep import RECORDED_ENV_PREFIXES, Case, Exchange, verify

STEPS = 4


def open_buffer(args, world: int) -> Buffer:
    config = Config(
        capacity_tokens_per_rank=args.capacity,
        hidden_size=args.hidden_size,
        num_experts=4 * args.ffn_ranks,
        top_k=2,
        ffn_ranks=args.ffn_ranks,
        attention_ranks=world - args.ffn_ranks,
        payload=args.payload,
        num_sms=args.num_sms,
    )
    return Buffer(group=dist.group.WORLD, config=config)


def cases(buffer: Buffer, epoch: int) -> list[Case]:
    n, sources = buffer.config.capacity_tokens_per_rank, buffer.config.attention_ranks
    prefix = tuple(min((3, 1, 2)[i % 3], n) for i in range(sources))
    return [
        Case("mixed", (n,) * sources, epoch),
        Case("skew", (n,) * sources, epoch + 1, "skew"),
        Case("partial_prefix", prefix, epoch + 2),
        Case("all_empty", (0,) * sources, epoch + 3),
    ]


def exchange_for(buffer: Buffer, case: Case) -> Exchange:
    if buffer.role == "attention":
        return Exchange(buffer, case.counts[buffer.role_rank])
    return Exchange(buffer, buffer.config.capacity_tokens_per_rank)


def eager_checks(buffers) -> int:
    """Alternate A and B without host synchronization between exchanges."""
    plans = []
    for index in range(len(cases(buffers[0], 0))):
        for epoch, buffer in enumerate(buffers):
            case = cases(buffer, 30 + 10 * epoch)[index]
            exchange = exchange_for(buffer, case)
            exchange.update(case)
            plans.append((buffer, case, exchange, exchange.run(delay=True)))
    torch.cuda.synchronize()
    for buffer, case, exchange, snapshot in plans:
        verify(snapshot, case, buffer, exchange.rows)
    dist.barrier()
    return len(plans)


def graph_checks(buffers, *, interleaved: bool) -> int:
    """One graph per Buffer replayed alternately, or one graph holding both."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    exchanges = [
        Exchange(buffer, buffer.config.capacity_tokens_per_rank) for buffer in buffers
    ]
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for exchange, buffer in zip(exchanges, buffers, strict=True):
            exchange.update(cases(buffer, 60)[0])
            exchange.run()
    torch.cuda.synchronize()
    dist.barrier()
    graphs, captured = [], []
    for group in [exchanges] if interleaved else [[e] for e in exchanges]:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            # Interleaved: A, B, A, B... in one capture.
            layers = [e.run(alternate=True) for _ in range(STEPS) for e in group]
        graphs.append(graph)
        captured.append((group, layers))
    checked = []
    with torch.cuda.stream(stream):
        for index in range(len(cases(buffers[0], 0))):
            for graph, (group, layers) in zip(graphs, captured, strict=True):
                group_cases = []
                for exchange in group:
                    epoch = 70 + 10 * buffers.index(exchange.buffer)
                    case = cases(exchange.buffer, epoch)[index]
                    exchange.update(case)
                    group_cases.append(case)
                graph.replay()
                snapshots = [(value.clone(), counts) for value, counts in layers]
                checked.append((group, group_cases, snapshots))
    torch.cuda.synchronize()
    total = 0
    for group, group_cases, snapshots in checked:
        for step, snapshot in enumerate(snapshots):
            exchange, case = group[step % len(group)], group_cases[step % len(group)]
            # Each Buffer's inputs flip sign before every one of its exchanges.
            sign = -1 if (step // len(group)) % 2 == 0 else 1
            verify(snapshot, case, exchange.buffer, exchange.rows, sign=sign)
            total += 1
    for graph in graphs:
        graph.reset()
    dist.barrier()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", choices=("bf16", "fp8"), default="bf16")
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--ffn-ranks", type=int, default=2)
    parser.add_argument("--num-sms", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--env-b",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Set on every rank before opening Buffer B and left set",
    )
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=300))
    world = dist.get_world_size()
    env_a = {k: v for k, v in os.environ.items() if k.startswith(RECORDED_ENV_PREFIXES)}
    buffers = [open_buffer(args, world)]
    for setting in args.env_b:
        key, _, value = setting.partition("=")
        os.environ[key] = value
    buffers.append(open_buffer(args, world))
    counts = dict.fromkeys(
        ("eager", "graph_alternating", "graph_interleaved", "two_graphs_one_buffer"), 0
    )
    for _ in range(args.rounds):
        counts["eager"] += eager_checks(buffers)
        counts["graph_alternating"] += graph_checks(buffers, interleaved=False)
        counts["graph_interleaved"] += graph_checks(buffers, interleaved=True)
        counts["two_graphs_one_buffer"] += graph_checks(
            [buffers[0]] * 2, interleaved=False
        )
    summary = {
        "rank": buffers[0].rank,
        "role": buffers[0].role,
        "gpu": torch.cuda.get_device_name(),
        "native_source_sha256": source_digest(),
        "env_a": env_a,
        "env_b": args.env_b,
        "registered_transport_bytes": [b.num_buffer_bytes for b in buffers],
        "checked_exchanges": counts,
    }
    summaries = [None] * world
    dist.all_gather_object(summaries, summary)
    for buffer in buffers:
        buffer.close()
    dist.destroy_process_group()
    if summary["rank"] == 0:
        print(json.dumps({"ok": True, "validation": "two Buffers", "ranks": summaries}))


if __name__ == "__main__":
    main()
