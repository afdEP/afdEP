# Performance and validation

## Provenance and scope

The numbers on this page come from the directional prototype measured on
October 15, 2026, before extraction into this standalone repository. They are
included to explain the design decisions. They are **not results from the new
AFDEP package** and do not establish a newly measured package-versus-DeepEP
comparison. The extracted wheel has since passed standalone eight-H100 BF16/FP8
eager/graph validation and short communication benchmarks; see the separate
[October 18 validation record](validation-2026-10-18.md). Its smoke workload,
capacity, and run length differ from the historical comparison below.

The baseline was Ronen Kat's masked DeepEP connector in
[AFD plugin PR #23](https://github.com/ms-llmd/afd-plugin/pull/23), at commit
`7cc7ebd2283eff0388a7aa9495a9445a1cbd8349`. The directional prototype used its
own request/reply data plane, with DeepEP infrastructure for bootstrap and JIT.
AFDEP extracts that directional implementation into its own extension and
bootstrap wrapper, retaining only minimal attributed transport helpers. It does
not retain the prototype's DeepEP JIT or module loader.

## Measurement setup

- Two four-GPU H100 80GB NVLink domains: four FFN workers and four Attention
  sources, with one active request/reply lane.
- Hidden size 2048, 128 real experts, top-k eight, expert alignment 128, and
  capacity 8192 tokens per source.
- BF16 or prequantized FP8 E4M3 block-128 activation dispatch with FP32 scales;
  BF16 replies in both cases.
- Eight GIN contexts and eight directional transport stripes. The network setup
  used one selected RoCE rail; these results do not establish peak aggregate NIC
  bandwidth.
- PyTorch 2.13.0+cu130, CUDA 13.0, NCCL 2.30.7, and Nsight Systems 2026.5.1 in
  the historical environment. This is a recorded setup, not an exhaustive
  compatibility guarantee for the standalone package.

The timed exchange includes real activations and routing metadata, but replaces
expert GEMMs with persistent BF16 output rows. Routing, launch skew,
synchronization, and communication remain in the measured host-wall time.
Separate correctness scenarios exercised expert compute. These are
communication-only results, not serving throughput or end-to-end model latency.

## Latency

CUDA-graph round-trip latency, milliseconds:

| Payload | Tokens/source | Masked DeepEP baseline | Directional, tuned local grid |
|---|---:|---:|---:|
| BF16 | 1 | 0.635 | 0.432 |
| BF16 | 128 | 0.797 | 0.562 |
| BF16 | 512 | 1.606 | 1.119 |
| BF16 | 2048 | 3.791 | 3.698 |
| FP8 | 1 | 0.678 | 0.454 |
| FP8 | 128 | 0.773 | 0.562 |
| FP8 | 512 | 1.495 | 1.043 |
| FP8 | 2048 | 3.582 | 3.145 |

The metric is the median across three runs of each run's median aligned
iteration maximum across Attention ranks. It is not a device kernel duration
or wire-only latency. Each timed bucket used five warmups and 100 iterations.

The original directional local grids used 24 CTAs and were about 2.2× slower at
2048 tokens. Increasing local grids to 132 CTAs, without changing the protocol,
reduced the directional BF16 latency from 8.257 to 3.698 ms and FP8 from 8.217 to
3.145 ms. The GIN send grid remained eight CTAs. This identified local
underparallelism as the main cause of the initial large-batch regression.

The table compares that later tuning experiment with earlier baseline
references. It is not an equal-resource, randomized protocol-only ablation.
The baseline's local epilogues already used 132 CTAs; matching a `num_sms`
setting did not give both implementations identical local execution geometry.
A final baseline check at 2048 tokens differed by less than one percent from the
earlier baseline reference. Small-batch drift was larger, so exact small-batch
percentages do not identify the causal benefit of eliminating bookkeeping.

## Memory

Separate sums across all eight ranks:

| Payload | Registered transport windows, baseline → directional (MiB) | Graph allocator peaks at 2048 tokens, baseline → directional (MiB) | Eager allocator peaks at 2048 tokens, baseline → directional (MiB) |
|---|---:|---:|---:|
| BF16 | 3201.6 → 1030.0 | 9656 → 2419 | 1818 → 2419 |
| FP8 | 3201.6 → 906.0 | 7627 → 1892 | 1681 → 1892 |

Registered capacity fell 67.8% in BF16 and 71.7% in FP8. The graph-case allocator
peaks fell about 75%, while eager-case allocator peaks increased by 33.0% and
12.5%. Conservative receive allocation remains an optimization opportunity.

Registered transport windows and PyTorch allocator peaks are different metrics;
neither is total GPU VRAM. Per-rank allocator peaks need not occur
simultaneously. Graph pools, retained output caches, padded expert outputs, and
other allocations affect the comparison. Do not add these columns to claim a
total memory saving.

## What Nsight established

Paired Nsight Systems captures covered all eight ranks and measured graph
replays for BF16 and FP8. The directional captures contained the new protocol
kernels rather than DeepEP dispatch/combine kernels.

At 2048 tokens, raising the directional local grid from 24 to 132 CTAs changed
per-worker median stage durations approximately as follows:

| Stage | BF16, 24 → 132 CTAs (ms) | FP8, 24 → 132 CTAs (ms) |
|---|---:|---:|
| Expert-row packing | 1.33–1.36 → 0.30–0.38 | 1.86–1.91 → 0.37–0.40 |
| Local partial reduction | 3.67–3.69 → 0.68 | 3.66–3.69 → 0.68 |
| Reply reduction | 0.72–0.84 → 0.14–0.16 | 0.73–0.84 → 0.14–0.16 |

GIN reply stages still occupied about 1.1 ms. Stages include their associated
completion semantics and may include waiting; these are not isolated wire
times. Timelines across different ranks overlap, so stage durations must not be
summed into an end-to-end estimate.

In the masked DeepEP baseline, Attention combine and FFN dispatch had long
resident durations. Those fused kernels include payload movement, coordination,
and peer waits. Systems traces alone did not isolate the latency contribution
of generic expert-parallel bookkeeping. Removing standalone epilogues alone
also did not establish a consistent latency speedup in the earlier partial
specialization experiment.

One tuned FP8 capture had host/GPU timestamp-alignment excursions. Coverage was
validated through exact graph-launch correlation and GPU-internal durations;
cross-clock offsets and tails were not interpreted. No cross-node timestamp
subtraction was used without alignment.

## Inherited correctness evidence

The prototype passed BF16 and FP8 eager and graph checks across all eight GPUs,
including real expert execution, empty and skewed routing, changing source/token
signatures, interleaved buckets, and delayed-peer buffer reuse. Device epochs and
credits were exercised without per-exchange CPU barriers in the replay-safety
scenario. A second correctness pass covered the tuned 132-CTA configuration.

Standalone AFDEP changes build ownership, import paths, packaging, and bootstrap
integration. Its own validation must therefore cover installation/import,
native extension compilation, placement rejection, and the same GPU safety scenarios.
Passing inherited tests is not a substitute for running the extracted library.

## Optimization priorities

1. Vectorize partial/reply reductions and improve data reuse so comparable
   performance needs fewer SMs, leaving resources for expert GEMMs.
2. Load a token activation once before expanding it into expert rows; improve
   FP8 scale packing and its strided accesses.
3. Pipeline chunks of request transfer, packing, computation, and reply, with
   explicit independent credits for every reusable chunk.
4. Reduce conservative eager receive allocation when host counts are available.
5. Measure transport phases and NIC utilization before claiming bandwidth
   saturation or assigning costs to notifications and barriers.

Serving integration, expert-GEMM overlap, multiple lanes, and broader N/M
topologies remain unmeasured. Full-device local grids can contend with expert
compute; communication-only wins do not guarantee higher serving throughput.
