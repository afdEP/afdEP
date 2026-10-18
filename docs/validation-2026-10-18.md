# Standalone Hopper validation — October 18, 2026

**Status: passed.** The installed standalone wheel compiled and loaded on both
H100 hosts. BF16 and FP8 both passed eight-rank eager/CUDA-graph correctness and
communication-only smoke benchmarks. The sanitized
[machine-readable record](validation-2026-10-18.json) identifies the tested
package files, configuration, and completed runs. These are initial validation
and smoke measurements, not a new comparison against DeepEP.

## Artifact identity

| Item | Identifier |
|---|---|
| Archived runtime/harness checkpoint | `bb299d6c5fa9a2b351c824a7bc31bf2fe81aa571` |
| Tested wheel SHA-256 | `22baf3059be9b53654ef674a0e2bcb3df75f0fabf81caf7b1c8ba30b7682b15f` |
| Native source digest | `6c4f3a03773af311b86e0ee40eaf6d25e059df41abf4743183113009ce1e0dd2` |

The wheel installs the Python adapter and native sources. PyTorch's extension
tooling compiles the native library from that installed package. These runs do
not import `afd_plugin`, `deep_ep`, or `vllm`, and do not use the prototype's
DeepEP JIT. A pre-validation native error-check macro shadowing issue was fixed
in `bb299d6`; the artifact above includes that fix.

The subsequent checkpoint `a2d309684a6b3401515b0114b63f605c2d49e1c4` has the
same runtime and harness, with documentation changes only.

## Configuration

The setup uses eight H100 GPUs in two separate four-GPU NVLink domains: FFN
group ranks 0–3 followed by Attention group ranks 4–7. There is one in-flight
lane. Hidden size is 2048, with 128 real experts, top-k eight, expert alignment
128, eight GIN contexts, eight transport stripes, and 132-CTA local grids.
BF16 dispatch and prequantized FP8 E4M3FN block-128 dispatch both use BF16
replies. The caller applies routing weights before reply reduction.

The installed environment used PyTorch 2.13.0+cu130, CUDA Toolkit 13.0, and
matching NCCL header/runtime version 2.30.7 (`23007`). The selected backend
reported `GIN_IB_GDAKI` (type 3), with RDMA traffic class 3. Each GPU was an
H100 80GB with 132 SMs. Correctness reserved capacity 128 tokens per Attention
rank. Smoke benchmarks reserve capacity 2048 and cover 1/128/512/2048-token
buckets, with five warmups and 20 measured exchanges per bucket. These are
different capacities and run lengths from the October 15 comparison.

## Correctness coverage

| Check | BF16 | FP8 |
|---|---|---|
| Standalone installed-wheel extension build/load on both hosts | Passed | Shared build |
| Eight-rank eager correctness | Passed | Passed |
| CUDA graph capture and changing-input replay | Passed | Passed |
| Delayed-peer reuse and interleaved captured buckets | Passed | Passed |
| Misaligned contiguous source/output rejection | Passed | Passed |

The suite checks source/token/epoch signatures, routing receipts and weights,
valid expert-row boundaries, output reduction, empty and uneven traffic, hot
expert ownership, all sources at capacity, and nonempty requests with all
routing slots disabled. FP8 additionally exercises strided source scales.
Replay tests vary routing and payload values in fixed captured storage without
per-epoch CPU barriers. A delayed FFN participant exercises request/partial/
reply storage reuse. For each payload, all eight eager cases run twice, followed
by 18 changing-input graph replays. FFN rank 1 is delayed by 20 ms before replay
epoch 8, which revisits the nonempty uniform case. Exact power-of-two inputs
keep the independent numerical oracle representable in BF16. All eight ranks
reported success in both payload runs.

## Smoke timing

All four payload/mode runs passed, with complete samples from all eight ranks.
Communication-only host-wall round trips, in milliseconds:

| Tokens per Attention rank | BF16 eager | BF16 graph | FP8 eager | FP8 graph |
|---|---:|---:|---:|---:|
| 1 | 0.342 | 0.359 | 0.430 | 0.365 |
| 128 | 0.498 | 0.472 | 0.550 | 0.480 |
| 512 | 1.068 | 0.989 | 0.943 | 0.876 |
| 2048 | 3.763 | 3.591 | 2.993 | 2.874 |

The benchmark uses persistent BF16 reply rows and no model GEMMs or quantization
in the timed exchange. The metric is the median iteration maximum across Attention
ranks, including GPU completion and launch/readiness skew. The pre-iteration
Gloo barrier, input quantization, and output checks are excluded. Each
payload/mode has one run; these are smoke measurements without repeated
comparison trials.

These fresh smoke runs do not remeasure Ronen's masked DeepEP baseline. Their
capacity is selected from the benchmark's maximum token bucket; the inherited
October 15 comparison reserved 8192 tokens per Attention rank. Differences in
capacity, run length, configuration, and routing fixture prevent treating the
smoke timings as a new speedup comparison. The fresh fixture uses deterministic
routes with periodically skipped selections, rather than the earlier random
routing workload. Fresh Nsight profiling and end-to-end serving overlap
are not established by this validation.

## Remaining scope

This validation concerns the first Hopper four-FFN/four-Attention backend.
It does not establish arbitrary N/M layouts, multiple in-flight lanes,
Blackwell support, expert-GEMM overlap, model accuracy, or serving throughput.
The inherited [October 15 performance analysis](performance.md) remains separate
from the standalone artifact validation on this page.

The two temporary validation workloads were removed after confirming no GPU
compute applications remained; all eight requested GPUs were released. Cleanup
was verified at 08:32 UTC on October 18, 2026.
