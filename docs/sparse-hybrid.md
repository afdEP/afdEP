# Sparse directional hybrid

From 1.1.0, when every NVLink domain holds one role, afdEP-comm's directional protocol
signals only the channels that carry tokens. It is part of the protocol, not an option:
there is nothing to set, and it runs wherever the directional hybrid runs. This page
shows what it adds; the design and the correctness argument are in
[design.md §17](design.md#17-sparse-directional-hybrid-default-across-nvlink-domains).

## The cost it removes

Across NVLink domains, DeepEP's hybrid transport splits every exchange into C channels
(192 with 24 SMs), and each channel completes with a signal whether it carried tokens
or not. In AFD the traffic flows one way, so these signals converge on the few FFN NICs:

- in dispatch, each of S Attention domains sends a finish tail on all C channels, S·C
  RDMA atomics into each FFN NIC (2,112 at 11:1);
- in combine, the FFN signals every Attention domain on every channel, another S·C out of
  the same NIC;
- token i of every source rides channel i mod C, so at small batches every source's
  first tokens queue on the same FFN forward warp.

At decode batch sizes this synchronization, not the data, sets the round trip: with the
dense protocol of 1.0.0, one token per Attention GPU at 11:1 takes 231 µs, and 128 tokens
take only twice that.

## What it does

- **Stagger.** Source s starts its tokens at channel s·C/S, so the sources' first tokens
  land on different FFN forward warps, SMs and queue pairs. A channel still carries at
  most ceil(T/C) tokens of a source.
- **Sparse dispatch.** Each source first sends its token count, one word per FFN rail
  peer. Channels that carried nothing send no finish tail, and FFN forwarders retire them
  from the count. An FFN NIC takes S·min(T, C) finish atomics plus S count words instead
  of S·C.
- **Sparse combine.** The FFN signals a source only on the channels where it replied to
  that source, plus channel 0 to every source. Channel 0 is a done witness, sent only
  after every forward warp has finished reading the exchange, so no source can start the
  next exchange and overwrite a buffer still in use, even a source that sent nothing.

The three parts change the wire protocol together: construction checks that every rank
resolved them alike and rejects a mismatch instead of hanging.

## Where it runs

| Topology | Protocol |
|---|---|
| Several NVLink domains, each holding one role (directional hybrid) | Sparse, always |
| One NVLink domain | Unchanged: the sparse kernels and launch branches require two or more NVLink domains |
| FFN GPUs sharing a domain with Attention GPUs | Union hybrid, or the opt-in mixed-role protocol ([§18](design.md#18-mixed-role-directional-hybrid-opt-in-ffn-ranks-sharing-an-nvlink-domain)); sparse does not apply yet |
| The baseline, FastAFD's DeepEP M2N | Not affected: it runs DeepEP V2 unspecialized |

## What it adds

### Round trip

Communication only: one dispatch plus combine round trip under CUDA graphs, maximum over
Attention ranks, 2 fresh processes per arm, paired against the dense protocol of 1.0.0
(native tree `958652a3`, which differs from the 1.0.0 release tree `131efc37` only in
comments). Top-8 of 128 experts, capacity 2,048, 24 SMs, H200 over InfiniBand (NCCL GIN).

| Layout, payload, tree | T=1 | 8 | 32 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|
| 11:1, 12 single-role domains (12 × 2 H200), FP8, hidden 2048, `200944ce` | 231.5 → 73.8 µs, **−68.1%** | 259.5 → 86.2, **−66.8%** | 285.0 → 140.5, **−50.7%** | 461.0 → 428.1, −7.1% | 712.4 → 691.2, −3.0% | 1,220.1 → 1,221.1, +0.1% |
| Two FFN domains, 4F:20A (12 × 2 H200), FP8, hidden 2048, `200944ce` | 207.9 → 68.5, **−67.1%** | 221.0 → 79.0, **−64.3%** | 240.0 → 127.2, **−47.0%** | 403.5 → 368.8, −8.6% | 615.1 → 604.6, −1.7% | – |
| Three Attention domains, 2F:6A (4 × 2 H200), FP8, hidden 2048, `200944ce` | 99.7 → 65.9, **−33.9%** | 115.7 → 74.4, **−35.7%** | 121.2 → 85.5, **−29.5%** | 166.8 → 156.8, −6.0% | 244.4 → 235.6, −3.6% | – |
| 5:1, 4F:20A (6 × 4 H200), FP8, hidden 2048, `7b8ed8b5` (done witness) | 138.0 → 70.7, **−48.8%** | 155.5 → 82.3, **−47.1%** | 164.6 → 97.7, **−40.7%** | 235.2 → 226.8, −3.6% | – | 604.3 → 603.8, −0.1% |
| 5:1, 4F:20A (6 × 4 H200), BF16, hidden 4096, `34fa5eac` (done witness) | 163.5 → 79.5, **−51.4%** | 208.5 → 97.6, **−53.2%** | 234.7 → 141.4, **−39.8%** | 410.1 → 400.3, −2.4% | 696.7 → 692.4, −0.6% | – |
| 5:1, 4F:20A (6 × 4 H200), BF16, hidden 4096, `21298d5b` (1.1.0, no switch set) against `8e63d39f` | 163.1 → 79.1, **−51.5%** | 206.8 → 97.3, **−53.0%** | 231.8 → 136.6, **−41.1%** | 408.2 → 396.6, −2.8% | 694.1 → 689.5, −0.7% | 1,278.3 → 1,289.6, +0.9% |
| 5:1, 4F:20A (6 × 4 H200), FP8, hidden 2048, `21298d5b` (1.1.0, no switch set) against `8e63d39f` | 138.8 → 70.3, **−49.3%** | 155.0 → 83.1, **−46.4%** | 164.6 → 97.9, **−40.6%** | 236.9 → 224.9, −5.1% | – | 607.5 → 605.5, −0.3% |

The gain grows with the number of Attention domains, because each one adds C signals to
the FFN NICs: −34% at one token with three, −49% with five and −68% with eleven. It holds
with two FFN domains, and it fades once the data dominates, from about 128 tokens per
Attention GPU. Against FastAFD's DeepEP M2N, one token at 11:1 takes 73.8 µs instead of
339.4 µs (−78%, against an earlier unpaired baseline run), and at 5:1 with hidden 4096,
79.1 µs instead of 205.5 µs on the 1.1.0 tree (−62%, paired in the same screen).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/results-xn-sparse-dark.svg">
  <img alt="Sparse directional hybrid against the dense 1.0.0 protocol and FastAFD's DeepEP M2N at 11:1 over 12 domains" src="assets/results-xn-sparse.svg">
</picture>

The parts compound. At 11:1 and one token, the stagger alone gains 12% and sparse
dispatch alone 34%; together 43%, and sparse combine, worth 2% alone, takes the
three to 68%. Once the stagger spreads the replies over many channels, the outbound
combine signals become the critical path, and sparse combine removes them (our reading of
the timings, not yet traced). Every part's screen is in the
[numbers below](#numbers).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/results-xn-scaling-dark.svg">
  <img alt="Sparse directional hybrid at other layouts" src="assets/results-xn-scaling.svg">
</picture>

### Serving

Qwen3-235B-A22B-FP8 under vLLM with the AFD plugin, 20 Attention GPUs and 4 FFN GPUs on
six 4 × H200 pods joined by InfiniBand, 4,096 input and 2,048 output tokens, one wave per
load (one run each, no repeats). The dense arm ran 1.0.0, the sparse arm the cross-node
tree `34fa5eac` with all three parts:

| Requests per Attention GPU | 4 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|
| Median TPOT, dense, ms | 43.5 | 51.9 | 58.9 | 72.8 |
| Median TPOT, sparse, ms | 32.5 | 39.8 | 47.6 | 65.1 |
| TPOT change | **−25.4%** | **−23.3%** | **−19.1%** | **−10.5%** |
| Decode tokens/s change | +34.0% | +30.5% | +23.6% | +11.6% |

### What it costs

- **The done witness.** Waiting for every forward warp before channel 0 signals costs
  5 µs (+6.6%) at 8 tokens at 5:1 against the earlier, unsafe witness, and is within
  ±1.2% elsewhere; the round trip there is still 47% below the dense protocol.
- **Large batches gain little.** From 128 tokens per Attention GPU the gain falls to 2–9%,
  and at 256 and 512 tokens to 0–4%. One cell measured slower than dense: +0.9% at 512
  tokens with BF16 and hidden 4096 on the 1.1.0 tree, in both repeats; with FP8 at
  hidden 2048 the same tree measured −0.3% at 512.
- **Per exchange:** one count word per source to each FFN rail peer, and the FFN dispatch
  waits for and clears every count at exit.
- **Code:** four generated kernel copies (`afdep_source_hybrid_dispatch.cuh`,
  `afdep_ffn_sparse_hybrid_dispatch.cuh`, `afdep_ffn_sparse_hybrid_combine.cuh`,
  `afdep_sparse_hybrid_collect.cuh`) and one header of protocol words. A protocol bug
  would show as a hang or wrong values rather than a slowdown, which is why the
  correctness tests below include faults and a negative control.

## Validation

The GPU suite runs 14 to 16 eager cases, 28 graph replays and 728 checked graph exchanges
per run, including empty, inactive and all-skipped sources, skewed routes, delayed
replies and dispatches, changing graph inputs and bucket cycles. It passed with the
sparse protocol on:

- `0c22f278` and `200944ce`, 24 ranks over 12 × 2 H200: FP8 at capacities 8 and 64, with
  one and two FFN domains;
- `2d22e5e9`, 6 × 4 H200 (4 FFN + 20 Attention ranks), FP8, capacity 8;
- `7b8ed8b5`, the first tree with the done witness, 6 × 4 H200, FP8, capacity 2,048, with
  and without forward warps delayed by about 200 µs;
- `34fa5eac`, the shipped cross-node tree, 6 × 4 H200: FP8 and BF16, 4 FFN + 20 Attention
  and 8 FFN + 16 Attention ranks, capacities 8 and 64, every run passing on all six pods;
- `21298d5b`, the 1.1.0 tree with no switch set, 6 × 4 H200: FP8 and BF16, 4 FFN + 20
  Attention and 8 FFN + 16 Attention ranks at capacities 8 and 64, and 4 + 20 at capacity
  2,048 with 24 SMs, every run passing on all six pods (832 checked graph exchanges per
  rank at capacity 8). Both ablation controls set on every rank pass too:
  `AFDEP_SHARED_HYBRID_FORWARD=1` (stagger and sparse combine without sparse dispatch) and
  `AFDEP_SHARED_HYBRID_COLLECT=1` (stagger and sparse dispatch without sparse combine).

As a negative control, a collect that waits only for channel 0 (`f92c479c`) fails with
value mismatches. A shape control set on one role only (for example an FFN-only
`AFDEP_SHARED_HYBRID_FORWARD=1`) is rejected with the agreement error on every rank,
on the 1.1.0 tree as before (all six pods report it, none hangs).

Not suite-validated: 8-GPU NVLink domains (measured in section 19's transport screens, not
suite-run) and worlds larger than 24 ranks. Record:
[validation-sparse-hybrid.json](validation-sparse-hybrid.json).

## Comparing with the dense protocol

There is no switch to turn it off. To measure against the dense protocol, run 1.0.0 as
its own arm, as the screens above did: `benchmarks/screen.py` takes a `_tree` per arm and
checks each arm's native digest. The `AFDEP_SHARED_HYBRID_*` and
`AFDEP_SHARED_ROLE_KERNELS` ablation controls also turn parts of it off, but they change
the kernel shapes as well, so they are not a clean comparison. Set them on every rank,
or construction rejects the mismatch.

## Why it is an AFD change

afdEP-comm ships only changes that depend on the AFD roles. This one exists only in the
single-role directional protocol, and its correctness depends on each phase flowing one
way: Attention domains receive only combine signals, and FFN domains receive only
dispatch traffic. The S:1 ratio of Attention to FFN domains is what concentrates the
signals on the FFN NICs. Signalling only used channels might also help DeepEP's own union
hybrid at small batches; that is neither implemented nor measured here.

## Numbers

H200 joined by InfiniBand (NCCL GIN), FP8, hidden 2048, top-8 of 128 experts, capacity 2,048, 24 SMs; 2 fresh processes per arm, max over Attention ranks; the dense directional protocol on `958652a3` (1.0.0), the sparse parts on the tree each table names. Each part was measured with the opt-in switches it had before 1.1.0. Records: [validation-sparse-hybrid.json](validation-sparse-hybrid.json) and, for the plotted series, [benchmarks.json](assets/benchmarks.json).

| All three parts (200944ce, before the done witness), 11:1 over 12 domains | T=1 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense directional (1.0.0), µs | 231.5 | 259.5 | 266.3 | 285.0 | 320.9 | 461.0 | 712.4 | 1,220.1 |
| Sparse directional, µs | 73.8 | 86.2 | 100.4 | 140.5 | 235.1 | 428.1 | 691.2 | 1,221.1 |
| Sparse directional vs dense | −68.1% | −66.8% | −62.3% | −50.7% | −26.7% | −7.1% | −3.0% | +0.1% |

| Stagger and sparse dispatch, 11:1 | T=1 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|
| Dense directional (1.0.0), µs | 231.3 | 259.9 | 266.8 | 285.2 | 321.4 | 460.7 |
| Stagger only, µs | 202.7 | 215.0 | 225.9 | 260.8 | 323.5 | 470.8 |
| Sparse dispatch only, µs | 153.1 | 183.8 | 194.7 | 221.5 | 277.6 | 441.0 |
| Stagger + sparse dispatch, µs | 131.0 | 144.3 | 154.2 | 197.9 | 280.8 | 452.3 |
| Stagger only vs dense | −12.4% | −17.3% | −15.3% | −8.5% | +0.7% | +2.2% |
| Sparse dispatch only vs dense | −33.8% | −29.3% | −27.0% | −22.3% | −13.6% | −4.3% |
| Stagger + sparse dispatch vs dense | −43.4% | −44.5% | −42.2% | −30.6% | −12.6% | −1.8% |

| Sparse combine, 11:1 | T=1 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|
| Dense directional (1.0.0), µs | 231.7 | 260.0 | 266.2 | 285.1 | 321.1 | 460.9 |
| Sparse combine only, µs | 226.0 | 255.7 | 264.8 | 284.5 | 320.4 | 460.3 |
| Sparse dispatch + combine, µs | 151.5 | 180.8 | 194.0 | 221.1 | 276.5 | 440.9 |
| Sparse directional, µs | 74.3 | 86.3 | 100.4 | 140.8 | 235.5 | 428.0 |
| Sparse combine only vs dense | −2.4% | −1.6% | −0.5% | −0.2% | −0.2% | −0.1% |
| Sparse dispatch + combine vs dense | −34.6% | −30.5% | −27.1% | −22.4% | −13.9% | −4.3% |
| Sparse directional vs dense | −67.9% | −66.8% | −62.3% | −50.6% | −26.6% | −7.1% |

| Stagger at every size against a bound of 48 tokens (a former option), 11:1 | T=1 | 8 | 16 | 32 | 48 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense directional (1.0.0), µs | 231.6 | 260.2 | 266.3 | 285.0 | 300.1 | 321.5 | 460.7 |
| Sparse dispatch + combine, µs | 148.3 | 179.8 | 193.4 | 221.2 | 247.7 | 277.3 | 441.0 |
| + stagger up to 48 tokens, µs | 74.2 | 86.2 | 100.2 | 140.6 | 188.8 | 277.5 | 440.8 |
| + stagger at every size, µs | 74.0 | 86.3 | 100.4 | 140.6 | 188.5 | 236.1 | 428.6 |
| Sparse dispatch + combine vs dense | −36.0% | −30.9% | −27.4% | −22.4% | −17.5% | −13.7% | −4.3% |
| + stagger up to 48 tokens vs dense | −68.0% | −66.9% | −62.4% | −50.7% | −37.1% | −13.7% | −4.3% |
| + stagger at every size vs dense | −68.0% | −66.8% | −62.3% | −50.7% | −37.2% | −26.6% | −7.0% |

| Two FFN domains, 4F:20A (12 × 2 H200) | T=1 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense directional (1.0.0), µs | 207.9 | 221.0 | 224.4 | 240.0 | 288.7 | 403.5 | 615.1 |
| Sparse directional, µs | 68.5 | 79.0 | 90.9 | 127.2 | 209.0 | 368.8 | 604.6 |
| Sparse directional vs dense | −67.1% | −64.3% | −59.5% | −47.0% | −27.6% | −8.6% | −1.7% |

| Three Attention domains, 2F:6A (4 × 2 H200) | T=1 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense directional (1.0.0), µs | 99.7 | 115.7 | 118.8 | 121.2 | 132.1 | 166.8 | 244.4 |
| Sparse directional, µs | 65.9 | 74.4 | 77.0 | 85.5 | 103.2 | 156.8 | 235.6 |
| Sparse directional vs dense | −33.9% | −35.7% | −35.2% | −29.5% | −21.9% | −6.0% | −3.6% |

| 5:1, 6 single-role domains (6 × 4 H200) | T=1 | 8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|---:|
| Dense directional (1.0.0), µs | 137.8 | 155.3 | 165.0 | 235.1 | 604.8 |
| Sparse directional, µs | 70.5 | 76.8 | 97.7 | 229.4 | 604.8 |
| FastAFD's DeepEP M2N, µs | 185.4 | 193.6 | 210.7 | 277.9 | 636.0 |
| Sparse directional vs dense | −48.8% | −50.5% | −40.8% | −2.4% | −0.0% |
| FastAFD's DeepEP M2N vs dense | +34.5% | +24.7% | +27.7% | +18.2% | +5.2% |

| Done witness cost, 5:1 (6 × 4 H200) | T=1 | 8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|---:|
| Dense directional (1.0.0), µs | 138.0 | 155.5 | 164.6 | 235.2 | 604.3 |
| Sparse, earlier witness on channel 0 only (unsafe), µs | 70.8 | 77.3 | 97.6 | 229.5 | 605.0 |
| Sparse, done witness after every forward warp, µs | 70.7 | 82.3 | 97.7 | 226.8 | 603.8 |
| Sparse, earlier witness on channel 0 only (unsafe) vs dense | −48.7% | −50.3% | −40.7% | −2.4% | +0.1% |
| Sparse, done witness after every forward warp vs dense | −48.8% | −47.1% | −40.7% | −3.6% | −0.1% |

In one NVLink domain, tree `200944ce` matches 1.0.0 (`958652a3`, labelled dense below) with the sparse parts off and on; the later
done witness runs only with two or more NVLink domains. one NVLink domain of 8 × H200 (one pod), BF16, hidden 2048, top-8 of 128 experts, capacity 2,048, 24 SMs; 2 fresh processes per arm, max over Attention ranks.

| 6A:2F | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP-comm 1.0.0, µs | 24.3 | 29.3 | 35.5 | 53.2 | 152.0 | 502.0 |
| Tree 200944ce, sparse parts off, µs | 24.2 | 29.3 | 35.3 | 53.2 | 151.9 | 501.6 |
| Tree 200944ce, sparse parts on, µs | 24.6 | 28.9 | 35.5 | 53.2 | 152.2 | 501.8 |
| Tree 200944ce, sparse parts off vs 1.0.0 | −0.6% | +0.2% | −0.6% | +0.0% | −0.0% | −0.1% |
| Tree 200944ce, sparse parts on vs 1.0.0 | +1.0% | −1.2% | +0.2% | +0.1% | +0.2% | −0.0% |

| 7A:1F | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP-comm 1.0.0, µs | 23.8 | 27.2 | 34.1 | 70.3 | 202.0 | 736.8 |
| Tree 200944ce, sparse parts off, µs | 23.9 | 27.0 | 34.1 | 69.9 | 201.6 | 736.5 |
| Tree 200944ce, sparse parts on, µs | 23.9 | 27.1 | 33.8 | 69.5 | 201.9 | 736.7 |
| Tree 200944ce, sparse parts off vs 1.0.0 | +0.3% | −0.8% | −0.0% | −0.6% | −0.2% | −0.0% |
| Tree 200944ce, sparse parts on vs 1.0.0 | +0.5% | −0.4% | −1.2% | −1.0% | −0.0% | −0.0% |

| 4A:4F | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP-comm 1.0.0, µs | 24.3 | 26.6 | 27.8 | 39.6 | 96.0 | 296.3 |
| Tree 200944ce, sparse parts off, µs | 24.3 | 26.5 | 27.8 | 39.9 | 96.0 | 296.5 |
| Tree 200944ce, sparse parts on, µs | 24.2 | 26.5 | 27.8 | 39.9 | 96.1 | 296.9 |
| Tree 200944ce, sparse parts off vs 1.0.0 | +0.0% | −0.2% | +0.3% | +0.8% | +0.0% | +0.1% |
| Tree 200944ce, sparse parts on vs 1.0.0 | −0.6% | −0.1% | +0.3% | +0.9% | +0.2% | +0.2% |
