# afdEP-comm: DeepEP V2 specialized for AFD roles

Baseline: **FastAFD's DeepEP M2N**, the way [FastAFD](https://github.com/hao-ai-lab/FastAFD)
runs AFD over DeepEP V2 ([`DeepEPM2NAdapter`](https://github.com/hao-ai-lab/FastAFD/blob/3c7161949310b6d59d6b4cf9bf997a4935c8113b/python/minisgl/moe/deepep_m2n_adapter.py)): one
symmetric expert-parallel group, dummy expert slots on Attention ranks, every rank
sending, receiving and waiting in both phases, upstream combine. The benchmarks run it
over the same vendored kernels ([benchmarks/fastafd.py](../benchmarks/fastafd.py)); tables
below call it FastAFD M2N. afdEP-comm's changes apply in one NVLink domain and across
NVLink domains joined by GIN, and the baseline's unspecialized instantiations keep the
upstream template arguments and kernels. Records written before 1.1.0 call the baseline
"Masked" or "Masked DeepEP" and its arm `role_aware=False`, and afdEP-comm "the role
path"; 1.1.0 removed that flag.

Digests: native source digests in the records are those of the trees as measured.
`958652a3` is the tree of commit `32c871d`, which merged the directional protocol; the
commits after it changed only comments, notices and other text in the native tree
(the 1.0.0 release tree is `131efc37`). Older digests name the trees each section
measured; the 4 × H100 result below, for example, is `bd4db4da`, which predates
sections 10 and 13–16. The earlier DeepEP-reuse role specialization, v0, has its own
record ([validation](validation.md)). Sections 17 and 18 were measured on development
trees; their published commits differ only in comments and text: `200944ce`,
`2d22e5e9`, `7b8ed8b5` and `81b71eb0` are published as `adfbaf65`, `b3821109`,
`ede73b32` and `c065385b`, and the comment fixes after review give `34fa5eac`.
The rename tree, `8e63d39f`, differs from section 19's `b5078d62` only in license and
provenance files (`LICENSE.FastAFD` and the two hash records). The 1.1.0 tree, `21298d5b`,
differs from it only in section 17's host-side switch functions, which no longer read
the environment, and in comments and the hash records; no kernel changed.

afdEP-comm ships a change only if it depends on the AFD roles. A change that would
speed up FastAFD's DeepEP M2N as well is a DeepEP improvement, not an AFD one, and is
rejected here even when it is faster (R1–R3). Every shipped change keeps a
measurement control switch, except section 17 from 1.1.0: the sparse directional hybrid
is the directional protocol itself, measured against the dense protocol of 1.0.0 as its
own tree ([value and evidence](sparse-hybrid.md)).

## Result

Communication-only, 4 × H100 in one NVLink domain, E128/top-k 8 (E96 for
1A:3F), capacity 2,048, BF16 unless noted, CUDA graphs, 24 transport SMs;
median of three fresh-process repeats, max over Attention ranks. Each cell: afdEP-comm µs (FastAFD M2N µs, geometric-mean paired
change). Every cell was faster in all three repeats.

| Layout / hidden / routing | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| 2A2F, 2,048, random | 20.7 (31.7, −34.6%) | 24.8 (34.4, −29.1%) | 26.3 (36.0, −26.5%) | 35.3 (44.3, −20.2%) | 70.6 (78.7, −10.4%) | 192.7 (201.0, −4.1%) |
| 2A2F, 4,096, random | 28.4 (39.1, −27.2%) | 39.2 (47.9, −18.9%) | 46.3 (55.7, −18.4%) | 61.5 (68.7, −10.3%) | 127.3 (136.2, −6.4%) | 383.2 (392.6, −2.4%) |
| 2A2F, 2,048, skew | 24.1 (34.3, −29.5%) | 25.7 (35.7, −27.6%) | 29.5 (38.9, −24.1%) | 41.2 (50.1, −17.5%) | 88.7 (95.9, −7.6%) | 269.9 (274.1, −1.5%) |
| 3A:1F, 2,048, random | 24.8 (36.4, −31.4%) | 27.4 (38.6, −29.4%) | 33.0 (44.8, −26.5%) | 45.9 (55.8, −17.5%) | 109.4 (119.5, −8.4%) | 367.8 (372.1, −1.1%) |
| 1A:3F, 2,048, random | 19.9 (30.5, −34.2%) | 21.5 (31.3, −31.3%) | 23.1 (32.8, −29.3%) | 29.4 (39.7, −24.9%) | 59.6 (67.7, −12.2%) | 188.3 (197.7, −4.7%) |
| 2A2F FP8, 2,048, random | 21.3 (32.3, −34.1%) | 25.9 (35.6, −27.8%) | 26.8 (36.7, −26.8%) | 34.5 (43.5, −20.6%) | 62.5 (71.8, −12.9%) | 159.1 (167.8, −5.2%) |

On the build before the per-CTA release, FP8 at H4096 measured −21.4% to −2.5%
against FastAFD M2N. Eager exchanges up to 128 tokens are bound by host launches (55–62 µs per round trip in both arms) and too noisy to rank; a
profiled trace shows afdEP-comm issuing fewer host API calls than FastAFD M2N. From
512 tokens eager runs are faster in every repeat (−3.6% and −2.0% at 2,048)
([record](validation-fp8-eager.json)).

The gain is the protocol overhead AFD removes (barriers, union counts, all-rank
exits), so it is largest at decode-sized token counts and shrinks as payload
movement dominates. Registered transport memory is 50% of union in 2A2F and
about 75% at 3:1 and 1:3 (section 4). The GPU matrix and both ratios pass on the
same native digest ([matrix](validation-cta-exit.json),
[screens](validation-result.json); the build before the per-CTA release:
[validation-clean.json](validation-clean.json)). These
are communication-only measurements, not serving claims.

**The baseline arm is the code FastAFD and the plugin ran.** It runs this tree's DeepEP
V2 kernels unspecialized (benchmarks/fastafd.py). Timed against the AFD plugin's
`deepep_m2n` buffer (FastAFD's `3c71619` DeepEP V2 snapshot with the plugin's three
build patches, which has no role specialization) through the same exchange, routes and
timing on 4 × H200, it
matches within −0.4% to +1.2% at 2A2F and −0.2% to +0.3% at 3A:1F from 1 to 2,048
tokens; against that reference afdEP-comm is 29.8% (2A2F) and 37.3% (3A:1F) faster
at one token and 3.6% / 4.6% at 2,048 ([record](validation-fastafd-equivalence.json)).

The role suites (sections 14 and 15) also give compute back the SMs that the
shared launches held. Under a synthetic GEMM load on a side stream, an FFN GEMM in
one NVLink domain takes 5–28% longer than alone, against 18–87% with the shared kernel
and 20–48% under FastAFD M2N (1–512 tokens, 2A2F and 3A:1F). In directional hybrid
(8F:8A) an Attention GEMM takes 5–7% longer than alone, against 31–44% with the
shared shapes or FastAFD M2N; the exchanges get faster too.

## Shipped (AFD-specific)

Contribution of each change on the clean build (before the per-CTA release of
section 3, which `AFDEP_GRID_SYNC_EXIT=1` measures separately): each arm switches one change
off with its control variable; cells are the paired change against the full
afdEP-comm (positive = slower), and every control at up to 128 tokens was slower
in all three repeats ([record](validation-ablation.json)).

| Arm (H2048 / H4096 random) | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| `AFDEP_KEEP_ENTRY_BARRIERS=1` | +32.5% / +22.5% | +26.5% / +18.2% | +23.2% / +15.8% | +18.0% / +11.2% | +9.4% / +5.9% | +4.0% / +2.0% |
| `AFDEP_UNION_NOTIFY=1` | +6.9% / +5.9% | +6.0% / +4.8% | +6.6% / +4.0% | +5.3% / +0.6% | +0.2% / +0.2% | +0.1% / +0.1% |
| `AFDEP_ALL_RANK_EXIT=1` | +4.3% / +3.4% | +4.5% / +6.7% | +4.4% / +8.3% | +2.3% / −0.8% | +0.1% / +0.4% | +0.3% / +0.1% |
| FastAFD M2N | +39.1% / +27.5% | +29.5% / +19.1% | +28.9% / +25.7% | +20.5% / +8.7% | +9.1% / +5.8% | +3.6% / +1.8% |

Section 4 (role-sized buffers) is a memory change and is latency-neutral
against its control.

### 1. Entry barriers

Upstream dispatch and combine each open with an all-rank NVLink barrier plus a
grid sync, so every call waits for the slowest peer to arrive before writing. On
afdEP-comm both are implied by order (proof beside `afdep_skip_entry_barriers`
in `csrc/kernels/elastic/afdep_roles.hpp`): each rank runs complete
dispatch→combine pairs on one ordered stream, the main kernels launch without
PDL after their epilogues, and every exchange-(N+1) remote write follows an exit
barrier that each peer reaches only after its exchange-N inbox reads. All
afdEP-comm's ranks skip both barriers; exit barriers, credits, counters and
lifetimes are unchanged. `AFDEP_KEEP_ENTRY_BARRIERS=1` restores them as a
measurement control.

AFD basis: the proof needs the role order (sources never receive dispatch
payloads, FFN ranks never receive combine payloads); a symmetric world, where
every rank writes every peer's inbox in both phases, keeps the barrier. On the clean
build the skip saves 7.3 µs per one-token round trip at H2048 (30.1 → 22.8 µs)
and 6.8 µs at H4096 (development screens:
[validation-barriers.json](validation-barriers.json)).

The GPU matrix passes with barriers skipped, including delayed FFN replies, a
delayed Attention source before dispatch, changing graph replays, host-count
reuse and invalid-route assertions.

### 2. Directional count exchange

Upstream dispatch reduces the union count vector (every rank and logical expert)
across all transport CTAs on every rank, sends it to every peer and waits for
every peer's vector, so Attention ranks wait for always-zero counts and FFN ranks
reduce their own zero counts. In afdEP-comm Attention sources reduce and send
only FFN rank/expert counts to FFN ranks and wait for none; FFN ranks skip their
own reduction and wait only for source counts. Prefix sums and host counts keep
their canonical positions (sources publish zeros). `AFDEP_UNION_NOTIFY=1` restores
the union exchange as a measurement control.

AFD basis: dispatch counts flow one way because sources receive no tokens and
FFN ranks send none; a symmetric rank both sends and receives.

Development screens: [validation-notify.json](validation-notify.json).

### 3. Directed exit

Upstream ends dispatch and combine with an all-rank barrier, so a sender also
waits for every receiver and every rank waits for unrelated peers. On the role
path, senders (Attention in dispatch, FFN in combine) flush their TMA stores,
grid-sync and release a per-peer monotonic arrival counter in each receiver's
workspace (the direct path's unused scale-out channel tails); receivers wait on
SM0 for every sender's counter to reach their own round. The section 1 ordering
argument carries over: exchange-(N+1) writes into FFN inboxes follow the
Attention rank's receipt of the FFN's combine-N arrival, which the FFN releases
after its copy epilogue and counter clears; writes into Attention inboxes follow
the FFN's receipt of the Attention rank's dispatch arrival, which follows its
reduce epilogue. `AFDEP_ALL_RANK_EXIT=1` restores the exit barriers as a
measurement control.

AFD basis: each phase has one sending role and one receiving role, so a sender
has nothing to wait for; in a symmetric world every rank is both.

Development screens and GPU matrix: [validation-exit.json](validation-exit.json).

Per-CTA release: the sending role receives nothing in its phase, so it does not
need the grid sync before releasing its arrival. Each sending CTA flushes its own
TMA stores and releases its own arrival; receivers wait for round × sender CTAs
(every rank uses the same transport SM count) and the last source CTA to finish
resets the slot counters. `AFDEP_GRID_SYNC_EXIT=1` keeps the grid-synced release
as a control; against it the control is slower in every cell and repeat (H2048
random: +9.7% at one token, 22.68 vs 20.74 µs, to +0.9% at 2,048; H4096 +7.1%
to +0.4%). With it the one-token round trip is 20.7 µs at H2048 (FastAFD M2N 31.6)
and 28.4 µs at H4096 (FastAFD M2N 38.6)
([record](validation-cta-exit.json)).

### 4. Role-sized registered buffers

AFD basis: in one NVLink domain only Attention sources write FFN dispatch inboxes
and only FFN ranks write Attention combine inboxes. Union sizing reserves a
dispatch region for every rank (FFN ranks never send) and, in the rank layout, a
combine region for every rank (Attention ranks never reply). afdEP-comm sizes
dispatch for the N sources (source s writes region s - F; the copy epilogue reads
with the same offset) and combine for the F FFN senders. Every rank registers the
same maximum, as the symmetric window requires. Without the rank layout
(more ranks than top-k) combine keeps its top-k regions.

| Layout (capacity 2,048, H2048, top-k 8, BF16) | Registered | vs union |
|---|---:|---:|
| Union (FastAFD M2N) | 34.6 MB | — |
| 2A2F | 17.3 MB | −50% |
| 3A:1F | 26.0 MB | −25% |
| 1A:3F | 25.6 MB | −26% |

Latency is unchanged against the `AFDEP_UNION_BUFFERS=1` control (H2048 random:
19.6 vs 19.6 µs at one token, 185.9 vs 185.9 µs at 2,048). The full 2A2F matrix
and both ratios pass ([record](validation-buffers.json)).

### 5. Attention:FFN ratios

afdEP-comm's kernels are written in terms of FFN and source rank counts. The
GPU suite takes `--ffn-ranks`; 3 Attention + 1 FFN and 1 Attention + 3 FFN pass
the full case set (BF16/FP8, small and large, baseline runs, delayed FFN and
Attention peers). Their latency is in the result table.

### 6. Two NVLink domains

The role API previously rejected more than one physical NVLink domain. Across
two waldorf nodes (2 × H200 each, InfiniBand GIN) afdEP-comm passes all 14
cross-node cases (BF16/FP8, hybrid on/off, small/large, baseline runs, and
3:1 / 1:3 layouts that mix roles in a domain), so the check only requires GIN
([record](validation-cross-node.json)). Sections 1–4 are single-domain.

Across three nodes (6 ranks) afdEP-comm passes all 14 cases with one FFN
domain and two Attention domains, with two FFN domains and one Attention domain,
and with mixed-role domains. With more than one FFN domain, DeepEP's hybrid
combine rounds each FFN domain's partial to BF16 before the cross-domain sum, so
the GPU suite's reference now rounds once per reduction stage (FFN rank, FFN
domain, total); FastAFD M2N showed the same one-ulp difference against the old
single-rounding reference ([record](validation-three-domain.json)).

Three-node latency, one FFN node and two Attention nodes (6 ranks, H2048 random,
three repeats, every cell faster than both controls in all repeats):

| Arm | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP-comm (directional) | 84.1 µs | 94.0 | 98.8 | 134.0 | 321.2 | 924.6 |
| `AFDEP_UNION_HYBRID=1` | +28.6% | +28.3% | +27.5% | +16.9% | +5.9% | +2.3% |
| FastAFD M2N | +26.6% (106.5) | +27.4% (119.1) | +26.7% (125.4) | +16.1% (155.4) | +5.5% (339.4) | +1.8% (940.7) |

Two FFN nodes with one Attention node (2 × H200 each, 6 ranks) pass BF16
small/large, FP8 large and the baseline run, and measure 74.7 µs at one token
against FastAFD M2N's 104.4 (−28.5%) to −3.1% at 2,048, every cell faster than both
controls in all repeats (same record).

Two FFN domains with two Attention domains (four single-GPU pods, each its own
domain over InfiniBand GIN; three share a node, so absolute latency is not a
deployment number) also pass BF16/FP8 small and large and the baseline run,
and the directional protocol measures 76.5 µs at one token against FastAFD M2N's
118.4 (−35%), −35% up to 32 tokens and −3.1% at 2,048, every cell faster than
both controls in all repeats ([record](validation-four-domain.json)).

Five single-GPU domains (two FFN, three Attention, on three nodes) pass BF16
small/large, FP8 large and the baseline run, and measure 95.1 µs at one token
against FastAFD M2N's 156.5 (−39%) to −2.5% at 2,048, every cell faster in all
repeats ([record](validation-five-domain.json)).

Eight ranks in four two-GPU domains (4 × 2 H200, physical domain [4, 2]) pass
BF16 small/large, FP8 large and the baseline run with two FFN and two
Attention domains and with one FFN and three Attention domains. One-token round
trips: 85.5 µs against FastAFD M2N's 129.4 (−34%) and 100.2 µs against 129.1 (−22%);
−3.0% and −1.4% at 2,048, every cell faster than both controls in all repeats
([record](validation-eight-rank.json)).

The gain over FastAFD M2N grows with the number of domains (two: −17% at one token;
three: −21%; four: −35%; five: −39%), as the union protocol's reverse-direction
tails, counts and barriers scale with domain pairs.

### 7. Directional hybrid protocol

AFD basis: DeepEP's hybrid (multi-domain) protocol is union-shaped. Every rank
exchanges counts with every scale-out peer, sends finish tails to all of them
and forwards for all of them, and both entry barriers include a cross-domain
RDMA barrier, so an Attention domain waits on an FFN domain's zero counts and
tails and vice versa. When every NVLink domain holds one role (FFN ranks divide
into whole domains), dispatch traffic flows Attention → FFN only and combine
traffic FFN → Attention only:

- Attention domains send counts, payloads and finish tails to FFN domains and
  skip count reception, forwarding and combine reduction.
- FFN domains skip scale-out sends, wait only for Attention domains, and signal
  combine arrival to Attention domains only.
- Both hybrid entry barriers keep only their scale-up (intra-domain) half. Every
  exchange-(N+1) cross-domain write follows its reader's exchange-N consumption
  through the round trip (argument beside `afdep_directional_hybrid` in
  `csrc/kernels/elastic/afdep_roles.hpp`).

- In this mode an Attention domain shares no state (no peer writes its tails,
  counters or slot allocators), so Attention domains also skip their scale-up
  barriers: −2.6% at one token (74.80 → 72.87 µs) to −0.4% at 2,048 against
  the protocol without the skip, every size faster in all repeats
  ([record](validation-attention-scaleup.json)).

**Correction (section 11):** as first shipped, this protocol let RDMA staging
overlap inboxes another domain writes. That was a race in layouts with two or
more FFN domains or two or more Attention domains. It is fixed, and those layouts
pay 1–2.7% at decode sizes for the fix. The multi-domain latency records below
predate the fix.

Layouts that mix roles in a domain keep the union hybrid protocol;
`AFDEP_UNION_HYBRID=1` keeps it as a control. A per-phase globaltimer trace
of the union protocol at one token showed the cross-domain entry barriers alone
on the critical path for about 17 µs of the 88 µs round trip.

Cross-node, H2048 random, three repeats, on the final build (each control slower
in every repeat; [record](validation-cross-node-final.json)):

| Arm | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP-comm | 73.0 µs | 80.3 | 83.3 | 103.9 | 201.4 | 542.4 |
| `AFDEP_UNION_HYBRID=1` | +22.4% | +23.5% | +26.5% | +17.7% | +8.3% | +2.1% |
| FastAFD M2N | +20.6% (88.3) | +22.1% (98.2) | +25.2% (104.3) | +16.5% (121.1) | +7.6% (216.6) | +1.4% (550.4) |

FP8 dispatch across nodes matches: 73.3 µs at one token (FastAFD M2N 88.0, −17%),
82.7 µs at 32 (FastAFD M2N 103.7, −20%) and −2.9% at 2,048, every cell faster in all
repeats (same record).

The 14 cross-node cases pass on the final native digest
([record](validation-attention-scaleup.json)); the single-domain matrix ran
on the protocol's first digest ([record](validation-directional-hybrid.json))
and the Attention-domain skip changes only hybrid kernels. The two waldorf nodes
were shared with other workloads.

### 8. Route checks inside the directional hybrid dispatch

AFD basis: the route guard exists only because the union world gives Attention
ranks dummy expert slots. A route into them would send tokens to a rank that
holds no experts. FastAFD M2N never launches it. On the direct path the dispatch
warps already check their own routes. In hybrid mode the guard ran as a separate
non-PDL kernel ahead of every source dispatch. A kernel trace of the two-node T=1
graph showed 0.9 µs for the kernel (1.5 µs at 512 tokens) plus a 0.45 µs edge
before each dispatch. With the directional protocol, the notify and scale-out
warps check full-width IDs and duplicates (`afdep_checked_expert`) before their
first route-derived index, store or put. Mixed-role layouts keep the guard;
`AFDEP_HYBRID_ROUTE_GUARD=1` keeps it as a control.

Cross-node, H2048 random, three repeats. The control was slower in every cell
and repeat; [record](validation-route-guard.json):

| Arm | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP-comm | 70.8 µs | 79.2 | 82.0 | 103.5 | 200.5 | 539.2 |
| `AFDEP_HYBRID_ROUTE_GUARD=1` | +2.7% | +1.2% | +1.4% | +1.1% | +0.7% | +0.5% |
| FastAFD M2N | +25.2% (88.8) | +24.1% (98.4) | +27.3% (104.4) | +17.1% (121.3) | +7.8% (216.1) | +2.0% (550.4) |

On the same native digest:
- the 14 cross-node cases pass;
- the eight invalid-route cases (dummy, duplicate, negative and wide IDs, eager
  and graph) stop at the in-kernel assertion, with no guard kernel launched;
- the single-domain GPU matrix, including its invalid-route cases, passes.

### 9. Attention-domain dispatch entry sync

AFD basis: with the directional protocol, Attention domains run neither half of
the hybrid dispatch entry barrier (section 7). Its grid sync remained. That sync
held every Attention CTA, and so the first put, behind the slowest CTA, but it
guards no cross-CTA state:
- SM0's notify reduction counts its own arrivals;
- send-buffer reuse follows the round trip.

Attention domains now skip it. `AFDEP_ATTENTION_GRID_SYNC=1` keeps it as a
control.

Cross-node, H2048 random, three repeats. The control was slower in every repeat
up to 512 tokens; [record](validation-attention-entry-sync.json):

| Arm | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP-comm | 70.7 µs | 78.2 | 81.2 | 102.6 | 199.6 | 539.3 |
| `AFDEP_ATTENTION_GRID_SYNC=1` | +0.4% | +1.3% | +1.1% | +0.7% | +0.6% | −0.1% |

On the same native digest, the 14 cross-node cases pass. So do the single-domain
GPU matrix (with its invalid-route cases) and the 3A:1F and 1A:3F small and large
cases.

### 10. Single-FFN reduction

AFD basis: DeepEP's FFN combine reduces each token's local top-k rows with a
branch-free prefix of ceil(top_k / ranks) slots. That is an expected local row
count for tokens spread over every rank, and in afdEP-comm it counts the
Attention ranks, which hold no experts. With one FFN rank, every token's top-k
rows are local, so all top_k slots are valid on every token and the whole
reduction takes the branch-free path. `AFDEP_UNION_REDUCE_PREFIX=1` keeps the
union prefix as a control.

Two or more FFN ranks keep the upstream prefix. There a larger prefix is generic
tuning: in a symmetric 4-rank world it is also faster (up to −7.5%). At 2A2F
H4096 the role-sized prefix was 2–10% slower.

Control slowdown against the new path, three repeats; every cell was faster in
all repeats ([record](validation-single-ffn-reduce.json)):

| Layout (GPU, hidden) | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| 2A1F (H200, H2048) | +16.4% (20.1 µs; FastAFD M2N 32.1) | +23.6% | +24.1% | +23.1% | +16.5% | +8.1% |
| 5A1F (H200, H2048) | +16.5% (23.1 µs; FastAFD M2N 35.5) | +23.3% | +23.0% | +10.0% | +11.1% | +5.5% |
| 3A:1F (H100, H2048) | +9.6% (22.4 µs; FastAFD M2N 36.4) | +19.3% | +24.4% | +17.0% | +4.3% | +3.9% |
| 3A:1F (H100, H4096) | +21.1% (27.6 µs; FastAFD M2N 48.2) | +32.9% | +31.6% | +18.2% | +18.9% | +13.4% |

On the same native digest, all of these pass on a 6-GPU H200 pod:
- the GPU matrix with its invalid-route cases;
- 3A:1F, 1A:3F, 2A1F, 4A2F and 5A1F, BF16 and FP8, small and large;
- the 2A1F and 5A1F baseline runs.

### 11. Staging outside remotely written inboxes (fix)

Section 7 dropped the cross-domain entry barriers. Those barriers had also
ordered when staging buffers could be reused, and the dropped part was not
covered by the section 7 argument. That argument covers readers consuming
inboxes, not a sender's NIC still reading its own staging:
- **Attention side.** An Attention domain's dispatch staging overlapped its
  combine scale-out inbox. FFN domain 0 can write its replies there as soon as
  its own dispatch arrives, while this NIC is still reading staging for FFN
  domain 1.
- **FFN side.** An FFN domain's combine staging overlapped its dispatch
  scale-out inbox. Attention domain 1 can write its next dispatch there as soon
  as its own replies arrive, while this NIC is still reading staging for
  Attention domain 2.

The race shows up when one destination lags. Fault injection that defers one
destination's puts reproduces both cases (patches in the record):
- the second Attention domain's replies mismatched 6.1% of elements;
- the second FFN domain hit an illegal address from corrupted tokens.

With one domain on the other side, every reply or next request follows that
domain's per-channel signals, each sent after the channel's puts on the same QP,
so those layouts are safe and unchanged. The fix:
- with two or more FFN domains, Attention domains stage at window offset 0,
  their scale-up inboxes, which are unused in both phases in this mode;
- with two or more Attention domains, FFN domains stage in a slice that
  `calculate_buffer_size` reserves at the window end.

Results ([record](validation-staging-fix.json)):
- Both injected cases pass after the fix.
- The 14 two-node cases, ten three-pod cases (two FFN or two Attention domains,
  BF16/FP8, small/large, generic) and the single-domain matrix pass on native
  digest a32a12d1.
- The two-node layout is unchanged (within ±0.4%).
- The three-pod layouts pay 1–2.7% up to 512 tokens and about 0.1% at 2,048.

### 12. Direct GIN path: in-kernel route checks

With hybrid off across NVLink domains (direct RDMA to every rank), afdEP-comm
used to run FastAFD M2N's union kernels plus the route-guard kernel. Its dispatch now
instantiates the role template with `kAfdFfnRanks = F`, so routes are checked
in-kernel as in sections 8 and 1. The barrier, count and exit protocol is still
union.

Against the guard-kernel control the change is −1.0% to −1.6% at every size up to
512 tokens across two nodes, and it brings the path level with FastAFD M2N. Direct
BF16/FP8 small and large cases and all eight invalid-route cases pass
([record](validation-direct-gin-routes.json)).

Why this path matters at decode sizes: FastAFD M2N with hybrid off is within 4% of
hybrid on at 1–32 tokens across two nodes. With two Attention domains it ties or
beats afdEP-comm's directional hybrid at 8 tokens (97.3 vs 99.1 µs). So the
one-hop direct path is the base for porting sections 1–3 to RDMA.

**Known limitation (pre-existing).** With one or three FFN ranks on 2-GPU pods,
an FFN and an Attention rank share a node, and direct mode fails for FastAFD M2N as
well as for afdEP-comm. The likely cause is DeepEP's direct GIN mode mixing
same-node NVLink stores with RDMA signals. Direct-mode work is therefore limited
to role-pure nodes.

### 13. Single-FFN collect view

AFD basis: with one FFN rank, each Attention token has exactly one reply row, the
FFN's final sum. Only that FFN writes the Attention combine inbox, and only after
the source's next dispatch. So the reduce epilogue is a pure copy.

`source_collect(view=True)` skips it and returns a strided `[T,H]` view of inbox
region 0. The view is valid until the next `source_dispatch` (contract in
docs/api.md). The source dispatch zeroes the rows of tokens with no valid route,
which no FFN writes. Layouts with more than one FFN rank, and calls with weights
or bias, return the ordinary result.

Copy slowdown against the view, three repeats; every cell was faster in all
repeats ([record](validation-collect-view.json)):

| Layout (H100, H2048) | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| 2A1F | +8.1% (19.2 µs; FastAFD M2N 35.3) | +9.9% | +6.1% | +6.6% | +2.7% | +2.0% |
| 3A:1F | +5.0% (20.2 µs; FastAFD M2N 36.9) | +8.3% | +8.2% | +6.2% | +1.9% | +1.5% |

### 14. Role suites (launch only what each role uses)

In afdEP-comm an Attention rank runs {dispatch send, combine receive} and an FFN
rank {dispatch receive, combine send}, but the shared path launched the same kernels, in the
same shape, on both roles. FastAFD M2N ranks send and receive in every phase, so every
warp class works there; in afdEP-comm some classes never do.

This is the idle-warp anomaly. In the shared directional hybrid launches (1F:1A, native
`8205d02b`, one token), both roles launched the dispatch as 24 blocks × 640 threads
and the combine as 24 × 512, each block at maximum shared memory. The Attention
combine held its 24 blocks for 56.6 µs only to wait for replies, the Attention
dispatch carried forward warps that never forward, and the FFN dispatch carried
scale-out warps that never send. In one NVLink domain the FFN's idle payload warps
released the copy epilogue at entry, which then sat on every SM through the wait
(section 15). Before and after:

| Kernel (directional hybrid unless noted) | Shared shapes | Role suites (`9675783b`) |
|---|---|---|
| Attention dispatch | 24 × 640 threads | 24 × 384 (no forward warps) |
| FFN dispatch | 24 × 640 threads | 24 × 384 (no scale-out warps; CTAs own channels) |
| Attention combine | 24 × 512 threads for the whole wait | 1 × 192 (`afdep_hybrid_collect`) |
| FFN copy epilogue, one NVLink domain | launched at receive entry, resident 8.8–11.3 µs | launched at the count trigger |
| Attention SM-µs per exchange (8F:8A, 1 / 512 tokens) | 1,705 / 5,419 | 699 / 1,257 |
| Attention GEMM under load (8F:8A), over solo | 31–44% | 5–7% |
| FFN GEMM under load (one domain, 2A2F, 512 tokens) | 45.8–46.1 µs | 30.2–30.7 µs |

The other cost seen with idle warps, each extra Attention domain adding time at small
batches, is not a warp problem. It is hybrid's per-channel signalling, which the role
path already halves against FastAFD M2N; the 3:1 analysis below measures it and names the
protocol change that would remove it, which section 17 implements.

The role suites (`csrc/kernels/elastic/afdep_suites.hpp`) pick each role's kernel
and launch shape per path. The shared device sources are unchanged (a unit test pins
their hashes). Two role kernels are generated copies of shared kernels
(`tests/role_kernels.py`, byte-checked by a unit test) and one is new. Every
channel, QP, shared-memory slot, count, tail, signal and staging address is
unchanged.

| Path | Attention suite | FFN suite |
|---|---|---|
| Directional hybrid | dispatch: the shared cubin launched without its forward warps (20 → 12 warps per CTA); combine: `afdep_hybrid_collect.cuh`, one CTA that clears and waits for every channel's finish signals (was 24 CTAs × 16 warps) | dispatch: `afdep_ffn_hybrid_dispatch.cuh`, `hybrid_dispatch.cuh` without its scale-out warps (20 → 12); combine: unchanged (both classes work) |
| One NVLink domain | the shared kernels (R10) | dispatch receive: `afdep_ffn_dispatch.cuh` (section 15); combine: unchanged |
| Direct GIN | the shared kernels: the union barriers make every CTA and warp take part | the shared kernels |

The dropped warp classes took part only in synchronisation that counts launched
threads (`__syncthreads`, grid syncs); every `thread_idx`-indexed statement on these
paths is in the notify warps. The collect drops the shared combine's two local entry
grid syncs, which ordered nothing on an Attention domain: neither barrier half runs
there, each tail and signal word belongs to one channel and one thread, and reuse
across exchanges is ordered by the kernel boundary and by the FFN dispatch N+1, which
needs this rank's dispatch N+1 finish tails.

Controls: `AFDEP_SHARED_ROLE_KERNELS=1` restores the shared launches everywhere;
`AFDEP_SHARED_HYBRID_SEND=1`, `AFDEP_SHARED_HYBRID_FORWARD=1`,
`AFDEP_SHARED_HYBRID_COLLECT=1` and `AFDEP_SHARED_FFN_RECEIVE=1` restore one kernel
each. Shapes are invisible on the wire: runs with one role on the role kernels and
the other on the shared kernels pass.

**Where SMs were held.** Each hybrid CTA owns one SM (opt-in maximum shared memory)
and owns channels, so an idle warp class cannot release anything by exiting. SM-µs
reserved per exchange (grid × kernel time, an upper bound), mean over the eight GPUs
of a role, 8F:8A on two 8 × H200 nodes (development digest, same device sources):

| | Role suites | shared shapes | FastAFD M2N |
|---|---:|---:|---:|
| Attention, 1 token | 699 | 1,705 | 2,582 |
| Attention, 512 tokens | 1,257 | 5,419 | 6,426 |
| FFN, 1 token | 1,928 | 1,878 | 2,739 |
| FFN, 512 tokens | 6,557 | 6,553 | 7,235 |

Per-CTA timers on the shared Attention combine (8F:8A, 8 × H100, a separate profiled
build) show how much of that reservation it really used: 1,049 of 1,420 SM-µs at one
token (74%; median CTA lifetime 43 µs of a 59 µs kernel) and 4,416 of 4,710 at 512
(94%). The one-CTA collect holds one SM for the kernel. The FFN keeps its 24 CTAs,
since the 192 channels pair sender and receiver CTAs; it no longer launches the idle
warps, and its SM-µs is unchanged at 512 tokens and 2.7% higher at one.

**Under compute load.** `benchmarks/overlap.py` runs a fixed batch of 2,048³ BF16
GEMMs on a side stream of every rank while the exchanges replay (same counts for
every arm; a synthetic load, not a serving claim). 8F:8A, two 8 × H100 nodes, three
repeats, every repeat agreeing:

| | 1 token | 32 | 512 |
|---|---:|---:|---:|
| Attention GEMM µs (solo 24.3–25.8): role suites / shared shapes / FastAFD M2N | 26.1–26.5 / 33.8–34.3 / 33.3–34.5 | 26.0–26.4 / 32.4–32.9 / 32.3–33.0 | 25.9–26.2 / 35.7–35.8 / 35.8–36.0 |
| Exchange µs under load | 119–128 / 124–131 / 139–150 | 142–144 / 146–152 / 172–177 | 366–368 / 368–371 / 383–394 |

The SMs the collect frees go to compute: Attention GEMMs take 5–7% longer than alone,
against 31–44% with the shared shapes or FastAFD M2N, and the exchanges also get slightly
faster.

**Latency without load.** Round trip, three fresh-process repeats, InfiniBand between
nodes. Each cell: role suites µs (the shared shapes, geometric-mean paired change;
FastAFD M2N µs):

| Layout | T=1 | 8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|---:|
| Two nodes, 1F:1A (2 × H200 each) | 71.2 (+0.4%; 88.3) | 78.0 (+0.9%; 98.3) | 80.5 (+0.5%; 104.2) | 101.7 (+0.6%; 121.0) | 199.3 (+0.2%; 216.1) |
| Three pods, 2F:1A domains | 77.7 (+1.5%; 106.9) | 82.1 (+1.4%; 112.9) | 89.1 (+1.3%; 116.5) | 122.0 (+0.9%; 143.4) | 279.9 (+0.4%; 295.5) |
| Three pods, 1F:2A domains ¹ | 85.4 (+0.7%; 107.4) | 98.9 (+0.4%; 119.7) | 103.2 (+0.5%; 126.0) | 138.9 (+0.4%; 158.6) | 331.0 (+0.0%; 344.0) |
| Two nodes, 8F:8A (8 × H200 each) | 62.7 (+0.8%; 82.8) | 71.9 (+0.5%; 93.4) | 73.8 (+0.6%; 98.9) | 97.0 (+0.5%; 115.2) | 214.1 (+0.3%; 232.6) |

¹ Development digest; its device sources equal the final ones.

The role suites are faster than the shared shapes in 59 of 60 repeat cells (the other is
−0.5%) and than an isolated tree before the role suites (`8205d02b`) in 58 of 60. An earlier two-node
screen on the development digest had 12 of 15; differences at 512 tokens are within
run-to-run noise. FastAFD M2N is 4–38% slower ([record](validation-role-suites.json)).

**Correctness.** Every rank of every run records its native digest. Passing:
- two nodes 1F:1A (all controls), three pods with two FFN or two Attention domains,
  and 8F:8A on two 8-GPU nodes, each with BF16/FP8, small and large;
- on `26cd2d09` (device sources equal to the final ones but for
  `afdep_ffn_dispatch.cuh`, which hybrid does not use), 51 runs on H100 pods, one pod
  per NVLink domain: two FFN *and* two Attention domains at 1, 2 and 4 GPUs per
  domain, two FFN and three Attention domains at 1 and 4, and 8-GPU domains at 2F:1A
  and 1F:2A; each with BF16 and FP8, BF16 at hidden 7168 (7 channels per SM) and runs
  mixing role kernels with the shared ones in either role; on the 2-GPU four-domain layout also
  FP8 at hidden 8192, 23 transport SMs and every control;
- on the final digest `9675783b`: 8F:8A on two 8 × H100 nodes (BF16/FP8, small and
  large, FFN on the shared kernels), the four-domain layouts at 1 and 2 GPUs per domain,
  and hidden 2048, 7168 and 8192 with BF16 and FP8 8192, logging 8, 7, 6 and 7
  channels per SM (`EP_BUFFER_DEBUG`);
- invalid routes stop at the in-kernel assertion in every layout; every valid run also
  changes buckets between back-to-back exchanges with different host counts per role,
  eager and in one graph.

Faults on the four-domain layout: with both recorded deferred-put patches (section
11) applied at once, every run passes and the patches' markers show the deferrals
firing (2,370 and 1,573 times). Negative controls fail as they must: a collect that
never waits (8 mismatches), a collect that waits on FFN domain 0 only (6 mismatches),
and the role kernels with the staging fix reverted (768 timeouts). On the final digest
the two patches again pass, with 2,533 deferrals.

On the final digest, 8F:8A on two 8 × H100 nodes (another cluster than the latency
table above), the role suites are faster than the shared shapes in all 15 repeat cells
(geometric mean 0.3–0.8%) and FastAFD M2N is 7–23% slower.

### 15. FFN receive: copy-epilogue trigger at the counts (one NVLink domain)

AFD basis: in one NVLink domain the FFN dispatch is receive-only. Its copy epilogue
is a PDL dependent of 132 CTAs at maximum shared memory that launches once every
receive CTA has triggered. The FFN's payload warps have nothing to send, so they
reach the trigger right after kernel entry, and the epilogue then occupies every SM,
waiting, for the whole count and arrival wait. In a one-token 2A2F trace of the shared kernel on
4 × H100 it launched 1.2 µs into each FFN rank's dispatch and stayed resident for
8.8–11.3 µs. A FastAFD M2N FFN also sends, so its trigger fires near the end.

`afdep_ffn_dispatch.cuh` is `dispatch.cuh` (generated by `tests/role_kernels.py`) in
which only the notify warps trigger: CTA 0 triggers once every source's counts have
arrived, early enough to hide the epilogue's launch behind the payload. Template,
launch shape, protocol words and every other statement are the shared kernel's.
`AFDEP_SHARED_FFN_RECEIVE=1` restores the shared kernel.

Measured on 4 × H200, fresh-process screens, native digest `9675783b`:

| Without load (5 repeats) | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| 2A2F, µs (the shared shape; FastAFD M2N) | 19.8 (20.2; 28.2) | 23.3 (23.4; 31.4) | 24.6 (24.6; 32.4) | 32.3 (32.4; 40.2) | 66.2 (66.5; 73.0) | 181.3 (181.1; 188.0) |
| 3A:1F view, µs (the shared shape; FastAFD M2N) | 18.5 (18.5; 32.5) | 20.8 (20.6; 35.3) | 23.8 (24.0; 40.0) | 35.4 (35.5; 50.5) | 96.3 (96.4; 110.6) | 322.5 (322.6; 342.6) |

No cell is slower than the in-tree shared shape in all repeats. Against an isolated tree
before the role suites, one cell (2A2F, one token) is slower in 5 of 5 by a median 0.26% (0.05 µs); the
in-tree shape, whose one-domain kernels compile from the same sources as that tree's,
differs from it by up to 2.2% in the same cell.

Under the synthetic GEMM load of section 14 (FFN rank, ranges over 3 repeats; the
role kernel's ranges never overlap the others'):

| Under load | 2A2F, 1 token | 2A2F, 512 | 3A:1F, 1 token | 3A:1F, 512 |
|---|---:|---:|---:|---:|
| FFN GEMM µs (solo 24.5–24.9): role kernel / shared shape / FastAFD M2N | 25.8–25.9 / 28.2–29.5 / 29.2–29.5 | 30.2–30.7 / 45.8–46.1 / 35.5–36.2 | 25.2–26.0 / 31.2–31.6 / 30.8–30.9 | 31.2–31.6 / 40.4–41.2 / 36.0–36.5 |
| Exchange µs | 34.3–34.7 / 35.9–37.0 / 43.4–43.6 | 111.9–112.2 / 120.4–121.1 / 119.4–121.1 | 34.6–34.7 / 37.9–38.6 / 47.4–47.8 | 148.0–149.4 / 150.3–151.7 / 163.6–165.9 |
| Makespan ms | 6.86–6.93 / 7.19–7.40 / 8.68–8.72 | 22.39–22.43 / 24.08–24.22 / 23.87–24.21 | 6.93–6.94 / 7.58–7.71 / 9.48–9.56 | 29.59–29.88 / 30.07–30.34 / 32.72–33.18 |

With the shared kernel, FFN GEMMs at 512 tokens ran slower than under FastAFD M2N (starved
behind the resident epilogue); with the count trigger afdEP-comm wins every cell.

Two other trigger points were measured on the same pod. Triggering after the
arrival wait gave the same compute gain but cost 0.3–0.6 µs without load (the
epilogue's launch is exposed). Triggering at the first count to arrive was slower
than the count trigger at one token on 3A:1F in 4 of 5 repeats (up to 0.6 µs) and
gave back part of the compute gain (2A2F, 512 tokens: FFN GEMM 31.9–33.0 µs against
28.8–30.1). In the two screens before the final digest (scratch builds, same device
code) the count trigger was slower than the shared shape at one token on 3A:1F in 7 of 9
repeats, by about 0.4 µs; in the final-digest run above, in 3 of 5 (geometric mean
+0.14%).

Correctness on `9675783b`: the GPU matrix, both ratios, the collect view, hidden
7168, 7 transport SMs, every control, runs with the FFN on the shared kernel and the
Attention in afdEP-comm, and fault injection (late sender CTA, late prefix sum,
late receiver pass; a receiver counting one arrival per sender returns stale rows and
one expecting an extra arrival times out) ([record](validation-role-suites.json)).

### 16. Count-first dispatch with a streaming FFN expansion (opt-in, one NVLink domain)

AFD basis: in one NVLink domain the FFN only receives in dispatch, so its idle SMs could
expand rows into expert slots while the sources are still pushing. A FastAFD M2N FFN also
sends and must wait for its own sends. The 3:1 analysis below showed why the expansion
cannot start early today: it needs the per-expert prefix sums, and the counts behind
them arrive with the end of the payload on the same link.

With `AFDEP_COUNT_FIRST_MIN_BUCKET=b`, buckets of at least b tokens run three role
kernels (generated copies, `tests/role_kernels.py`, plus `afdep_count_first.cuh`):
- **Sources** (`afdep_source_dispatch.cuh`): every notify thread counts tokens (one round
  instead of up to 22 at 2,048 tokens); the count word carries a per-source epoch in its
  high half; each stored row releases an epoch-valued flag in the FFN's workspace. By
  default payload warps also wait until the counts are issued (a gate);
  `AFDEP_COUNT_FIRST_UNGATE=1` removes it, which measures better (below): fast counting
  alone gets the counts out before much payload is queued, so the transfer need not wait.
- **FFN receive** (`afdep_ffn_stream_dispatch.cuh`): triggers the epilogue once every
  count has arrived and publishes the prefix sums through a release word.
- **Epilogue** (`afdep_ffn_stream_epilogue.cuh`): blocks claim units of rows, chunk-major
  and source-minor, wait for each row's flag and expand it; the last claimant waits for
  the dispatch grid and clears the two cleared words. Flags are epoch-valued and never
  cleared.

The words live in the scale-out signal-tail region that one NVLink domain does not use.
Below b, and by default (b = 0), every kernel is the existing one. Count-first is
visible on the wire, so every rank must use the same configuration; a mismatch is
rejected at construction on every rank. `AFDEP_PAYLOAD_FIRST=1` forces the existing
kernels.

**Why it is off by default.** It was to become the default only if it won with no
small-batch cost (this document's rule): at least 5% faster from its threshold in every
repeat, the gain explained, and FFN compute under load within 1 µs of the control.
3A:1F, collect view, 4 × H200, paired against the in-tree control (3 repeats; this table
and the traces are from `9a775052`, before review changed only assertion placement,
diagnostics and host-side checks; everything else is on `958652a3`):

| | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| Count-first at every size | +13 to +17% | +16 to +19% | +16 to +21% | +3.6 to +3.9% | +0.4 to +0.7% | −11.0 to −11.3% |

At 2A2F it gains −7.2 to −7.3% at 2,048 tokens and −2.9 to −3.3% at 1,024; at 1A:3F,
−1.5% at 2,048. Traces explain both sides: the epilogue now starts 5.7 µs (512) and
6.3 µs (2,048) after the source, instead of 21.7 and 70.5, so the expansion overlaps the
ingress; but the per-row flag releases slow the senders by 4.3 and 9.6 µs, and at 512
tokens the epilogue still finishes no sooner, because it waits for rows in slot order
while they complete in a different order. Under the GEMM load of section 14 at 2,048
tokens, the epilogue resident during ingress takes SMs from the FFN's compute, the
same starvation section 15 removed:

| 2,048 tokens, under load (3 repeats) | 3A:1F FFN GEMM | exchange alone | makespan | 2A2F FFN GEMM | exchange alone | makespan |
|---|---|---|---|---|---|---|
| Control | 34.31–34.46 µs | 326.5–327.5 µs | 95.8–96.2 ms | 32.70–32.79 | 185.5–186.1 | 70.3–70.5 |
| Count-first | 37.86–38.13 | 291.0–294.3 | 89.2–90.0 | 34.30–34.48 | 171.5–171.8 | 68.6–69.0 |
| Count-first, epilogue on 64 SMs | 34.78–35.96 | 292.5–293.6 | 91.4–91.6 | 31.99–32.53 | 171.4–172.5 | 67.4–68.4 |
| Count-first, epilogue on 32 SMs | 33.79–34.28 | 336.1–336.8 | 100.2–100.5 | 32.02–32.92 | 173.3–174.0 | 69.3–69.6 |

Capping the epilogue at 64 SMs keeps the latency gain (−10.4% at 3A:1F, −7.6% at 2A2F)
and improves makespan, but at 3A:1F FFN compute still exceeds the control by up to
1.5 µs, and below 2,048 tokens count-first does not win. It therefore ships off. A
deployment that wants it should set `AFDEP_COUNT_FIRST_MIN_BUCKET=2048`,
`AFDEP_COUNT_FIRST_EPILOGUE_SMS=64` and `AFDEP_COUNT_FIRST_UNGATE=1`. Against the
in-tree control (3 repeats, every repeat agreeing):

| Recommended setting | 1A:3F | 2A2F | 3A:1F |
|---|---:|---:|---:|
| 1,024 tokens (count-first forced on) | – | −2.1% | −4.9% |
| 2,048 tokens | −1.0% (gated) | −8.75% | −11.2% |
| Under load, 2,048 tokens, 3A:1F | FFN GEMM 35.1–35.6 µs (control 33.4–34.5), makespan 91.7–91.9 ms (95.9–96.2) | | |

Below the threshold the setting runs today's kernels: at 1–1,024 tokens it matches both
the in-tree control and an isolated `9675783b` tree within their run-to-run spread
(−1.3% to +1.8%) at all three ratios.

Correctness on `958652a3` with count-first at every size (`MIN_BUCKET=1`): 44 GPU-suite
runs pass: 2A2F, 3A:1F and 1A:3F; BF16 and FP8; hidden 7168 and FP8 8192; the collect
view; 7 and 23 transport SMs; the source-only arm; every screen knob; every control;
and mixed role kernels with `AFDEP_PAYLOAD_FIRST=1`. A configuration mismatch is
rejected at construction, and invalid routes stop at the in-kernel assertion. With
count-first off, the GPU matrix (18 runs) and four afdEP-comm cases pass on the same
digest. Fault injection (3A:1F and 2A2F, every patch's marker firing): late rows, flags
released late after their stores, late counts, a late prefix-ready release and late
prefix sums all pass; an epilogue that skips the row wait gives mismatches, and one that
skips the prefix wait while the prefix sums are late trips assertions. A first version
of that last control delayed only the release word, after the sums existed, and passed:
it did not create the race.
([record](validation-count-first.json))

## Additions after 1.0.0

Sections 17 to 19 were merged after the 1.0.0 release (#3 and #12) as opt-in additions;
v1.0.0 does not ship them. From 1.1.0, section 17 is the directional protocol's default,
with no switch; sections 18 and 19 stay opt-in and off by default.

### 17. Sparse directional hybrid (default across NVLink domains)

AFD basis: with one FFN NVLink domain and S Attention domains (directional hybrid), the
per-channel signals scale with the channel count C, not with the tokens. Every source
sends a finish tail on all C channels (S·C RDMA atomics into each FFN NIC: 2,112 at
11:1 with 24 SMs), the FFN signals every source on every channel in combine (another
S·C, out of the same NIC), and token i of every source rides channel i mod C, so at small
batches all S first tokens queue on FFN forward warp 0. FastAFD's DeepEP M2N has the same
signals, but there every domain sends and receives in both phases. Here the flow is
one-directional, so they converge on the few FFN NICs, and each source knows exactly
which channels it used. Three parts, each through kernels generated by
`tests/role_kernels.py` (`afdep_sparse_hybrid.cuh` holds the words); shared kernels are
unchanged. From 1.1.0 all three always run in the directional hybrid. Until then each was
an opt-in switch (`AFDEP_HYBRID_STAGGER`, `AFDEP_HYBRID_SPARSE_DISPATCH`,
`AFDEP_HYBRID_SPARSE_COMBINE`, with the stagger bound `AFDEP_HYBRID_STAGGER_MAX_TOKENS`),
and the measurements below were taken with them; 1.1.0 removes them, and the kernels are
the ones measured with all three set and no bound.

- **Stagger**: source s puts token i on channel (i + s·C/S) mod C, so the sources' first
  tokens use different FFN forward warps, SMs and QPs. A channel still carries at most
  ceil(T/C) tokens of a source.
- **Sparse dispatch**: SM 0 of each source writes its token count + 1 to a word per
  source domain on every FFN rail peer at entry, and a channel that carried no token
  sends no finish tail. FFN forwarders retire such a lane once the count shows the
  channel unused (data never waits on the count); after the exit barrier's grid sync,
  SM 0 waits for every count and clears it. An FFN NIC then takes S·min(T, C) finish
  atomics plus S count words instead of S·C.
- **Sparse combine**: each source channel records in a local word the FFN domains it put
  a token for. The FFN combine signals a source on a channel only if it replied to that
  source there (it received the source's tokens there in dispatch, and replies to every
  token it received), plus channel 0 to every source. The Attention collect waits for
  exactly those signals.

Why the protocol holds. Each awaited signal still follows that channel's puts on the
same QP, so every reply a source waits for has landed, and the source waits on exactly
the channels the FFN replies on. Channel 0 is a done witness: every combine signal
follows the FFN's dispatch kernel in stream order, so even a source that sent nothing
cannot start its next exchange (count word, tails, inbox writes) before the FFN has
consumed and cleared this one. Count words are cleared in that dispatch kernel, before
any combine signal. The witness must also certify the FFN combine's own reads: a rail
forwarder's combine scale-up inbox (the holders' partials) can overlap the dispatch
scale-out inbox a source writes next (at FP8, once a forwarder has passed a few
thousand tokens to one holder), and a source waits only for its own channels and
channel 0. So every forward warp releases a rank-local DONE count after its last read,
and channel 0 sends the witness only after all C have. An earlier tree (`2d22e5e9`) sent
the witness after channel 0's own warp only; an independent review found the gap (see
below). Staging is unaffected: unused channels stage nothing, and every
used channel's staging reads are covered by the same finish tails and signals as
before (with two or more Attention domains, FFN replies stage in the reserved slice of
section 11). The parts change the wire protocol, so every rank must resolve them
alike; `afdep/deepep.py` gathers each rank's resolved values after construction and
rejects a mismatch on every rank (for example an FFN-only `AFDEP_SHARED_HYBRID_FORWARD=1`,
which turns sparse dispatch off on the FFN ranks only and would otherwise hang); the same
tuple carries the mixed-role F (section 18) and the FP8 combine replies setting and its
minimum bucket (section 19). Other topologies, one NVLink domain included, report the
parts off, so they never reject there. The parts need the role-suite shapes:
`AFDEP_SHARED_ROLE_KERNELS=1` or an `AFDEP_SHARED_HYBRID_*` control set on every rank
turns the parts that need those shapes off (these ablation controls change the kernel
shapes too, so they are not a clean control for the sparse protocol), and the check
rejects such a control set on one role only. The check runs once, at construction, while
the kernels read those controls at every launch: set them before constructing the
`Buffer` and do not change them afterwards.

12 pods × 2 H200 (piggy, InfiniBand, GIN): 2 FFN ranks in one domain, 22 Attention ranks
in 11 domains, FP8, hidden 2048, top-8 of 128 experts, capacity 2,048, 24 SMs (C = 192),
CUDA graphs, 2 fresh processes per arm in rotated order, every Attention rank's round
trip checked. Paired against afdEP-comm 1.0.0 on `958652a3` (1.0.0's dense
protocol); each arm on the digest shown:

| Arm (µs at T=1; change at each T) | T=1 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP-comm 1.0.0 (dense) | 231.3 µs | 259.9 | 266.8 | 285.2 | 321.4 | 460.7 |
| FastAFD's DeepEP M2N (earlier run, same layout) | +43.8% | +33.8% | +29.2% | +26.0% | +21.2% | +4.5% |
| Parts off (`e551ab70`, `841bc731`) | −0.03 to −0.63% | −0.37 to +0.08% | ≤ ±0.17% | ≤ ±0.11% | ≤ ±0.14% | ≤ ±0.08% |
| Stagger (`c301de5b`) | −12.4% | −17.3% | −15.3% | −8.5% | +0.7% | +2.2% |
| Sparse dispatch (`c301de5b`) | −33.8% | −29.3% | −27.0% | −22.4% | −13.6% | −4.3% |
| Stagger + sparse dispatch (`c301de5b`) | −43.4% | −44.5% | −42.2% | −30.6% | −12.6% | −1.8% |
| Sparse combine (`841bc731`) | −2.4% | −1.7% | −0.5% | −0.2% | −0.2% | −0.1% |
| Sparse dispatch + combine (`841bc731`) | −34.6% | −30.5% | −27.1% | −22.4% | −13.9% | −4.3% |
| All three (`841bc731`) | **−67.9%** | **−66.8%** | **−62.3%** | **−50.6%** | **−26.7%** | **−7.1%** |

Then `200944ce` (the bound defaulting to none and the agreement check, before the done
witness below), all three parts, paired again with 256 and 512 tokens added:

| `200944ce`, all three | T=1 | 8 | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Default (µs) | 231.5 | 259.5 | 266.3 | 285.0 | 320.9 | 461.0 | 712.4 | 1,220.1 |
| All three (µs) | 73.8 | 86.2 | 100.4 | 140.5 | 235.1 | 428.1 | 691.2 | 1,221.1 |
| Paired change (both repeats) | −68.2% to −68.0% | −66.8% | −62.3% | −50.7% | −26.8% to −26.7% | −7.2% to −7.0% | −3.0% to −2.9% | +0.0% to +0.2% |
| Against FastAFD M2N (earlier run, unpaired) | −78.3% | −75.7% | −71.5% | −61.2% | −41.2% | −13.4% | – | – |

The gain scales with S, the number of Attention domains, and holds with two FFN
domains (both on `200944ce`, 2 repeats, every repeat faster):

| `200944ce`, all three (default µs, paired change) | T=1 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Two FFN domains: 4 FFN + 20 Attention ranks, 12 × 2 pods | 207.9 µs, −67.1% | 221.0 µs, −64.3% | 224.4 µs, −59.5% | 240.0 µs, −47.0% | 288.7 µs, −27.6% | 403.5 µs, −8.6% | 615.1 µs, −1.7% |
| Three Attention domains: 2 FFN + 6 Attention ranks, 4 × 2 pods (3:1) | 99.7 µs, −33.9% | 115.7 µs, −35.7% | 118.8 µs, −35.2% | 121.2 µs, −29.5% | 132.1 µs, −21.9% | 166.8 µs, −6.0% | 244.4 µs, −3.6% |

On a second cluster with 4-GPU NVLink domains, the parts give similar gains, and the
mixed-role layout shows what remains (2 repeats each; default `958652a3`):

| 6 kermit pods × 4 H200 | T=1 | 8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|---:|
| Single-role domains 5:1 (4F:20A), default µs | 137.8 | 155.3 | 165.0 | 235.1 | 604.8 |
| same, all three parts (`2d22e5e9`) | −48.8% | −50.5% | −40.8% | −2.4% | 0.0% |
| same, FastAFD M2N | +34.5% | +24.7% | +27.7% | +18.2% | +5.2% |
| Mixed-role 11:1 (FFN domain 2F + 2A: union hybrid), afdEP-comm µs | 211.5 | 234.9 | 264.2 | 332.4 | 763.8 |
| same, FastAFD M2N | −0.3% | +0.0% | +0.3% | +0.0% | −0.4% |
| same, 8 SMs instead of 24 | −17.1% | −16.3% | −13.7% | +18.2% | +55.8% |

These 12-domain and kermit 5:1 screens predate the done witness. The tree with it
(`7b8ed8b5`) was screened only on kermit 4F:20A: +6.6% at 8 tokens and within ±1.2%
elsewhere against `2d22e5e9` (see Correctness). The 12-domain layouts have not been
re-measured with it.

In mixed-role domains afdEP-comm runs the union hybrid and matches FastAFD M2N within
0.5%, and its small-batch cost is again channel-bound (8 SMs: −17% at one token, but
+56% at 512). The sparse protocol does not apply there; section 18 adds a directional protocol for
mixed-role domains.

In one NVLink domain nothing changes: every in-node device source, JIT code string and
cache key is identical, every new kernel and launch branch requires two or more NVLink
domains (only the construction-time agreement gather runs in every topology), and a
paired screen agrees (`200944ce` against `958652a3`, 2 repeats; the later done witness and
section 18 are also gated on two or more domains):

| In-node, one NVLink domain of 8 H200, BF16 (1–2,048 tokens) | Default µs at T=1 / 2,048 | Parts off | Parts on | FastAFD M2N at T=1 / 2,048 |
|---|---|---|---|---|
| 6A:2F | 24.3 / 502.0 | −0.6% to +0.2% | −1.2% to +1.0% | +32.3% / +1.2% |
| 7A:1F | 23.8 / 736.8 | −0.8% to +0.3% | −1.2% to +0.5% | +65.2% / +15.0% |
| 4A:4F | 24.3 / 296.3 | −0.2% to +0.8% | −0.6% to +0.9% | +27.8% / +2.2% |

The parts are not additive. Sparse combine alone gains 2.4% and adds 1% on top of
sparse dispatch, but halves the stagger + sparse dispatch time (131 → 74 µs at T=1).
Our reading, not yet traced: unstaggered, all first tokens queue on FFN channel 0, and
that queue hides the S·C outbound combine signals, which the other warps issue
meanwhile; once the stagger spreads the replies over S channels, those signals become
the critical path, and sparse combine removes them. The same interaction explains the
stagger bound: without sparse combine, staggering every batch costs 0.7–2.2% at 64 and
128 tokens (a bound of C/4 avoids it), and with it, staggering every batch wins at
every size (64 tokens −26.6% against −13.7% at the C/4 bound, 128 tokens −7.0% against
−4.3%; `0c22f278`), so the stagger applies to every batch, and 1.1.0 has no bound.

Correctness: the GPU suite passes on 24 ranks across the 12 pods (14 eager cases,
28 graph replays and 728 checked graph exchanges per run, including empty, inactive
and all-skipped sources, skewed routes, delayed replies and dispatches, changing graph
inputs and bucket cycles): all three parts at capacity 8 (every source staggered)
and 64 (bound 16, so full and small sources mix staggered and unstaggered channel maps
in one exchange), and sparse dispatch + combine without the stagger (`0c22f278`).
On `200944ce` it passes again with all three parts at capacities 8 and 64 (24 ranks, digest checked on each), also with two FFN domains (`--ffn-ranks 4`) at both capacities, and an FFN-only `AFDEP_SHARED_HYBRID_FORWARD=1` is rejected with the agreement error on all 24 ranks. On `2d22e5e9` (the agreement check reporting the parts off outside the directional hybrid) it passes on kermit with 4-GPU domains (4 FFN + 20 Attention, 24 ranks), the FFN-only control is still rejected there, and in one NVLink domain the same control with every part's switch set now runs the suite normally. As a negative control, a collect that waits only for channel 0
(`f92c479c`) fails the suite with value mismatches on Attention ranks. The suite now
also has a one-small-source case (source 0 with one token, the others full) with
spread routing (a token's experts on every FFN rank), the condition for the DONE gap.
The runs above predate the done witness. On the tree with it (`7b8ed8b5`) the suite ran
on kermit 4F:20A at capacity 2,048 (FP8, hidden 2048, 24 SMs), with and without the
fault-injection delay below; the other configurations were last run before it.
Three fault-injection runs on kermit (6 × 4 H200, 4 FFN + 20 Attention ranks, FP8,
hidden 2048, capacity 2048, 24 SMs, all three parts, forward warps delayed by about
200 µs before their reads) did not make the tree without DONE fail, so the gap is
argued from the layout and the code, not reproduced; the tree with DONE passes all
three, with and without the delay. DONE costs latency where a source waits on few of
the forwarder's channels: paired on kermit 4F:20A with all three parts (2 repeats,
`2d22e5e9` without against `7b8ed8b5` with), 70.8 / 77.3 / 97.6 / 229.5 / 605.0 µs
become 70.7 / 82.3 / 97.7 / 226.8 / 603.8 µs at 1 / 8 / 32 / 128 / 512 tokens: +6.6%
(5 µs) at 8 tokens, in both repeats, and within ±1.2% elsewhere (default: 138.0 / 155.5 /
164.6 / 235.2 / 604.3 µs). A zero-cost alternative is to place the dispatch scale-out
inbox past the combine scale-up inbox, so the two cannot overlap; not yet done. The comm screen's
own check alone could not show this: every exchange returns the same values. Unit tests
41/41, Ruff clean.

Against this document's rule (ship only role-dependent changes): the implementation
exists only in the single-role directional protocol of section 7, and its
proof depends on each phase flowing one way (sources receive only combine signals, FFN
domains only dispatch traffic). The S:1 ratio is what concentrates the cost on the FFN
NICs. The underlying idea, signalling only used channels, would plausibly also help
DeepEP V2's union hybrid at small batches, where FastAFD M2N spends extra signals on dummy
Attention domains; that is neither implemented nor measured. For 1.1.0 the maintainers
decided to make it the directional protocol's default regardless, since it exists only
in that protocol.

Decision: merged as opt-in (#3); from 1.1.0 always on in the directional hybrid, with no
switch. BF16 was covered afterwards: on the shipped cross-node tree (`34fa5eac`, done
witness) the GPU suite passes on kermit with FP8 and BF16, 4 FFN + 20 Attention and
8 FFN + 16 Attention ranks, at capacities 8 and 64, and a BF16 screen at hidden 4096
(4F:20A) gives −51.4% at one token, −39.8% at 32 and −2.4% at 128 against the dense
protocol. In serving (Qwen3-235B, 20A:4F, 4,096 in / 2,048 out) median TPOT fell by 25.4%
at 4 requests per Attention GPU and 10.5% at 64. The 1.1.0 tree (`21298d5b`, no switch
set) passes the GPU suite on the same pods with FP8 and BF16, 4F:20A and 8F:16A, at
capacities 8, 64 and 2,048, with both ablation controls set on every rank, and rejects an
FFN-only control on all six pods; paired against main's dense tree (`8e63d39f`) it
gives −49.3% (FP8, hidden 2048) and −51.5% (BF16, hidden 4096) at one token, −40.6% and
−41.1% at 32, −5.1% and −2.8% at 128, and −0.3% and +0.9% at 512. Not covered: the union hybrid and the mixed-role path of
section 18. Value, costs and every table: [sparse-hybrid.md](sparse-hybrid.md).
([record](validation-sparse-hybrid.json))

### 18. Mixed-role directional hybrid (opt-in, FFN ranks sharing an NVLink domain)

AFD basis: at high Attention:FFN ratios the FFN group is smaller than a node, so on 8-GPU
(or 4-GPU) nodes the FFN ranks share their NVLink domain with Attention ranks. The
directional protocol (section 7) needs one role per domain, so these layouts fall back to
the union hybrid, which matches FastAFD M2N within 1.4%: AFD buys nothing there. Measured on kermit
(6 pods × 4 H200, FP8), at 11:1 the union path takes 211.8 µs at one token, and its
small-batch cost is synchronisation, not data: cutting the transport from 24 to 8 SMs
saves 17% at one token but costs 56% at 512, and a per-kernel trace shows every GPU,
pure-Attention ones included, spending about 85-115 µs in dispatch at both 1 and 32 tokens.
Splitting nodes into single-role domains (section 17's layout) recovers small batches but
funnels the FFN's ingress through its own few NICs (1.7× slower than the mixed layout at
128 tokens on 3 × 8 GPUs).

With `AFDEP_MIXED_HYBRID=1` (every rank alike, set before constructing the `Buffer`; the
agreement check carries F), a layout
with FFN ranks [0, F), 0 < F < U, runs the directional protocol with one FFN-holding
domain and roles per rank, through two kernels generated by `tests/role_kernels.py`
(`afdep_mixed_hybrid_dispatch.cuh`, `afdep_mixed_hybrid_combine.cuh`); shared kernels are
unchanged:
- every domain-0 GPU stays its rail's forwarder, so remote tokens still enter through all
  U NICs and are forwarded over NVLink;
- only ranks below F hold experts (the in-kernel route bound is F × experts per rank);
- domain 0's Attention GPUs (A0) are also sources: their own tokens take the existing
  local bypass into their own forwarder (rail lane 0), they count to themselves, and
  their own replies are written locally by their forward warps;
- pure-Attention domains run the directional Attention suite (send block, one-CTA
  collect, no barriers), and the rail halves of every barrier are gone.

Why it holds (the comment at `afdep_mixed_hybrid` in `afdep_roles.hpp` has the full
argument): domain 0 keeps all three scale-up barriers across its U GPUs, so its shared
state is ordered as in union; a pure-Attention source writes dispatch N+1 into its
forwarder only after its collect N saw that forwarder's signal on every channel, each
after that channel's combine-N reads. The one region the forwarder still reads after
that is A0's own reply inbox (its reduce epilogue), so every rank's reply inbox moves
into a slice reserved just before the combine staging slice at the window end, past
both layouts; dispatch and combine assert the slices clear each call's layouts.

kermit, 6 pods × 4 H200 (InfiniBand, GIN; the FFN domain holds F FFN and 4 − F Attention
GPUs), FP8, hidden 2048, top-8, capacity 2,048, 24 SMs, CUDA graphs, 2 fresh processes per
arm in rotated order, every Attention rank's round trip checked, every repeat faster:

| 2 FFN + 22 Attention (11:1), 128 experts | T=1 | 8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|---:|
| afdEP-comm, union hybrid, default (`958652a3`), µs | 212.1 | 235.4 | 265.0 | 333.5 | 763.8 |
| Mixed-role directional (`81b71eb0`) | **−23.2%** | **−19.7%** | **−21.7%** | **−14.8%** | **−8.1%** |
| FastAFD M2N | −0.4% | +0.1% | −0.2% | −0.3% | −0.4% |

| 1 FFN + 23 Attention (23:1), 64 experts | T=1 | 8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|---:|
| afdEP-comm, union hybrid, default (`958652a3`), µs | 280.3 | 319.4 | 353.4 | 437.5 | 1,082.7 |
| Mixed-role directional (`81b71eb0`) | **−20.7%** | **−16.2%** | **−16.8%** | **−12.4%** | **−6.2%** |
| FastAFD M2N | −0.4% | +0.1% | −0.2% | −0.3% | −0.2% |

At 11:1 the first screen, on `d1c3c18a` (before the call-top-k slice check, a host-only
change), measured −22.8 / −19.8 / −21.7 / −14.9 / −8.0%. With 128 experts at F = 1 the
union world exceeds DeepEP's expert workspace (24 × 128 logical experts), for union,
FastAFD M2N and mixed alike, hence 64 experts there.

The gain is the synchronisation section 7 removed for single-role domains (rail barriers,
the dense two-way signals and waits, the union count exchange, Attention-domain barriers
and forward warps), now also for mixed nodes; the payload path keeps node-wide ingress,
so the gain persists at 128 and 512 tokens, where 2-GPU single-role domains lose to the
mixed layout.

Correctness on `81b71eb0`: the GPU suite passes on 24 ranks (16 eager cases, 32 graph
replays, 832 checked graph exchanges per run, including the one-small-source case with
spread routing) at F = 2 with FP8 and BF16 at capacity 8 and BF16 at capacity 64 with
hidden 2048 and 24 SMs, at F = 1 (8 experts) and at F = 3 (12 experts); an earlier tree
(`d1c3c18a`) also passed F = 2 FP8 at capacity 64 with hidden 2048. Unit tests 41/41,
Ruff clean. Not yet run: fault injection for the reply-slice and staging orderings,
and a two-buffer interleave.
([record](validation-mixed-hybrid.json))

Not yet: the section-17 sparse protocol on the mixed path (it would need per-lane
count retirement for the self lane, and a DONE witness), more than one
FFN-holding domain, 8-GPU nodes, and a serving run. DBO contract: A0 GPUs join domain 0's
scale-up barriers, so every domain-0 GPU must enqueue two buffers' exchanges in one
order, as union already requires.

### 19. FP8 combine replies (opt-in, across NVLink domains)

AFD basis: in the directional hybrid every reply byte of S Attention domains leaves through
the few NICs of the FFN domain (each FFN NIC sends S_A·T reply rows while each Attention NIC
receives F_D·T). Motivating measurements on kermit (6 × 4 H200, InfiniBand, 4 FFN + 20
Attention ranks, Qwen3-235B shapes, hidden 4096, cross-node tree `34fa5eac`; the same
layout's screens with this switch are in the [record](validation-fp8-combine.json)):
above about 64 tokens per Attention GPU the round trip is bound by bytes through the FFN
node's NICs at 0.33–0.45 µs per arriving token; a token moves about 4.2 KB of FP8 dispatch
(with scales) and 8.2 KB of BF16 combine replies, so replies are two thirds of the bytes,
and the FFN combine kernel takes about 81 µs of a 179 µs round trip at 64 tokens.

With `AFDEP_HYBRID_FP8_COMBINE=1` (every rank alike, set before constructing the `Buffer`),
for every exchange whose bucket is at least `AFDEP_HYBRID_FP8_COMBINE_MIN_BUCKET` tokens
(default 0, every bucket; latched and checked with the switch; smaller buckets keep the BF16
path unchanged, and since every rank passes the same bucket both roles pick the same format):

- An FFN forward warp reduces each reply row exactly as before (the same loads, BF16 adds
  and FP32 accumulation, so the same BF16 row), then quantizes it in registers to E4M3 with
  one power-of-two scale 2^e per 128 elements, e the smallest integer with
  amax ≤ 448 · 2^e (exact from amax's FP32 bits, clamped at −117), and stores it in the first
  part of the same shared-memory TMA row. One TMA store stages it and one put sends
  H + align32(H/128) bytes into the same (slot, token) row of the Attention inbox:
  4,128 B instead of 8,256 B at H = 4096 (−50.0%; −49.8% at 7168 and 8192).
- Row format: E4M3 values at [0, H), then byte g = 127 + e_g, so groups 4p..4p+3 form
  DeepEP's UE8M0 `sf_pack_t` p; no metadata (expanded layout, no top-k weights, asserted on
  both sides). UE8M0 rather than FP32 scales: scaling and dequantizing are exact in FP32
  (every nonzero product stays normal), so the only rounding is the E4M3 cast and the GPU
  suite can check the result bit-exactly; it is also 96 B per row smaller.
- The Attention reduce epilogue (a generated copy of `combine_reduce_epilogue.cuh`, same
  launch shape and slot deduplication) dequantizes each FFN domain's row, sums the rows and
  any bias in FP32 in slot order (ascending last top-k lane per domain), and rounds once to
  BF16.
- Unchanged: the scale-up (NVLink) partials stay BF16, and so do both collects, every
  protocol word, signal and DONE witness, channels, QPs, launch shapes, shared memory, window
  sizing and every shared kernel's bytes, code string and cache key. The kernels are
  `afdep_ffn_fp8_hybrid_combine.cuh` (`afdep_ffn_sparse_fp8_hybrid_combine.cuh` under sparse
  combine), `afdep_fp8_combine_reduce.cuh` and `afdep_fp8_combine_reduce_epilogue.cuh`,
  generated by `tests/role_kernels.py`, plus `afdep_fp8_combine.cuh`.

Why the protocol holds. The format is a function of the bucket, which every rank passes
identically per exchange, so a format change between back-to-back exchanges is always a
bucket change, which already moves every region; each exchange's writer and reader use the
same format, signals are words rather than byte counts, and the FFN rewrites each row's scale
padding, so nothing carries a row stride from one exchange to the next. Both FP8 regions keep their BF16 base addresses: the reply inbox
still starts at the end of the scale-up buffer, and the FFN staging at the BF16 inbox's end
(one Attention domain) or at the reserved window-end slice (two or more). Every FP8 row is
smaller than the BF16 row, so every FP8 byte lies inside the BF16 region it replaces, and
the staging and ordering arguments of sections 7, 11 and 17 apply unchanged: no word, slot,
signal, message count or ordering changes. FFN and Attention compute the same (slot, token)
addresses from the same base, region count and bucket.

Switch and agreement. The row format is on the wire, and a mismatch would mis-decode
silently instead of hanging, so the native `ElasticBuffer` latches the switch at
construction (setting it later has no effect) and reports it, with its minimum bucket, as
the fifth and sixth elements of the tuple section 17 checks; `afdep/deepep.py` rejects a
mismatch on every rank and exposes the values as `Buffer.fp8_combine` and
`Buffer.fp8_combine_min_bucket`. The minimum defaults to 0, every bucket; negative values
clamp to 0, so neither turns FP8 off, and it reports 0 whenever FP8 is off. It is on only in the directional hybrid: one NVLink domain,
mixed-role domains (section 18), `AFDEP_UNION_HYBRID=1`, the baseline's buffers, deterministic mode
and builds without SM90 features report it off, and `AFDEP_SHARED_ROLE_KERNELS=1` turns it off
(set on one role only, the check rejects it). Each combine asserts it still runs the
directional hybrid with the expanded, unweighted layout. It composes with the stagger and
sparse dispatch (dispatch-only), with sparse combine (the FFN runs the sparse FP8 copy), with
both collects and with `AFDEP_SHARED_HYBRID_COLLECT=1` (neither touches reply data), and with
BF16 or FP8 dispatch.

Accuracy. Each FFN domain's BF16 partial is rounded to E4M3 before the FP32 sum: per
element the error is at most 2^-4 of its 128-group's amax (relative 2^-4 in the normal
range, roughly 2.5–3% RMS), elements below roughly 2^-14 to 2^-15 of the amax become
subnormal and below roughly 2^-18 to 2^-19 flush to zero. NaN stays NaN at its own element; an infinity sets its group's
scale to 2^120, so it decodes to infinity and the group's other elements below about 2^110
become zero. The same top scale means a finite group amax of 248·2^120 (about 3.3e38) or
more also decodes to infinity: the top binade of BF16 is not representable. DeepSeek-V3 keeps its combine in BF16; this switch needs a model-level accuracy
check (per-layer MoE-output error, next-token KL and greedy match, a small task suite) before
any serving use, and communication-only evidence supports no quality claim. The GSM8K check
under Status is the task-level part; the per-layer error and next-token KL are still open.

Expected gain, not measured: half the scale-out reply bytes at the same message and signal
counts, so the byte-bound part of the round trip at 64 tokens and above should shrink by up
to a third (the dispatch bytes stay); at 16 tokens or fewer, where latency and signals
dominate, about neutral. One NVLink domain, mixed-role and union paths are unchanged by
construction.

Status ([record](validation-fp8-combine.json), kermit H200 over InfiniBand, 2026-10-23).
- Correctness: on `024e98b0` the GPU suite passes bit-exact against the FP8-aware reference
  on 6 × 4 GPUs at hidden 2048, 4096, 7168 and 8192, FP8 and BF16 dispatch (with
  `fp8_range`), top-k 2, every combination of the sparse parts (then switches), two FFN domains, one Attention
  domain and the shared collect; an FFN-only `AFDEP_SHARED_ROLE_KERNELS=1` is rejected on
  every rank, and mixed-role, union and in-node runs report the switch off. On the gated
  `2cee2f51` ten more runs pass with the minimum inside the capacity, so the bucket cycle
  changes format both ways in one Buffer and one graph, including a minimum above the
  capacity (FP8 latched, never used). The new kernels use 120 (FFN) and 72 (epilogue)
  registers with no spills.
- Fault injection: the section 11 deferred-put patches, ported to the FP8 and BF16 FFN
  combines and the Attention dispatch copies, fire 16k–65k times per FFN rank in 17 runs
  (4F:12A and 8F:8A, FP8, FP8 with the sparse protocol, minimum 32), and every run passes;
  dropping the deferred FP8 replies instead fails both negative controls. A pass shows no
  failure under this delay, not that it reaches the overlap window here; BF16 dispatch and
  the section 17 read delay were not fault-tested.
- Transport (paired, against the sparse protocol, hidden 4096). With two or more Attention
  domains (4F:12A, 4F:20A and 8F:16A on 4-GPU domains; 8F:16A and 8F:24A on 8-GPU domains):
  −3% to −5% at 64 tokens per Attention GPU, −6% to −9% at 128 (one +2.1% on 8F:16A with
  8-GPU domains), −7% to −14% at 256 and −13% to −21% at 512. With one Attention domain
  (4F:4A, 8F:8A), where fewer replies share each FFN NIC: +2% at 64, −1% to −4% at 128–256,
  −10% to −11% at 512. Ungated, 1–8 tokens cost +3% to +8% on 4-GPU domains, which traces to
  the Attention epilogue (3.9 → 7.9 µs, no BF16-add fast path), while the replies' wire time
  falls (the Attention wait drops 188 µs at 512 tokens). With
  `AFDEP_HYBRID_FP8_COMBINE_MIN_BUCKET=64` buckets below 64 are within 0.6% (4F:12A) and
  1.8% (8-GPU domains) of BF16 and the large-bucket gain is unchanged.
- Serving (Qwen3-235B, 4,096 in / 2,048 out, 16A:8F, sparse protocol): TPOT −1.4% to +0.9%
  with the minimum bucket 64; decode buckets there are mostly below 64 tokens, where the
  round trip is not byte-bound.
- Accuracy, task level (tree `b5078d62`, whose native changes since `2cee2f51` are comments):
  GSM8K test (1,319 problems, zero-shot, greedy, thinking off) on Qwen3-235B served by vLLM
  with 16A:8F and the sparse protocol. FP8 replies at every bucket (minimum 0) score 95.07%
  against 95.30% and 95.07% for two BF16 runs. 39 answers differ from the first BF16 run,
  against 35 between the two BF16 runs (McNemar p = 0.75), so the 95% interval for the FP8
  loss is about −0.7 to +1.2 points. FP8 makes greedy outputs diverge earlier (median 66–68
  characters, against 144 between BF16 runs), as rounding should.
- Not yet: per-layer MoE-output error and next-token KL.

Against this document's rule (ship only role-dependent changes): the byte cut itself would
also shrink FastAFD M2N's combine, so it was merged as opt-in (#12), as section 17 was, and stays
off by default; the implementation exists only in the directional protocol, where the S:1
ratio concentrates reply bytes on the FFN NICs. It is also the one place the library
quantizes (AGENTS.md keeps quantization policy with the caller): opt-in, lossy and off by
default.

### 20. Per-domain FFN prefix (FFN ranks on every node)

AFD basis: a serving deployment often scales by identical nodes, each running Attention
and FFN GPUs (4A1F on a 5-GPU domain, 6A2F on an 8-GPU node). FFN ranks are ranks [0, F)
and DeepEP numbers ranks node by node (`nccl.cu:118-119`), so every FFN rank lands on the
first node(s) and these layouts could not run (#18).

With `ffn_ranks_per_domain=f`, local ranks [0, f) of each of the D = F / f NVLink
domains are FFN: FFN index j is rank (j / f)·U + j % f and still owns real experts
[j·E, (j + 1)·E). Callers keep real IDs. The source's route check
(`afdep_domain_routes.cuh`, which replaces the route guard there) validates them as the
guard does and writes DeepEP's logical IDs, e + ⌊e / (f·E)⌋·(U − f)·E, which dispatch reads
and copies for combine. Python requires the physical domains to be D × (W / D) with
hybrid mode on, before the native constructor.

Every domain then holds both roles, which neither directional protocol covers (section 7
needs one role per domain, section 18 one FFN-holding domain). These layouts run the
union hybrid with per-rank roles: both selectors return 0 for f > 0, so the sparse parts
and FP8 combine replies stay off, and shared device sources are unchanged.

Correctness on `ece614e6`: the GPU suite passes on 2 × 4 H200 with f = 1 and 2 (BF16 and
FP8, capacities 8 and 64, hidden 256 and 2048) and on 3 × 4 with f = 1 and 2, every
check exact; FastAFD's DeepEP M2N passes in the same layouts. Duplicate, out-of-range,
negative and wide routes assert in the route check. The global-prefix layouts (one domain,
FFN domain first, union and mixed-role with F < U, FP8 combine) pass unchanged
([record](validation-per-domain.json)).

Results (comm-only, [record](validation-per-domain-screens.json)): kermit 4-GPU domains,
FP8, hidden 4096, top-8, capacity 512, CUDA graphs, two fresh processes per arm, max over
Attention ranks. Per-node FFN runs at FastAFD M2N's speed in the same layout (FastAFD is
0.4–1.7% faster: afdEP-comm's route check is a kernel, and the baseline's IDs were mapped
outside timing). Against today's layouts at the same rank counts it is slower up to 128
tokens and faster only at 512, where spreading the FFN's ingress over every node pays.
Change against per-node FFN (negative: the other layout is faster):

| Layout | T=1 | 8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|---:|
| 2 × 4, F = 2: per-node FFN (f = 1), µs | 120.6 | 145.2 | 155.1 | 180.8 | 388.5 |
| FFN on node 0, union | −2.6% | −2.7% | −1.8% | −1.9% | +4.2% |
| FFN on node 0, mixed-role (opt-in) | −20.5% | −19.1% | −19.2% | −13.9% | −5.3% |
| 2 × 4, F = 4: per-node FFN (f = 2), µs | 103.0 | 116.4 | 121.7 | 147.1 | 294.0 |
| FFN node + Attention node, directional | −27.0% | −26.1% | −22.8% | −18.4% | −8.0% |
| 3 × 4, F = 3: per-node FFN (f = 1), µs | 148.1 | 162.8 | 173.8 | 219.1 | 493.8 |
| FFN on node 0, union | −3.9% | +0.1% | −1.0% | −2.2% | +5.4% |
| FFN on node 0, mixed-role (opt-in) | −23.0% | −18.1% | −15.6% | −12.0% | +1.6% |

So the layout runs, but the union protocol costs what the directional protocols removed:
12–27% up to 128 tokens.

Every rail now holds one role. Rails u < f join only FFN GPUs: no source sits on them, so
all they carry across domains is protocol words (zero counts, finish tails, combine
signals and their rail barrier, which spans rail peers only). Rails u ≥ f join only
Attention GPUs, and each is both a source and its rail's forwarder for the other domains,
so domain ingress uses U − f NICs. Two steps remain. First, FFN rails can skip all of that
on their own. Second, a directional protocol on Attention rails needs its own
ordering argument: dispatch tails and combine signals share words that a dual-role GPU
would receive in both phases, and sources now stage dispatch in domains that hold FFN
ranks. Both need deferred-put fault injection, as section 18 does.

Since then: FFN GPUs skip all of that (silent FFN rails, `afdep_domain_hybrid_*`; per call when
the combine scale-up regions they write fit below a forwarder's unflushed dispatch staging, 2L ≤ U;
[record](validation-silent-ffn-rails.json)), and the layout also runs with hybrid off, where each
Attention GPU reaches each FFN GPU in one RDMA hop ([record](validation-direct-per-domain.json)).
That needed a fix to DeepEP itself: its direct kernels write NVLink-reachable peers with TMA stores,
then pass a GIN barrier that flushes RDMA and signals through the NIC, so a same-node receiver could
read stale rows whenever a node holds senders and receivers (FastAFD M2N too; 100% of a same-node
source's values on 2 × 4 with the FFN on node 0). The GIN barrier now fences at system scope first:
8 of 8 runs passed with it, 8 of 8 failed without it or with only a proxy fence. It costs about
2.5 µs at 1–8 tokens on the direct path and nothing on hybrid paths.

Decode screen, round trip against per-node FFN on the direct path ([record](validation-direct-per-domain-screens.json);
same shapes as above, 4-GPU domains; negative: the other layout is faster):

| Layout | T=1 | 8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|---:|
| 2 × 4, F = 2: per-node FFN (f = 1), direct, µs | 95.9 | 113.2 | 125.9 | 201.4 | 529.3 |
| per-node FFN, hybrid (silent FFN rails) | +7.1% | +11.0% | +9.4% | −18.8% | −30.0% |
| FFN on node 0, mixed-role (opt-in) | −8.2% | −3.9% | −5.6% | −26.7% | −31.7% |
| 3 × 4, F = 3: per-node FFN (f = 1), direct, µs | 106.9 | 117.1 | 139.1 | 288.9 | 901.8 |
| per-node FFN, hybrid (silent FFN rails) | +23.4% | +23.4% | +11.4% | −29.7% | −47.2% |
| FFN on node 0, mixed-role (opt-in) | +2.7% | +10.2% | +0.6% | −35.1% | −44.8% |
| 2 × 4, F = 4: per-node FFN (f = 2), direct, µs | 99.7 | 104.2 | 113.6 | 156.7 | 355.5 |
| per-node FFN, hybrid (silent FFN rails) | −13.0% | −5.9% | −8.1% | −16.7% | −21.6% |
| FFN node + Attention node, directional | −31.5% | −23.2% | −23.0% | −27.8% | −25.9% |

With one FFN GPU per node the single hop wins up to 32 tokens, and on three nodes it beats every
FFN-first layout there. From 128 tokens hybrid wins: in direct mode every remote token for a node's
FFN GPU enters through that GPU's NIC, while hybrid spreads the same bytes over the node's Attention
NICs and NVLink. With two FFN GPUs per node hybrid wins at every size. The direct path still runs
DeepEP's union protocol (FastAFD M2N is within 1.3%); sections 1–3 over RDMA are next.

## 3:1 Attention:FFN: where the time goes (analysis)

Three Attention ranks per FFN rank make one FFN serve three sources. One NVLink
domain, 4 × H200, BF16, hidden 2,048, top-k 8 of 128 experts, 24 transport SMs,
native digest `9675783b` (role suites). Per-kernel times from Nsight traces of graph
replays, max over the ranks of a role ([record](validation-ratio-3to1.json)):

| Layout, tokens per Attention rank | Round trip | Attention dispatch send | FFN expand copy | FFN combine (reduce + reply) |
|---|---:|---:|---:|---:|
| 3A:1F, 512 | 99.8 µs | 23.0 | 18.1 | 54.7 |
| 3A:1F, 2,048 | 328.4 | 79.7 | 77.7 | 175.5 |
| 2A2F, 2,048 | 181.6 | 56.3 | 50.6 | 86.5 |
| 1A:3F, 2,048 | 184.9 | 79.0 | 56.0 | 83.0 |

With one FFN rank every top-k expert is local, so the FFN's data movement is eight
rows per received token, three times 2A2F's per FFN (about four local experts for
twice the tokens):
- **FFN combine, 54% of the round trip.** It reads eight expert rows per token
  (201 MB at 2,048 tokens) at about 1.15 TB/s on its 24 SMs, a quarter of HBM
  bandwidth, and sends 25 MB back at 144 GB/s. It is parallelism-bound.
- **Expansion copy, 24%.** The copy epilogue writes each received token to its eight
  expert slots (201 MB) at about 2.6 TB/s on all 132 SMs, close to HBM-bound, and only
  after the whole dispatch.
- **Dispatch, 24%.** Three sources push 25 MB into one GPU at about 316 GB/s: the
  FFN's NVLink ingress, not the senders' SMs (below).
- The Attention ranks wait for about three quarters of the round trip.

**Transport SMs.** More SMs for every kernel (all ranks, both phases) cut 3A:1F by 20%
at 2,048 tokens (64 SMs) and 2A2F by 9%; 3A:1F at 512 by 15%. Giving the extra SMs to
the FFN's combine alone (a scratch host switch, measured under the grid-sync exit of
section 3, which costs about 3 µs at small T) recovers all of it: −15% at 512 and
−20% at 2,048 with 64 SMs, −16% and −20% with 132. Dispatch does not gain, so it is
bound by FFN ingress.

**Not an AFD change.** A 64-SM combine grid for every rank, with the per-CTA exit
intact (only the SM count passed to combine changes), gives afdEP-comm −15.6% at
512 tokens and −20.1% at 2,048, and gives FastAFD M2N −13.9% and −21.1%. afdEP-comm
stays ahead of FastAFD M2N at either grid (14–17% at 512 tokens, 5–6% at 2,048). The only
role-specific difference is small: under the GEMM load of section 14 the Attention
GEMM rises 0.5 µs at 2,048 tokens in afdEP-comm (its extra combine CTAs exit at
once) and 1.3 µs under FastAFD M2N. The FFN GEMM rises 0.5 µs in afdEP-comm. Sizing the
combine grid by the FFN's load is therefore a deployment setting for any EP backend at
high Attention:FFN ratios, not an afdEP-comm change, and is not shipped (as R1–R3).

**AFD-specific candidate: expansion during ingress (section 16).** The
FFN sends nothing in dispatch, so its idle SMs could expand rows while the sources are
still pushing; a FastAFD M2N FFN also sends and cannot. R4 tried this at 2A2F and gained
nothing. At 3A:1F the copy is 24% of the round trip, but a timing bound, with the copy
epilogue started at the FFN receive's count trigger without its whole-dispatch
dependency (wrong data; scratch `073ced76`), gains only 0.6–0.8% at 128–512 tokens and
3.4% at 2,048 (3 of 3 repeats). The traces show why: the count trigger fires when
every source's counts have arrived, and the counts arrive with the end of the payload
(7, 11, 25 and 73 µs after the FFN receive starts at 1, 128, 512 and 2,048 tokens;
the sources send for 5, 9, 23 and 80 µs). The sources' payload warps push from kernel
entry, and the count stores follow on the same FFN ingress link. Rows cannot be placed
before the per-expert prefix sums exist, so streaming the expansion needs the counts
to arrive first.

A stand-in for count-first ordering confirms it: on top of the bound, the sources'
payload warps idle a fixed time at entry so their count stores lead (scratch
`cfc0d694` and an 8 µs twin; 3 repeats, each agreeing):

| 3A:1F, against the final digest | T=1 | 512 | 2,048 |
|---|---:|---:|---:|
| Bound (epilogue at the count trigger) | −1.9 to −5.9% | −0.4 to −1.4% | −3.3 to −3.7% |
| Bound + 3 µs payload delay | −1.0 to +0.4% | −13.2 to −13.4% | −4.2 to −4.5% |
| Bound + 8 µs payload delay | +25 to +27% | −8.8 to −9.0% | −11.9 to −12.1% |

The lead the counts need grows with the token count, and at one token a short lead
is free because the FFN waits for the counts anyway. A sender that releases its
payload exactly when its counts are published therefore bounds the gain at about 12–13%
from 512 tokens. Shipping it needs two changes, both in afdEP-comm only: the
Attention dispatch publishes counts before payload, and the FFN copy epilogue starts at
the count trigger and waits per chunk of slots (sources interleaved, a counter per
source and chunk, so the fence is paid per chunk rather than per token as in R4).
FastAFD M2N cannot do either: its FFN also sends, and every rank's epilogue must wait for
its own sends. Section 16 builds it; it gains −11% at 2,048 tokens but not at 512, and
ships off by default.

**Across nodes (directional hybrid).** One 4-GPU FFN domain and three 4-GPU Attention
domains (4F:12A) on four 8 × H100 nodes with one 400 Gb/s InfiniBand port per GPU,
final digest, 3 repeats:

| 4F:12A | T=1 | 8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|---:|
| Role suites, µs (the shared shapes; FastAFD M2N) | 108.7 (+0.7%; +29%) | 127.8 (+0.2%; +19%) | 137.4 (+0.1%; +19%) | 194.4 (+0.1%; +12%) | 493.5 (+0.05%; +4.7%) |

Without compute load the role suites match the shared shapes here; their gain is the SMs
they free (section 14). The time is again on the FFN side: at 512 tokens an Attention
rank's sends are issued in 24 µs, but the FFN domain's dispatch (RDMA receive and
NVLink forwarding) takes 312 µs, its expansion 30 µs and its combine 148 µs, while the
Attention's one-CTA collect waits 461 µs. Each FFN GPU receives the three Attention
GPUs on its rail, 1,536 tokens (6.3 MB) in 312 µs: about 20 GB/s of a 50 GB/s port,
so not line rate. It is not channel parallelism either: with 16, 24, 32 and 48
transport SMs the 512-token round trip is 479, 493, 463 and 483 µs (more channels only
add latency at one token, 100 → 143 µs). Nor is it the three-into-one fan-in: at 1:1
(4F:4A) an FFN GPU's dispatch takes 135.6 µs for 512 tokens and 289–290 µs for 1,536,
against 312 µs for the same 1,536 tokens from three sources. The limit is the FFN
GPU's own receive-and-forward rate, about 21 GB/s here whatever the fan-in, and 3:1
triples the tokens each FFN GPU carries. That rate belongs to the transport and
applies to FastAFD M2N as well.

**Per extra Attention domain at small batches.** At one token per Attention rank,
4-GPU domains, final digest, 3 repeats:

| Round trip at T=1 | 1 Attention domain (4F:4A) | 2 (4F:8A) | Per extra domain |
|---|---:|---:|---:|
| afdEP-comm, directional hybrid | 77.7 µs | 88.8 | +11.2 |
| FastAFD M2N, hybrid | 95.5 | 116.6 | +21.1 |
| Direct, afdEP-comm (= FastAFD M2N, section 12) | 92.7 | 96.0 | +3.3 |

Direct pays little per source but is slower than hybrid from 32 tokens (131 against
108 µs with two domains, 761 against 351 at 512), so it is no 3:1 answer. The hybrid
cost grows with channels × sources: at one token the latency slope per added channel
(64 to 384 channels) is 0.061 µs with one source, 0.093 with two and 0.168 with three,
and the two-source gap extrapolates to about 7 µs at zero channels. In the code every
channel sends a finish atomic to each FFN domain in dispatch, and the FFN sends a
signal per channel to each Attention domain in combine: 2 × channels × sources
atomics through each FFN GPU's NIC per round trip (1,152 at 192 channels and three
sources), all but 2 × sources of them from channels that carry no data at one token.
Every source's first token also rides channel 0, so the FFN forwards and reduces them
one after another (the zero-channel term). Splitting each channel's sources across a
second set of FFN warps cannot help: the waits on sources already run in parallel, one
lane each, and an empty source costs one loop iteration.

Removing the signals needs the FFN to know which channels carry data before it waits,
which is count-first dispatch with per-channel occupancy: a wire-protocol change,
AFD-specific (counts flow one way; the FFN never sends in dispatch), modelled at about
−32 µs at 4F:12A and one token, and the same change the one-domain expansion overlap
needs. It is not part of the role suites, which keep the shipped protocol; section 17
implements it across NVLink domains (the sparse protocol). Fewer channels
are a protocol-neutral setting (16 instead of 24 SMs saved 9 µs at 4F:12A and one token,
flat at 512) but help FastAFD M2N as well.

## Parked

Relay dispatch, for layouts with more FFN than Attention ranks, is validated and
kept on an unpublished branch. Its 1A:3F results: −6% at 128 tokens and −22%
at 2,048. It is neutral or slower in a symmetric EP world. It is parked because
the AFD plugin's layouts all have N ≥ F.

## Rejected

### R1. FFN combine sizing and unroll 2 (generic)

The FFN combine reduction's branch-free top-k prefix is `ceil(top_k / ranks)`,
with dummy Attention ranks counted, and it unrolls by four. An AFD prefix of
`ceil(top_k / ffn_ranks)` spills at unroll 4, so it shipped with unroll 2. The
audit applied unroll 2 to FastAFD M2N's combine: FastAFD M2N alone gained up to
20% (H4096), and the AFD prefix over upstream's was mixed (better at H2048
512–2,048 tokens, worse at H4096 8–128). The change is removed.

### R2. Few-token split combine (generic)

Spreading a few tokens' hidden chunks across idle combine warps cut FFN combine
latency, but the same split applied to FastAFD M2N gained up to 32% (H4096, 32
tokens): upstream's one-warp-per-token loop is slow for both. Removed.

Audit, H4096 random, µs ([record](validation-generic-audit.json)):

| Arm | T=1 | 32 | 128 | 2,048 |
|---|---:|---:|---:|---:|
| FastAFD M2N | 38.2 | 55.1 | 68.9 | 391.8 |
| FastAFD M2N + unroll 2 | 34.8 | 46.5 | 55.2 | 341.0 |
| FastAFD M2N + split | 31.1 | 37.4 | 56.1 | 387.1 |
| FastAFD M2N + unroll 2 + split | 28.8 | 32.5 | 46.5 | 341.4 |
| afdEP-comm with R1 + R2 | 20.5 | 24.5 | 44.6 | 332.6 |
| Same, upstream prefix | 20.1 | 23.6 | 40.0 | 333.8 |

### R3. Transport grid per bucket (generic)

A full-device transport grid for buckets of at least 512 tokens was 10–16 points
faster than the 24/132 default, but the same knob speeds up FastAFD M2N equally: it is
a generic DeepEP launch setting, not an AFD property, so afdEP-comm does not ship it.
The screens remain in [the record](validation-grid.json). Sections 4–7 below
were measured while that policy was active: their paired changes compare arms
that shared it, but their absolute 512- and 2,048-token values include it.

### R4. Streaming FFN expand (no gain)

AFD basis: an FFN rank sends nothing in dispatch, so its copy epilogue could
expand slots while Attention sources are still pushing, instead of waiting for
the whole dispatch kernel. The candidate had sources release a per-slot ready
flag after each TMA store completed; the FFN dispatch kernel released its PDL
dependent at entry, and the epilogue waited for the prefix sums and then for
each slot. It was slower than its own control at every size (H2048, 2A2F):

| Variant | T=1 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|
| Control (whole-dispatch dependency) | 19.6 µs | 30.3 | 67.9 | 185.7 |
| Per-slot release flags | 22.9 | 35.4 | 72.7 | 193.3 |
| Relaxed flags, no fence (unsafe timing bound) | 21.7 | 32.3 | 69.6 | 188.1 |
| Relaxed flags, back-off polling (unsafe) | 25.8 | 38.0 | 76.2 | 197.1 |

Removing the per-token fence recovers part of the loss but no overlap gain
appears, so the change is not shipped
([record](validation-stream-rejected.json)).

### R5. Attention receive merge (no meaningful gain)

AFD basis: Attention ranks send nothing in combine, so their main combine kernel
only waits for FFN arrivals before the PDL reduce epilogue. A scratch build
skipped it and let every reduce-epilogue CTA wait on the arrival counters. Paired
against the two-kernel path (H2048, three repeats) it changed one-token latency
by −1.3% (22.67 → 22.45 µs) and other sizes by −1.8% to +0.4%, while occupying
every Attention SM for the whole wait instead of one. Not shipped
([record](validation-merge-rejected.json)).

### R6. Voided screen of the per-CTA exit

The first screen of section 3's per-CTA release selected the tree with
`PYTHONPATH` while `python -m` put the HEAD tree first on `sys.path`, so both
arms ran HEAD ([void record](validation-cta-exit-rejected.json)). Screens
now isolate trees with `PYTHONSAFEPATH=1` and record each arm's native digest;
the re-run shipped the change.

### R7. Hybrid role-sized buffers (latency cost)

AFD basis: with the directional protocol each role uses only part of the hybrid
window, so the inboxes can be sized for the Attention source domains (dispatch
scale-up/scale-out, combine scale-up and staging) and the FFN domains (combine
scale-out). Across two nodes this halved registered transport memory (78.8 →
40.0 MB at capacity 2,048, H2048) and passed every cross-node and single-domain
case, but the union-buffer control was faster in every repeat up to 512 tokens
(one token: 73.07 vs 74.10 µs, −1.4%). Latency is the primary objective and the
saving is about 39 MB per rank at this capacity, so it is not shipped
([record](validation-hybrid-buffers-rejected.json)).

### R8. Per-CTA counts (slower)

AFD basis: sources never consume counts, and with per-CTA exit every source CTA
already releases its own arrival, so its partial counts could travel with it
instead of through SM0's grid-wide reduction and put (a one-token trace showed
SM0 releasing at 2.1 µs, the other CTAs at 0.5 µs). Each source CTA added its
nonzero counts into the FFN inboxes before its release, and FFN ranks read them
after every arrival. The SM0 control was faster in every H2048 repeat (one token
20.65 vs 21.01 µs, −1.3% to −3.1% up to 512 tokens): moving the prefix sums
behind the arrival and completing the remote count atomics before each release
cost more than SM0's late release. Not shipped
([record](validation-per-cta-counts-rejected.json)).

### R9. Shared arrival word (about 1%, inconsistent)

AFD basis: receivers of the directed exit poll one counter per sender in turn,
and an FFN rank starts that wait only after its count wait. FFN ranks send no
payload, so a dispatch warp could wait from kernel entry, with all senders adding
into one word.

Against its control, the shared word was:
- 0.5–2% faster at 2A2F;
- 1–3.5% faster at 3A:1F, consistent at 32 and 128 tokens;
- mixed at 1A:3F, where one token was slower in 3 of 5 repeats.

In two of the three layouts the repeats disagreed on the winner. It is not
shipped ([record](validation-shared-arrival-rejected.json)).

### R10. One-CTA receive in one NVLink domain (nothing to free, latency moved)

The receive-only kernels of one NVLink domain (FFN dispatch, Attention combine)
launch the senders' 24-CTA grid, although only CTA 0 waits. A role-sized receive
launches CTA 0 alone (the same cubin, one CTA). It passed the GPU matrix, both
ratios, the collect view, 7 and 23 transport SMs, every control, mixed-role runs
and fault injection: a late sender CTA, a late prefix sum and a late receiver all
pass. A receiver counting one arrival per sender returns stale rows, and one
expecting an extra arrival times out.

There is little to free: with per-CTA timers, the shared launch's CTAs 1–23 live 0.6 µs and
exit, and CTA 0 owns its SM either way. Latency against the shared shape (H100, 2A2F,
fresh-process screens):
- On the candidate digest (`a11fe415`), one token was slower in 6 of 8 repeats over
  two pods (3 of 3 on one, median +0.22 µs; 3 of 5 on the other, median +0.04 µs)
  and eight tokens faster.
- On scratch builds that only add per-kernel host switches (same device code), one
  token was slower in 16 of 16 repeats (+0.3–0.4 µs) and eight tokens faster in
  most.
- Launched as one cooperative CTA, or as two clustered CTAs, the one-token cost
  shrank only partly (−0.4% and −0.5% against the one-CTA launch; the shared shape was
  −1.3% in the same screen).
- 24 CTAs with the idle warps not launched stayed within ±0.6% of both the in-tree
  shape and an isolated tree before the role suites at every size except 128 tokens (+1.2–1.3%, two
  repeats at 3–4%) and 2,048 (+0.1%).

No mechanism was found. An in-process A/B is not possible with this DeepEP build
(two Buffers in one process fault with an illegal address, also without the role suites, and two
graphs captured on one Buffer corrupt each other), and single-process traces vary
more between processes than the effect. So the one-NVLink-domain suites keep the shared
launch shapes. What the FFN receive did hold, the copy epilogue, is section 15
([record](validation-role-suites.json)).

### R11. Batched count-first flag release and slot-at-issue (slower)

Two fixes for section 16's measured costs. Batched release: payload lanes post completed
rows to a shared-memory mailbox and the idle notify warp releases their flags with one
system fence per batch. Slot-at-issue: the first gated token of each warp takes its slot
after the gate, so slot order follows issue order. Both passed review and compiled
(`00a7c096`), but neither helped. 3A:1F against the control, 3 repeats:

| | 512 | 1,024 | 2,048 |
|---|---:|---:|---:|
| Both | +1.3 to +1.7% | +0.7% | −8.6 to −8.7% |
| Slot-at-issue only (per-row release) | +1.0 to +1.4% | −3.0 to −3.1% | −11.0 to −11.1% |
| Batched release only | +0.7 to +1.0% | −0.7 to +0.4% | −8.7 to −9.6% |

The signaller's lag costs more than the per-row fences it saves, and slot order alone
does not fix the epilogue's wait at 512 tokens. Section 16 keeps per-row release.
