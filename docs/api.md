# API contract

`Buffer(group=..., config=Config(...))` owns one NCCL communicator and one
vendored DeepEP V2 ElasticBuffer. Initialization and `close()` are collective,
outside CUDA capture. The caller owns a CPU **Gloo** process group, sets the
current CUDA device first, and keeps the group alive until close.

## Roles and tensors

FFN ranks come first; each owns `num_experts / ffn_ranks` consecutive real experts.
Attention ranks follow. Every rank uses identical configuration, exchange order,
`bucket`, and `read_host_counts` policy. Only one unmatched dispatch per Buffer
is allowed. IDs must be unique per token, in `[0, num_experts)`, or `-1` to skip.
Use `buffer.topk_idx_dtype`; invalid real-role routes assert on the GPU, including
on graph replay. All payload tensors are on the selected CUDA device.

| Operation | Contract |
|---|---|
| `source_dispatch(x, ids, weights, x_scale=None, *, bucket, read_host_counts=False)` | Attention: contiguous `x [T,H]`, IDs/FP32 weights `[T,K]`, `T ≤ bucket ≤ capacity` |
| `worker_receive(*, bucket, read_host_counts=False)` | FFN: expert-major rows, weights, device expert prefix sums and opaque receipt |
| `worker_reply(rows, receipt)` | FFN: contiguous BF16 `[receive_rows,H]`, routing weight applied exactly once |
| `source_collect(*, view=False)` | Attention: BF16 `[T,H]` summed over selected experts, in original token order |

BF16 dispatch uses BF16 activations. FP8 uses `float8_e4m3fn` activations and
FP32 block-128 scales `[T,H/128]`; strided scale views are supported. The library
does not quantize or run GEMMs, except the opt-in lossy reply transport
`AFDEP_HYBRID_FP8_COMBINE` below. `H` is a multiple of 256, `K ≤ min(32,E)`, and
real experts divide evenly over FFN ranks.

Receive prefix sum `ends[e]` is a device tensor: expert e occupies
`[align_up(ends[e-1], expert_alignment), ends[e])`, with `ends[-1]=0` for e=0.
Padding rows are unspecified and must not be treated as live tokens. Return the
entire receive shape to `worker_reply`, with valid expert rows in place.
`recv_counts_host`, when requested, reports **aligned** per-expert counts, not
raw live counts. Keep the receipt unchanged through the matching reply.

## Graphs, streams and lifetime

Capture **complete** dispatch/reply pairs with `read_host_counts=False` after
warming up the kernels on all ranks. Keep captured tensors, Buffer and graph
storage alive through every replay. Source tensor shapes and bucket specialize
the capture. Change payloads and IDs in place; encode inactive choices as `-1`.
This is capacity-bounded graph execution, not arbitrary-shape graph replay.
Graph capture/instantiation and replay are different operations; replay still
has a launch cost.

`read_host_counts=True` performs host synchronization for exact eager receive
allocation on FFN. Otherwise receive capacity is conservative and device counts
identify live rows. Attention has zero receive payload rows. FFN bounds count
Attention sources only. Registered transport buffers are distinct from receive
tensors, except for `source_collect(view=True)`. With one FFN rank in one NVLink
domain (and no more ranks than top-k), each token's only reply row is already its
final output. That call then skips the reduce kernel and returns a strided `[T,H]`
view of the registered combine inbox. Treat the view as read-only. It is valid
until the next `source_dispatch` on the same Buffer is enqueued; copy it if the
caller needs it longer. Other layouts return the ordinary result. In one NVLink domain
registered buffers are sized by role: FFN dispatch inboxes for the Attention sources and,
when ranks do not exceed top-k, Attention combine inboxes for the FFN ranks; every rank
registers the same size. `AFDEP_UNION_BUFFERS=1` keeps DeepEP's symmetric sizing as a
measurement control.

Operations order their internal communication stream with the current CUDA
stream using DeepEP events. Use caller stream dependencies before passing data
produced on another stream. Outputs/receipts and user inputs must remain alive
until dependent GPU work completes. A completed Python pair allows the next
pair to be enqueued; device-side protocol and stream ordering protect reuse.
`close()` synchronizes outstanding work and frees registered storage; destroy
graphs before closing the Buffer.

A process may hold several Buffers, for example one per micro-batch. Every rank
must enqueue their exchanges in the same order: by default all Buffers in a
process share one DeepEP communication stream
(`AFD_DEEPEP_PER_BUFFER_COMM_STREAM=1` gives each its own). Each Buffer latches
the `AFDEP_*` controls set when it is opened and keeps them for its lifetime, so
Buffers opened under different controls (an in-process A/B) keep their own launch
shapes and registered layout, and later environment changes only affect Buffers
opened afterwards. `tests/gpu/two_buffers.py` checks both cases, eager and in
graphs.

The `AFDEP_*` variables are measurement controls, read once when a Buffer is
opened. The ones that change barriers, the count exchange, the registered layout
or the hybrid protocol (`afdep.deepep.PROTOCOL_CONTROLS`, plus the count-first and
hybrid settings they resolve) are gathered then, and ranks that disagree all raise
`ValueError` instead of hanging (`tests/gpu/protocol_agreement.py`). The rest are
rank-local launch shapes and may differ by role.

## Transport and tuning

afdEP-comm keeps DeepEP's uniform logical expert numbering (Attention ranks own empty
expert blocks, never legal routing targets) and its collective participation. The
low-level connector interface is `afdep.deepep.ElasticBuffer`: pass `ffn_ranks=M` for
afdEP-comm's transport, as `Buffer` does from `Config`. Without `ffn_ranks` it runs DeepEP
V2 as upstream ships it, over a symmetric group; the benchmarks use that for the
baseline, FastAFD's DeepEP M2N ([benchmarks/fastafd.py](../benchmarks/fastafd.py)).

`allow_hybrid_mode`, `num_sms`, `num_qps`, timeouts and RDMA traffic class pass
through to the pinned transport. `num_sms=0` selects approximately 24/132 of the
smallest device's SM count; choose/tune explicitly for a workload. Both hybrid
settings are validated in one NVLink domain and across NVLink domains joined by GIN.
Across domains, role specialization runs the directional hybrid when every domain
holds one role, and the union hybrid otherwise ([design](design.md), sections 6 and 7).
The directional hybrid always signals only the channels that carry tokens (the sparse
directional hybrid, section 17 and [sparse-hybrid.md](sparse-hybrid.md)); there is
nothing to set. The opt-in mixed-role path (`AFDEP_MIXED_HYBRID`, section 18) changes
the protocol when FFN ranks share a domain with Attention ranks. Set it, and any
`AFDEP_SHARED_*` ablation control, alike on every rank before constructing the `Buffer`
and keep them; construction gathers every rank's resolved values once and rejects a
mismatch.

`AFDEP_HYBRID_FP8_COMBINE=1` (section 19; validated on H200; GSM8K accuracy unchanged, no per-layer or next-token KL check yet) sends the
directional hybrid's combine replies across NVLink domains as FP8: set it alike before
construction, which latches it (later changes have no effect), checks it on every rank and
reports it as `Buffer.fp8_combine`; it is off outside the directional hybrid. It is lossy:
each FFN domain's BF16 partial is rounded to E4M3 with a power-of-two scale per 128 elements
(error at most 2^-4 of the group's largest magnitude; tiny elements flush to zero; NaN stays
in place; an infinity zeroes its group's finite elements unless they are near the BF16 range
limit) before the source sums the domains in FP32. `source_collect` still returns BF16.
`AFDEP_HYBRID_FP8_COMBINE_MIN_BUCKET=b`, latched and checked with it and reported as
`Buffer.fp8_combine_min_bucket`, keeps BF16 replies for exchanges whose bucket is below b
tokens (every rank passes the same bucket, so both roles agree on the row format). The
default 0 means every bucket, negative values clamp to 0, and it reports 0 when FP8 is off.

Native compilation is lazy and process-locked. `AFDEP_CACHE_DIR` controls the
extension cache; `EP_JIT_CACHE_DIR` controls DeepEP kernel JIT cache. Both must be
writable. Use matching NCCL headers/runtime before starting Python; already
loaded libraries cannot be replaced by changing environment variables later.
