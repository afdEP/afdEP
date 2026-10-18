# API and execution contract

## Configuration

`Config` describes one fixed-capacity buffer:

| Field | Contract or initial default |
|---|---|
| `capacity_tokens_per_rank` | Positive upper bound per Attention rank |
| `hidden_size` | Positive multiple of 128 |
| `num_experts` | Positive real expert count divisible by four |
| `top_k` | In `[1, min(32, num_experts)]` |
| `ffn_ranks`, `attention_ranks` | Both four; other values unsupported |
| `expert_alignment` | Positive expert-row alignment; default 128 |
| `payload` | `"bf16"` or `"fp8"` |
| `num_gin_contexts` | Default eight; must support the selected stripes |
| `transport_stripes` | Default eight; positive and no greater than GIN contexts |
| `num_sms` | Default 132; positive and no greater than the device SM count |
| `gpu_timeout_s` | Positive device polling timeout; default 60 |
| `rdma_traffic_class` | RDMA traffic class; default zero |

The default local grid uses all 132 H100 SMs. This was useful for isolated
communication in the inherited prototype. It is not a recommendation for overlap
with expert GEMMs; tune and measure the concurrent workload.

`Buffer(group=group, config=config)` requires an initialized CPU Gloo process
group with eight participants and the placement described in
[architecture.md](architecture.md).
Each process must first select its CUDA device. All participants must use the
same configuration and construct and close buffers in the same order. Do not
construct or close a buffer inside CUDA graph capture.

Set lifecycle/rendezvous timeouts when creating the Gloo process group. AFDEP's
device polling timeout does not override that process group's timeout.

## Source methods

```python
buffer.source_dispatch(x, ids, weights, x_scale=None, *, bucket=bucket)
buffer.source_collect(out)
```

These methods are valid only on Attention group ranks 4–7.

| Argument | Shape / dtype |
|---|---|
| `x` | `[T, H]`, contiguous CUDA BF16, or FP8 `torch.float8_e4m3fn` for FP8 mode |
| `ids` | `[T, K]`, contiguous CUDA INT64 |
| `weights` | `[T, K]`, contiguous CUDA FP32 |
| `x_scale` | FP8 only: `[T, H / 128]` CUDA FP32; strided scales accepted |
| `out` | `[T, H]`, contiguous CUDA BF16, including in FP8 mode |

`T` may be zero. `bucket` must be positive, no greater than configured capacity,
and at least `T`. All workers must use the same bucket for the exchange; use a
shared plan across the full group. The actual source counts may differ by rank.
All tensors must be on the buffer's CUDA device.

`x` and `out` must also have 16-byte-aligned data pointers for vector copies.
Normal PyTorch allocations satisfy this; a contiguous view with a nonaligned
storage offset may not. AFDEP rejects those views before launching the copy.

Valid expert IDs are `[0, num_experts)`. An ID of `-1` disables that routing
slot; other negative IDs and duplicate nonnegative experts within a token are
invalid. IDs refer only to real experts; there are no Attention-owned dummy
expert slots.

FP8 inputs must already be quantized by the caller. Each FP32 scale corresponds
to 128 consecutive hidden channels of one token. AFDEP copies values and scales
without choosing a quantization scheme or converting the activations. Both the
caller and expert implementation must agree on how those scales reconstruct
the values. Replies remain BF16.

`source_collect` waits for the matching reply on the device, copies it to `out`,
and returns the reply-window credit. Completion of the Python call means kernels
have been enqueued; it does not establish CPU-visible completion of `out`.

## Worker methods

```python
received = buffer.worker_receive(bucket=bucket, read_host_counts=False)
buffer.worker_reply(weighted_rows, received.receipt)
```

These methods are valid only on FFN group ranks 0–3. A worker owns the contiguous
range `[rank * E / 4, (rank + 1) * E / 4)` of real expert IDs.

`worker_receive` returns `ReceiveResult`:

| Field | Meaning |
|---|---|
| `recv_x` | Contiguous expert-major activation rows, in configured dispatch dtype |
| `recv_scale` | FP8 only: FP32 scales `[allocated_rows, H / 128]`, with column-major/padded strides |
| `recv_topk_weights` | FP32 routing weight per allocated row; padding weights are zero |
| `recv_count_psum` | Device INT32 end offsets, one per local expert, including alignment gaps before each segment |
| `recv_counts_host` | With `read_host_counts=True`, per-expert counts rounded up to alignment; otherwise empty |
| `receipt` | Opaque CUDA INT32 mapping for this receive and its matching reply |

For local expert `e`, its valid rows are:

```text
start[e] = 0                                      if e == 0
           round_up(recv_count_psum[e - 1], A)     otherwise
end[e]   = recv_count_psum[e]
```

`A` is `expert_alignment`. Only `[start[e], end[e])` contains that expert's
received activations. Alignment gaps and the unused tensor tail are not valid
activation rows and may be uninitialized. Do not infer valid counts from tensor
length or read padded activations as inputs to meaningful computation.

The conservative allocation bound per worker is:

```text
allocated_rows = 4 * bucket * min(top_k, num_experts / 4)
                 + (num_experts / 4) * (expert_alignment - 1)
```

`weighted_rows` must be a contiguous BF16 CUDA tensor of shape
`[allocated_rows, hidden_size]`, matching this receive's allocation. For each
valid row, supply the caller's expert output multiplied by its corresponding
`recv_topk_weights` exactly once. AFDEP only sums supplied rows; it does not
apply routing weights itself. Reply reduction does not read padding rows whose
receipt entries are absent. Preserve the full required tensor shape even if
only valid rows are computed.

Zeroing unused output rows is a useful application convention, but the protocol
does not require reading or reducing them.

The receipt is specific to this buffer, exchange, bucket, and receive result.
Do not modify, fabricate, reuse for a different exchange, or pair it with a
different output layout. Within-expert row order can vary; the receipt restores
the source token/route association.

Reduction accumulates local expert rows in FP32 and rounds a per-worker partial
to BF16, then accumulates those partials in FP32 and rounds the final reply to
BF16. This two-stage reduction need not be bitwise identical to a single FP32
sum followed by one BF16 rounding.

`read_host_counts=True` copies counts to the CPU and synchronizes. It is useful
for eager host scheduling, cannot be used during CUDA graph capture, and does
not reduce the conservative receive allocation.

## Stream and tensor lifetime

All exchange kernels enqueue on the current PyTorch CUDA stream. AFDEP does not
create an implicit background communication stream or insert dependencies on
arbitrary application streams. Use one ordered stream per buffer. If inputs are
produced or outputs consumed on other streams, establish CUDA events/waits and
tensor lifetimes explicitly.

Keep source inputs alive and unchanged until their dispatched work has finished.
Keep the receive tensors and receipt alive until expert computation and
`worker_reply` have finished using them. Keep `weighted_rows` alive until its
reply work has completed. Keep the buffer alive until all its work completes.
Normal same-stream ordering is sufficient to sequence receive, expert compute,
and reply; a CPU synchronize between those stages is unnecessary.

Single-lane ordering per buffer is mandatory:

```text
Attention: source_dispatch → source_collect → next source_dispatch
FFN:       worker_receive  → worker_reply   → next worker_receive
```

All ranks execute the same number of logical exchanges in the same order. Empty
sources still dispatch and collect, and workers with no selected experts still
receive and reply. Progress depends on those participants; skipping an empty
rank can cause peers to time out.

## CUDA graphs

Initialize and warm up the buffer outside capture so the extension, kernels,
communicator, registered windows, and runtime caches are ready. Capture a
complete role-specific exchange, including the worker's caller-owned expert
computation. Use `read_host_counts=False`; device count metadata must drive
graph-safe expert work.

Captured tensor addresses, shapes, bucket, and launch arguments remain fixed.
Routing IDs, weights, activations, and scales may change in place between
replays with appropriate stream ordering. A different token-count shape or
bucket requires a corresponding captured graph. Replays can interleave
pre-captured buckets when every rank follows the same logical sequence and the
single-lane ordering is preserved.

Device epochs advance during each replay; the protocol does not rely on a
Python counter advancing during capture or replay. Credits protect registered
request, partial, and reply buffers from reuse while peers are still reading.
These guarantees do not permit concurrent graphs on the same buffer.

## Shutdown and errors

Call `buffer.close()` explicitly and collectively, outside capture, after the
last exchange. It synchronizes local GPU work and the group before deregistering
windows. Do not rely on Python garbage collection to perform a collective close.

Invalid device-side routing, bounds errors, or a missing peer may cause a device
assertion/trap or timeout. Do not replay or reuse a buffer after such a failure.
Terminate and reconstruct the participating process group; a device failure may
also leave the CUDA context unusable. The timeout is a diagnostic, not fault
tolerance or an automatic retry mechanism.
