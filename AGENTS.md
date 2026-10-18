# AFDEP engineering instructions

AFDEP is a communication library. Keep model weights, expert GEMMs, routing
policy, quantization policy, and serving schedulers outside its core. It has no
runtime dependency on DeepEP or vLLM. Preserve the explicit role-specific API.

## Supported scope and contracts

- The initial backend is Hopper SM90, exactly two pure-role four-GPU NVLink
  domains: FFN group ranks 0–3 and Attention group ranks 4–7. It has one in-flight
  lane. Reject unsupported layouts; do not imply arbitrary N/M support.
- Exchange kernels run on the current PyTorch CUDA stream. Preserve same-stream
  ordering, caller-owned tensor lifetimes, and collective construction/close.
  Failed distributed jobs terminate and reconstruct; there is no rank-local
  recovery or automatic retry contract.
- FP8 dispatch is caller-prequantized E4M3FN with FP32 block-128 scales. Replies
  are BF16. Expert rows passed to `worker_reply` are already weighted exactly
  once; AFDEP sums them without applying routing weights again.
- Receive allocation length is not a valid-row count. Expert ends include
  alignment gaps before segments. Receipts restore the source/route mapping;
  preserve their opaque, exchange-specific ownership.

## Protocol safety

Before changing ordering or reuse, identify every writer and reader, the exact
completion boundary, and the release/acquire or GIN handoff that protects it.
Request credits protect all FFN packing readers, partial credits protect all
ingress reductions, and reply credits follow the complete Attention copy grid.
Every active GIN context signals and is awaited, including empty stripes.

Device epochs must advance on graph replay without Python counter updates.
Do not remove waits based only on a zero-count example. Empty participants still
advance the protocol. Multiple lanes or topologies require a new addressing and
reader-credit design with corresponding validation.

Validate dimension/index arithmetic before allocation and check tensor dtype,
shape, device, contiguity, and vector-pointer alignment before launching. Build
with the same NCCL version that is loaded at runtime. Preserve cache integrity
and process locking when changing the extension loader.

## Implementation and validation

Use descriptive names and named constants, keep imports at module scope except
for deliberate lazy loading of optional runtime dependencies, and avoid mutable
global state. Keep Python integration, native protocol, and transport bootstrap
separate. Favor focused changes over unrelated refactoring.

Run the relevant CPU checks after Python/build changes:

```bash
python -m pytest tests/unit
python -m ruff check .
python -m ruff format --check .
```

Native changes require a supported Linux/Hopper extension build and eight-rank
BF16/FP8 eager and graph checks. Cover empty/skewed routing, changing routes and
token signatures, interleaved buckets, delayed peers, and buffer reuse. Check
the installed wheel/source distribution contains every runtime source, header,
license, and provenance file. Do not substitute CPU configuration tests for GPU
transport validation.

## Provenance and measurements

Retain Apache-2.0 notices for AFDEP/AFD-derived code and the MIT notices on the
adapted DeepEP host runtime. Keep upstream inputs and extraction differences
recorded in the provenance files. Mark and explain local upstream-derived
changes; do not silently overwrite attribution or frozen origin hashes.

Historical October 15, 2026 measurements precede this standalone extraction.
Label them as inherited evidence until this package is built, validated, and
remeasured. Keep profiled runs separate from latency runs, distinguish registered
windows from allocator peaks and total VRAM, and report communication-only
measurements as such. Do not attribute polling-kernel duration entirely to wire
transfer or bookkeeping.

Do not commit credentials, private cluster identifiers, hostnames, connection
files, or tokens. Use generic deployment instructions and public source links.
