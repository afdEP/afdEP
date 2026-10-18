# Contributing to AFDEP

Keep AFDEP focused on communication between attention and FFN roles. Routing
policy, quantization, model weights, expert kernels, and serving scheduling belong
outside the core. New backends must expose their supported topology and execution
contracts explicitly rather than silently accepting unvalidated placements.

## Development

Use an isolated environment with the repository's development dependencies.
Changes to Python validation and configuration should have focused CPU tests.
Changes to native launch, transport, memory layout, or synchronization require
GPU validation on a supported configuration; a successful import or compile is
not sufficient.

```bash
python -m pip install --no-build-isolation -e '.[dev]'
python -m pytest tests/unit
python -m ruff check .
python -m ruff format --check .
```

Keep Python imports at module scope, use named constants, and avoid mutable
global state. Pass communication state through explicit owners. Keep native
transport and protocol code separate from PyTorch integration and application
logic.

## Protocol changes

Explain the ownership and ordering of every buffer affected by the change. In
particular, identify when remote readers finish and which completion signal or
release/acquire handoff makes reuse safe. Review the checklist in
[docs/protocol.md](docs/protocol.md).

Meaningful GPU checks include:

- BF16 and FP8 routing/output correctness, including empty and skewed cases.
- Changed routes and token signatures across consecutive device epochs.
- CUDA graph replay with stable addresses and interleaved pre-captured buckets.
- Delayed peers and reuse of request, partial, and reply regions.
- All GIN contexts completing, including empty transfer stripes.
- Collective startup, teardown, and rejection of unsupported placements.

Document unsupported cases and fatal-error behavior. Do not introduce a
multiple-lane or general N/M claim without implementing and validating its
addressing and reader-credit design.

## Performance changes

Separate unprofiled latency runs from profiler captures. Record the exact source
revision, toolchain, GPU topology, payload, capacity, token bucket, expert
ownership, alignment, GIN contexts, stripes, and launch grids. State the metric,
warmups, iterations, repeats, and any run-to-run drift.

Keep registered-window capacities separate from allocator peaks and total device
memory. Kernel durations that include polling must not be reported as wire-only
latency. Do not sum overlapping kernels or parallel ranks to estimate elapsed
time. Communication-only measurements must be labeled as such, especially when
local kernels occupy all SMs.

## Upstream-derived code and licensing

AFDEP uses Apache-2.0 and retains the attribution and license terms of copied
upstream components. Keep each vendored component's provenance pinned and
reviewable. Prefer unmodified copies; when a local change is required, mark and
explain it so reviewers can distinguish it from upstream. Do not copy a full
upstream data plane merely to access a small bootstrap or build helper.

Before publishing artifacts, exclude credentials, private hostnames, cluster
connection details, and environment-specific tokens. Hardware and software
configuration should remain sufficient to interpret a benchmark without leaking
access information.
