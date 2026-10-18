# afdEP engineering

afdEP is a general EP communication library for AFD: N Attention token sources
communicate with M expert workers. Keep model/GEMM, routing, quantization policy,
and serving scheduling outside the core. No DeepEP or vLLM runtime dependency.

## Scope

[README support](README.md#support) owns architecture/topology support and tested
hardware. Group the support matrix by architecture, not GPU model.
Implementation limits belong to backends, not the library's identity. Do not
claim unvalidated hardware or general N/M layouts are implemented. Do not treat
unvalidated hardware as rejected unless the code actually rejects it.

[docs/api.md](docs/api.md) owns tensor, weighting, stream, graph, and lifecycle
contracts. Preserve device epochs and every reader credit. Readiness must cover
all active GIN contexts, including empty stripes. New topologies/lanes require
explicit addressing and ownership designs. Validate index bounds and alignment.

## Changes and checks

Use named constants, explicit state owners, and module-scope imports except
intentional lazy runtime loading. Avoid mutable globals and unrelated changes.
Python/build changes: run unit tests and Ruff. Native behavior/build changes: run
affected backend's GPU suite, including empty/skewed routes, changing graph
inputs, delayed peers, and buffer reuse. CPU tests do not validate transport.
Check installed distributions include runtime sources, licenses, and provenance.

Retain Apache/MIT notices and frozen origin hashes; document upstream-derived
changes. Preserve frozen validation JSON. [docs/performance.md](docs/performance.md)
separates the October 15 H100 prototype comparison from October 18 package checks.
Keep profiled/unprofiled runs separate, report memory metrics accurately, and do
not attribute polling time solely to network or bookkeeping. No serving claims
from communication-only evidence. Exclude credentials and private cluster data.
