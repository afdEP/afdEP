# Third-party notices

AFDEP's own code is licensed under Apache-2.0. Its small NCCL host runtime
contains code adapted from DeepEP and remains licensed under MIT.

## DeepEP host runtime

- Upstream: [DeepEP](https://github.com/deepseek-ai/DeepEP).
- Copyright: Copyright (c) 2025 DeepSeek.
- License: MIT; the complete text is shipped in
  [`afdep/third_party/deepep/LICENSE`](afdep/third_party/deepep/LICENSE).
- Adapted files: `afdep/csrc/nccl_runtime.hpp` and
  `afdep/csrc/nccl_runtime.cpp`.
- Origin: the DeepEP V2 snapshot vendored by FastAFD at recorded abbreviated
  commit `3c71619`, then copied into AFD plugin PR #23. The snapshot does not
  record the original DeepEP commit or the full FastAFD revision. Exact input
  file SHA-256 hashes and the extraction differences are recorded in
  [`PROVENANCE.json`](afdep/third_party/deepep/PROVENANCE.json).

The extracted runtime implements NCCL communicator bootstrap, topology queries,
registered windows, and a GPU-initiated rail communicator. AFDEP does not ship
or invoke DeepEP's expert dispatch/combine kernels, dummy expert layouts, or
its generic JIT compiler. No DeepGEMM, CUTLASS, or fmt source is included.

## AFD plugin directional protocol

AFDEP's Apache-2.0 directional kernels, layout, and endpoint originated in the
[AFD plugin](https://github.com/ms-llmd/afd-plugin) experiments on the PR #23
base commit `7cc7ebd2283eff0388a7aa9495a9445a1cbd8349`. The directional code was
an additional measured patch, rather than part of that commit itself. Input
hashes, the measured snapshot fingerprint, and the host launcher extraction
are recorded in [`afdep/csrc/ORIGIN.json`](afdep/csrc/ORIGIN.json).

The original AFD plugin attribution is retained in the extracted files.
AFDEP compiles those kernels directly into its native extension instead of
using DeepEP's per-kernel runtime compiler.

## External runtime dependencies

PyTorch, NVIDIA CUDA, and NVIDIA NCCL are external build/runtime dependencies.
Their libraries and headers are not copied into this source distribution and
remain subject to their respective licenses.
