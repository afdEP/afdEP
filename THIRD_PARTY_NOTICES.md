# Third-party notices

afdEP-comm builds on [DeepEP](https://github.com/deepseek-ai/DeepEP). Its Python and
build adapters are Apache-2.0; the vendored DeepEP V2 native transport and afdEP-comm's
role specializations of it are MIT-licensed.

| Component | Origin | License | Copyright | What afdEP-comm uses |
| --- | --- | --- | --- | --- |
| DeepEP V2, by DeepSeek | [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) | MIT ([LICENSE](afdep/third_party/deepep_v2/LICENSE)) | Copyright (c) 2025 DeepSeek | The elastic dispatch/combine transport, its kernels and JIT runtime |
| FastAFD, by Hao AI Lab | [hao-ai-lab/FastAFD `3c716194`](https://github.com/hao-ai-lab/FastAFD/commit/3c7161949310b6d59d6b4cf9bf997a4935c8113b) | MIT ([LICENSE.FastAFD](afdep/third_party/deepep_v2/LICENSE.FastAFD)) | Copyright (c) 2026 sgl-project, as its license states (FastAFD started from a mini-sglang codebase) | The trimmed DeepEP V2 snapshot in `python/minisgl/kernel/csrc/`, with FastAFD's few local lines in `csrc/elastic/buffer.hpp`, `hybrid_combine.cuh` and `csrc/python_api.cpp` |
| AFD plugin | [ms-llmd/afd-plugin `1198c52`](https://github.com/ms-llmd/afd-plugin/commit/1198c52f8bed0c20a2b51ff550db5ccf2fda0906) | Apache-2.0 | Copyright contributors to the AFD plugin project | The plugin's patches to the snapshot (per-buffer communication stream variable, optional GIN initialization, module-local bindings), and `deepep_m2n/{buffer,layout,jit_build}.py`, which `afdep/deepep.py` and `afdep/_build.py` adapt |

The snapshot did not record its original DeepEP commit; the
[immutable input hashes](afdep/third_party/deepep_v2/PROVENANCE.json) identify it, and
[LOCAL_CHANGES.json](afdep/third_party/deepep_v2/LOCAL_CHANGES.json) lists every later
change with its file hashes. afdEP-comm is an independent project, not affiliated with
or endorsed by DeepSeek, the DeepEP authors or the FastAFD authors.

afdEP v0 (the project's first release) changed role-specific receive bounds, empty-side epilogues,
route validation, and NCCL version diagnostics. afdEP-comm also changes the
role path's synchronization and protocol: it skips the entry barriers, sends
counts from Attention to FFN only, replaces the exit barriers with directed
per-CTA arrival, sizes receive buffers per role, adds a directional hybrid
protocol across NVLink domains, and launches role kernels generated from the
shared ones. Masked DeepEP (`role_aware=False`) keeps the upstream-derived
transport, barriers and credits. Every change and its file hashes are listed
in [LOCAL_CHANGES.json](afdep/third_party/deepep_v2/LOCAL_CHANGES.json).

`afdep/deepep.py` and `afdep/_build.py` adapt the snapshot's Apache-2.0
`deepep_m2n/{buffer,layout,jit_build}.py`. No DeepGEMM, CUTLASS or fmt subtree is
included. PyTorch, CUDA and NCCL are external dependencies under their own licenses.
