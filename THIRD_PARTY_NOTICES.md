# Third-party notices

afdEP's Python adapters are Apache-2.0. The vendored DeepEP V2 native transport
and its afdEP role specializations are MIT-licensed.

- Upstream: [DeepEP](https://github.com/deepseek-ai/DeepEP), Copyright (c) 2025 DeepSeek.
- Snapshot: [FastAFD `3c716194`](https://github.com/hao-ai-lab/FastAFD/commit/3c7161949310b6d59d6b4cf9bf997a4935c8113b),
  retained through [AFD plugin `1198c52`](https://github.com/ms-llmd/afd-plugin/commit/1198c52f8bed0c20a2b51ff550db5ccf2fda0906).
  That snapshot did not record its original DeepEP commit.
- [MIT license](afdep/third_party/deepep_v2/LICENSE),
  [immutable input hashes](afdep/third_party/deepep_v2/PROVENANCE.json), and
  [local changes and hashes](afdep/third_party/deepep_v2/LOCAL_CHANGES.json).

The snapshot includes AFD plugin patches for optional GIN and module-local
bindings. afdEP v0 changes role-specific receive bounds, empty-side epilogues,
route validation, and NCCL version diagnostics. afdEP v1 also changes the
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
