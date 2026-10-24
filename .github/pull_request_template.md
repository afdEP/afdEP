## What and why

## Checks

- [ ] `python -m pytest tests/unit`, `ruff check .` and `ruff format --check .` pass
- [ ] Native changes: the GPU suite on the affected backend (BF16 and FP8, eager and graphs, empty and skewed routes, delayed peers, buffer reuse)
- [ ] Performance claims: paired fresh-process screens against Masked DeepEP and the change's own control switch, with each arm's native source digest
- [ ] Records and docs/design.md name the exact native source digest measured
- [ ] Optimizations are AFD-specific and keep a control switch (docs/design.md)
- [ ] Communication-only results are labeled as such; no serving claims without serving evidence

## Evidence
