# Security policy

Report vulnerabilities privately through
[GitHub security advisories](https://github.com/afdEP/afdEP/security/advisories/new),
not in public issues. Include the affected release or commit, the output of
`python -m afdep doctor`, and a reproduction.

afdEP compiles and loads native code at runtime (cached under `AFDEP_CACHE_DIR`) and
exchanges data through registered GPU memory and NCCL GIN. Every rank in a group is
trusted: afdEP validates expert IDs before route-derived indexing, but does not
authenticate peers.

Fixes go into the latest release.
