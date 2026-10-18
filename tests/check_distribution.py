# SPDX-License-Identifier: Apache-2.0
"""Fail if a built wheel omits native runtime sources or third-party notices."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

REQUIRED = {
    "afdep/__init__.py",
    "afdep/buffer.py",
    "afdep/config.py",
    "afdep/csrc/bindings.cpp",
    "afdep/csrc/buffer.hpp",
    "afdep/csrc/kernels.cu",
    "afdep/csrc/kernels.cuh",
    "afdep/csrc/kernels.hpp",
    "afdep/csrc/layout.hpp",
    "afdep/csrc/nccl_runtime.cpp",
    "afdep/csrc/nccl_runtime.hpp",
    "afdep/csrc/ORIGIN.json",
    "afdep/third_party/deepep/LICENSE",
    "afdep/third_party/deepep/PROVENANCE.json",
}


def main() -> None:
    wheels = list(Path("dist").glob("*.whl"))
    sdists = list(Path("dist").glob("*.tar.gz"))
    assert wheels and sdists, "Build both wheel and sdist first"
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            names = set(archive.namelist())
            assert REQUIRED <= names, REQUIRED - names
            assert any(name.endswith("/licenses/LICENSE") for name in names)
            assert any(
                name.endswith("/licenses/THIRD_PARTY_NOTICES.md") for name in names
            )
    for sdist in sdists:
        with tarfile.open(sdist) as archive:
            names = {name.split("/", 1)[-1] for name in archive.getnames()}
            assert REQUIRED <= names, REQUIRED - names
    print("Wheel and sdist contain the standalone runtime and licenses")


if __name__ == "__main__":
    main()
