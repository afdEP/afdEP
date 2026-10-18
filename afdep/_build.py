# SPDX-License-Identifier: Apache-2.0
"""Content-addressed, process-locked build of the standalone CUDA extension.

Uses PyTorch's CUDAExtension in a subprocess so CUDA/compiler environment
selection cannot change another library's build state in the caller.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import ModuleType

PACKAGE_ROOT = Path(__file__).resolve().parent
MIN_NCCL_VERSION = 23007
CUDA_CAPABILITY = (9, 0)
CUDA_ARCH = "9.0a"
DEFAULT_MAX_JOBS = "4"
NCCL_DISTRIBUTIONS = ("nvidia-nccl-cu13", "nvidia-nccl-cu12")
SOURCE_NAMES = ("bindings.cpp", "nccl_runtime.cpp", "kernels.cu")
CXX_FLAGS = ("-O3", "-std=c++20", "-Wno-sign-compare")
NVCC_FLAGS = (
    "-O3",
    "-std=c++20",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--ptxas-options=--register-usage-level=10",
    "--diag-suppress=39,161,174,177,186,940,3012",
)
SETUP_SCRIPT = """\
import json
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

spec = json.loads(Path(__file__).with_name("spec.json").read_text())
setup(
    name=spec["name"],
    ext_modules=[CUDAExtension(
        name=spec["name"],
        sources=spec["sources"],
        include_dirs=spec["include_dirs"],
        library_dirs=spec["library_dirs"],
        extra_compile_args={"cxx": spec["cxx_flags"], "nvcc": spec["nvcc_flags"]},
        extra_link_args=spec["link_flags"],
    )],
    cmdclass={"build_ext": BuildExtension},
)
"""


def resolve_nvcc() -> Path:
    candidates = (
        os.environ.get("CUDA_NVCC_EXECUTABLE"),
        str(Path(os.environ["CUDA_HOME"]) / "bin/nvcc")
        if "CUDA_HOME" in os.environ
        else None,
        shutil.which("nvcc"),
        "/usr/local/cuda/bin/nvcc",
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate).resolve()
    raise RuntimeError(
        "AFDEP needs a CUDA toolkit at runtime. Set CUDA_HOME or "
        "CUDA_NVCC_EXECUTABLE, or put nvcc on PATH."
    )


def nccl_paths() -> tuple[Path, Path]:
    """Find matching headers/library; never silently mix different installs."""
    roots: list[Path] = []
    if override := os.environ.get("AFDEP_NCCL_ROOT"):
        roots.append(Path(override).expanduser().resolve())
    else:
        for name in NCCL_DISTRIBUTIONS:
            try:
                installed = distribution(name)
            except PackageNotFoundError:
                continue
            roots.append(Path(str(installed.locate_file("nvidia/nccl"))).resolve())
        roots.extend((Path("/usr/local/nccl"), Path("/usr")))
    for root in roots:
        include = root / "include"
        for lib in (root / "lib", root / "lib64", root / "lib/x86_64-linux-gnu"):
            if (include / "nccl.h").is_file() and (lib / "libnccl.so.2").is_file():
                if not (include / "nccl_device.h").is_file():
                    raise RuntimeError(
                        f"NCCL device headers are missing under {include}"
                    )
                return include, lib
    raise RuntimeError(
        "AFDEP needs NCCL >= 2.30.7 including nccl_device.h and libnccl.so.2. "
        "Use a supported NVIDIA NCCL wheel or set AFDEP_NCCL_ROOT."
    )


def cuda_wheel_include_flags() -> list[str]:
    """Supply optional pip CUDA headers that PyTorch CUDAContext includes."""
    roots: set[Path] = set()
    for name in NCCL_DISTRIBUTIONS:
        try:
            installed = distribution(name)
        except PackageNotFoundError:
            continue
        root = Path(str(installed.locate_file("nvidia")))
        roots.update(root.glob("cu*/include"))
        roots.update(root.glob("*/include"))
    return [flag for root in sorted(roots) for flag in ("-isystem", str(root))]


def source_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted((PACKAGE_ROOT / "csrc").glob("*")):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def build_signature(nvcc: Path, include: Path, lib: Path) -> dict:
    import torch

    # The local import keeps package inspection independent of torch loading.
    nvcc_version = subprocess.check_output([str(nvcc), "--version"], text=True)
    header_digest = hashlib.sha256()
    for path in sorted(include.rglob("*.h")):
        header_digest.update(str(path.relative_to(include)).encode())
        header_digest.update(path.read_bytes())
    return {
        "source": source_digest(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
        "python_abi": sysconfig.get_config_var("SOABI"),
        "nvcc": str(nvcc),
        "nvcc_version": nvcc_version,
        "cxx": os.environ.get("CXX", "c++"),
        "nccl_include": str(include),
        "nccl_lib": str(lib.resolve()),
        "nccl_headers": header_digest.hexdigest(),
        "cxx_flags": CXX_FLAGS,
        "nvcc_flags": NVCC_FLAGS,
        "setup_script": hashlib.sha256(SETUP_SCRIPT.encode()).hexdigest(),
        "source_names": SOURCE_NAMES,
        "link_flags": ["-l:libnccl.so.2", f"-Wl,-rpath,{lib}"],
        "arch": CUDA_ARCH,
    }


def load_native() -> ModuleType:
    """Build/load an immutable source/toolchain-specific native extension."""
    import fcntl

    import torch

    if platform.system() != "Linux" or not torch.cuda.is_available():
        raise RuntimeError("The current CUDA backend requires Linux with CUDA GPUs")
    if torch.cuda.get_device_capability() != CUDA_CAPABILITY:
        raise RuntimeError("The current CUDA backend requires SM90")
    nvcc = resolve_nvcc()
    include, lib = nccl_paths()
    signature = build_signature(nvcc, include, lib)
    fingerprint = hashlib.sha256(
        json.dumps(signature, sort_keys=True).encode()
    ).hexdigest()
    name = f"_afdep_{fingerprint[:16]}"
    if name in sys.modules:
        return sys.modules[name]
    cache_root = Path(
        os.environ.get("AFDEP_CACHE_DIR", Path.home() / ".cache/afdep")
    ).expanduser()
    build_dir = cache_root / fingerprint
    build_dir.mkdir(parents=True, exist_ok=True)
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    so_path = build_dir / "lib" / f"{name}{extension_suffix}"
    complete = build_dir / "build.complete"
    with (build_dir / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (
            not so_path.is_file()
            or not complete.is_file()
            or complete.read_text() != fingerprint
        ):
            system_headers = cuda_wheel_include_flags()
            spec = {
                "name": name,
                "sources": [str(PACKAGE_ROOT / "csrc" / p) for p in SOURCE_NAMES],
                "include_dirs": [str(PACKAGE_ROOT / "csrc"), str(include)],
                "library_dirs": [str(lib)],
                "cxx_flags": [*CXX_FLAGS, *system_headers],
                "nvcc_flags": [*NVCC_FLAGS, *system_headers],
                "link_flags": signature["link_flags"],
            }
            (build_dir / "spec.json").write_text(json.dumps(spec, indent=2))
            (build_dir / "signature.json").write_text(json.dumps(signature, indent=2))
            setup = build_dir / "setup.py"
            setup.write_text(SETUP_SCRIPT)
            env = dict(os.environ)
            for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"):
                env.pop(key, None)
            env["CUDA_HOME"] = str(nvcc.parent.parent)
            env["PYTORCH_NVCC"] = str(nvcc)
            env["TORCH_CUDA_ARCH_LIST"] = CUDA_ARCH
            env.setdefault("MAX_JOBS", DEFAULT_MAX_JOBS)
            subprocess.run(
                [
                    sys.executable,
                    str(setup),
                    "build_ext",
                    "--build-lib",
                    str(build_dir / "staging"),
                    "--build-temp",
                    str(build_dir / "temp"),
                ],
                cwd=build_dir,
                env=env,
                check=True,
            )
            artifact = build_dir / "staging" / so_path.name
            if not artifact.is_file():
                raise RuntimeError(f"Native build did not produce {artifact}")
            so_path.parent.mkdir(exist_ok=True)
            # Another process never observes a partially linked extension.
            os.replace(artifact, so_path)
            complete.write_text(fingerprint)
        fcntl.flock(lock, fcntl.LOCK_UN)
    module_spec = importlib.util.spec_from_file_location(name, so_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Could not load AFDEP extension at {so_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    sys.modules[name] = module
    return module
