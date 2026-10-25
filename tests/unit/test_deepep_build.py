# SPDX-License-Identifier: Apache-2.0
"""Check vendored transport integrity and lazy, content-addressed build inputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from fractions import Fraction

from afdep._build import SOURCE_NAMES, VENDOR_ROOT, source_digest


def test_pinned_origin_and_declared_role_specialization():
    provenance = json.loads((VENDOR_ROOT / "PROVENANCE.json").read_text())
    assert provenance["source_commit"] == "1198c52f8bed0c20a2b51ff550db5ccf2fda0906"
    assert provenance["local_changes"]
    changes = json.loads((VENDOR_ROOT / "LOCAL_CHANGES.json").read_text())
    expected = provenance["file_sha256"].copy()
    for name, entry in changes["modified_files"].items():
        assert entry["upstream_sha256"] == expected[name]
        expected[name] = entry["sha256"]
    expected.update(
        {name: entry["sha256"] for name, entry in changes["added_files"].items()}
    )
    actual = {
        path.relative_to(VENDOR_ROOT).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in VENDOR_ROOT.rglob("*")
        if path.is_file() and path.name not in {"PROVENANCE.json", "LOCAL_CHANGES.json"}
    }
    assert actual == expected
    assert all(name in expected for name in SOURCE_NAMES)
    assert "MIT License" in (VENDOR_ROOT / "LICENSE").read_text()
    # Direct and hybrid dispatch/combine add opt-in role specializations whose
    # defaults keep upstream kernels (see the template-default test). The
    # credit/barrier kernel remains the pinned input, independently reviewable here.
    path = "deep_ep/include/deep_ep/impls/barrier.cuh"
    assert expected[path] == provenance["file_sha256"][path]


def test_role_template_parameters_default_to_upstream_kernels():
    """Unspecialized (baseline) instantiations keep the upstream kernels."""
    impls = VENDOR_ROOT / "deep_ep/include/deep_ep/impls"
    combine = (impls / "combine.cuh").read_text()
    assert "bool kAfdSkipEntryBarrier = false," in combine
    assert "int kAfdDirectedFfnRanks = 0," in combine
    assert "bool kAfdPerCtaExit = false," in combine
    assert "bool kAfdSingleFfnReduce = false," in combine
    dispatch = (impls / "dispatch.cuh").read_text()
    assert "int kAfdFfnRanks = 0," in dispatch
    assert "bool kAfdSkipEntryBarrier = false," in dispatch
    assert "bool kAfdDirectionalNotify = false," in dispatch
    assert "bool kAfdDirectedExit = false," in dispatch
    assert "bool kAfdRoleBuffers = false," in dispatch
    assert "bool kAfdPerCtaExit = false," in dispatch
    assert "int kAfdZeroCollectRowBytes = 0," in dispatch
    epilogue = (impls / "dispatch_copy_epilogue.cuh").read_text()
    assert "int kAfdRegionOffset = 0," in epilogue
    for hybrid in ("hybrid_dispatch.cuh", "hybrid_combine.cuh"):
        assert "int kAfdFfnScaleoutRanks = 0," in (impls / hybrid).read_text()
    hybrid_dispatch = (impls / "hybrid_dispatch.cuh").read_text()
    assert "bool kAfdDirectionalRoutes = false," in hybrid_dispatch
    assert "bool kAfdSkipAttentionEntrySync = false," in hybrid_dispatch


# Device sources before the role suites (native digest 8205d02b). The role suites
# size launches of these kernels and add separate role kernels; they never edit a
# shared kernel, so the baseline and every shared instantiation keep their code, include
# hash and cubin.
SHARED_KERNEL_SHA256 = {
    "dispatch.cuh": (
        "81e189fab1985fba3f8c772ecfe0b2efbfe4618744c0ae0195566ae20326ce6a"
    ),
    "combine.cuh": "8847dfb9b41cc0cb0c971ac0c35ca25d95cacc2443413ca3a6d6b7dcb43a51e1",
    "hybrid_dispatch.cuh": (
        "56ebe2ee6854380fa575881bf53e4a8d9392f7e4bf3ad1510c57b65854d1c313"
    ),
    "hybrid_combine.cuh": (
        "a6fd0f17475bf1b1783a9671cbe09218e982ae904ad86ff2e6aa12efe2c9ee8a"
    ),
    "dispatch_copy_epilogue.cuh": (
        "0b573abeb0dc2bb8d4c65d9a3ce28f82537e129d3dc4cf5e3709dd4e9cbf85cf"
    ),
    "afdep_roles.cuh": (
        "4f7ed0d81d7215753c99dfe1364e4ba6e707101cb5ede0d1b139e8b1923e937a"
    ),
}


def test_role_suites_leave_shared_kernels_untouched():
    provenance = json.loads((VENDOR_ROOT / "PROVENANCE.json").read_text())
    include = VENDOR_ROOT / "deep_ep/include/deep_ep"

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    for name, sha in SHARED_KERNEL_SHA256.items():
        assert digest(include / "impls" / name) == sha, name
    upstream = [
        include / "impls/combine_reduce_epilogue.cuh",
        *sorted((include / "common").glob("*")),
    ]
    for path in upstream:
        name = path.relative_to(VENDOR_ROOT).as_posix()
        assert digest(path) == provenance["file_sha256"][name], name


def test_role_suite_shapes_are_host_launch_overrides():
    """The Attention hybrid send block relaunches the shared cubin, generated first."""
    elastic = VENDOR_ROOT / "csrc/kernels/elastic"
    suites = (elastic / "afdep_suites.hpp").read_text()
    controls = ("ROLE_KERNELS", "FFN_RECEIVE", "HYBRID_SEND", "HYBRID_FORWARD")
    for control in (*controls, "HYBRID_COLLECT"):
        assert f'"AFDEP_SHARED_{control}", 0) == 0' in suites
    # Count-first dispatch keeps the payload-first kernels as its control.
    assert '"AFDEP_PAYLOAD_FIRST", 0) == 0' in suites
    dispatch = (elastic / "dispatch.hpp").read_text()
    generate = dispatch.index("const auto code = DispatchRuntime::generate(args);")
    assert generate < dispatch.index("shaped_args.launch_args = jit::LaunchArgs(")


def test_ffn_role_kernels_are_the_generated_copies():
    from tests import role_kernels as rk

    hybrid = (rk.IMPLS / "hybrid_dispatch.cuh").read_text()
    assert (rk.IMPLS / rk.FFN_HYBRID_DISPATCH).read_text() == rk.ffn_hybrid_dispatch(
        hybrid
    )
    source_hybrid = (rk.IMPLS / rk.SOURCE_HYBRID_DISPATCH).read_text()
    assert source_hybrid == rk.source_hybrid_dispatch(hybrid)
    ffn_sparse = (rk.IMPLS / rk.FFN_SPARSE_HYBRID_DISPATCH).read_text()
    assert ffn_sparse == rk.ffn_sparse_hybrid_dispatch(hybrid)
    combine = (rk.IMPLS / "hybrid_combine.cuh").read_text()
    ffn_combine = (rk.IMPLS / rk.FFN_SPARSE_HYBRID_COMBINE).read_text()
    assert ffn_combine == rk.ffn_sparse_hybrid_combine(combine)
    collect = (rk.IMPLS / "afdep_hybrid_collect.cuh").read_text()
    sparse_collect = (rk.IMPLS / rk.SPARSE_HYBRID_COLLECT).read_text()
    assert sparse_collect == rk.sparse_hybrid_collect(collect)
    mixed_dispatch = (rk.IMPLS / rk.MIXED_HYBRID_DISPATCH).read_text()
    assert mixed_dispatch == rk.mixed_hybrid_dispatch(hybrid)
    mixed_combine = (rk.IMPLS / rk.MIXED_HYBRID_COMBINE).read_text()
    assert mixed_combine == rk.mixed_hybrid_combine(combine)
    fp8_combine = (rk.IMPLS / rk.FFN_FP8_HYBRID_COMBINE).read_text()
    assert fp8_combine == rk.ffn_fp8_hybrid_combine(combine)
    sparse_fp8 = (rk.IMPLS / rk.FFN_SPARSE_FP8_HYBRID_COMBINE).read_text()
    assert sparse_fp8 == rk.ffn_sparse_fp8_hybrid_combine(combine)
    utils = (rk.IMPLS / "combine_utils.cuh").read_text()
    assert (rk.IMPLS / rk.FP8_COMBINE_REDUCE).read_text() == rk.fp8_combine_reduce(
        utils
    )
    reduce_epilogue = (rk.IMPLS / "combine_reduce_epilogue.cuh").read_text()
    fp8_epilogue = (rk.IMPLS / rk.FP8_COMBINE_EPILOGUE).read_text()
    assert fp8_epilogue == rk.fp8_combine_reduce_epilogue(reduce_epilogue)
    direct = (rk.IMPLS / "dispatch.cuh").read_text()
    assert (rk.IMPLS / rk.FFN_DISPATCH).read_text() == rk.ffn_dispatch(direct)
    assert (rk.IMPLS / rk.SOURCE_DISPATCH).read_text() == rk.source_dispatch(direct)
    stream = (rk.IMPLS / rk.FFN_STREAM_DISPATCH).read_text()
    assert stream == rk.ffn_stream_dispatch(direct)
    epilogue = (rk.IMPLS / "dispatch_copy_epilogue.cuh").read_text()
    assert (rk.IMPLS / rk.FFN_STREAM_EPILOGUE).read_text() == rk.ffn_stream_epilogue(
        epilogue
    )


def test_count_first_kernels_stay_out_of_shared_runtimes():
    """Each count-first copy has one runtime; no shared runtime includes the header."""
    from tests import role_kernels as rk

    dispatch = (VENDOR_ROOT / "csrc/kernels/elastic/dispatch.hpp").read_text()
    for name in (rk.SOURCE_DISPATCH, rk.FFN_STREAM_DISPATCH, rk.FFN_STREAM_EPILOGUE):
        assert dispatch.count(f"#include <deep_ep/impls/{name}>") == 1, name
    assert "afdep_count_first.cuh" not in dispatch
    for name in (
        "dispatch.cuh",
        "dispatch_copy_epilogue.cuh",
        "afdep_ffn_dispatch.cuh",
    ):
        assert "afdep_count_first.cuh" not in (rk.IMPLS / name).read_text(), name


def test_sparse_hybrid_kernels_stay_out_of_shared_runtimes():
    """Only sparse copies include the sparse header, each via one runtime."""
    from tests import role_kernels as rk

    dispatch = (VENDOR_ROOT / "csrc/kernels/elastic/dispatch.hpp").read_text()
    for name in (rk.SOURCE_HYBRID_DISPATCH, rk.FFN_SPARSE_HYBRID_DISPATCH):
        assert dispatch.count(f"#include <deep_ep/impls/{name}>") == 1, name
    combine = (VENDOR_ROOT / "csrc/kernels/elastic/combine.hpp").read_text()
    for name in (rk.FFN_SPARSE_HYBRID_COMBINE, rk.SPARSE_HYBRID_COLLECT):
        assert combine.count(f'"{name.removesuffix(".cuh")}");') == 1, name
    # The mixed-role copies: one runtime each, neither including the sparse header.
    mixed = rk.MIXED_HYBRID_DISPATCH
    assert dispatch.count(f"#include <deep_ep/impls/{mixed}>") == 1
    assert combine.count(f'"{rk.MIXED_HYBRID_COMBINE.removesuffix(".cuh")}");') == 1
    header = "#include <deep_ep/impls/afdep_sparse_hybrid.cuh>"
    assert header not in dispatch and header not in combine
    shared = ("hybrid_dispatch.cuh", "hybrid_combine.cuh", "afdep_hybrid_collect.cuh")
    fp8 = (rk.FFN_FP8_HYBRID_COMBINE, rk.FP8_COMBINE_EPILOGUE, rk.FP8_COMBINE_REDUCE)
    for name in (*shared, rk.FFN_HYBRID_DISPATCH, mixed, rk.MIXED_HYBRID_COMBINE, *fp8):
        assert header not in (rk.IMPLS / name).read_text(), name


def test_fp8_combine_kernels_stay_out_of_shared_runtimes():
    """Each FP8 combine copy has one runtime; no shared kernel includes FP8 headers."""
    from tests import role_kernels as rk

    elastic = VENDOR_ROOT / "csrc/kernels/elastic"
    combine = (elastic / "combine.hpp").read_text()
    dispatch = (elastic / "dispatch.hpp").read_text()
    copies = (
        rk.FFN_FP8_HYBRID_COMBINE,
        rk.FFN_SPARSE_FP8_HYBRID_COMBINE,
        rk.FP8_COMBINE_EPILOGUE,
    )
    for name in copies:
        assert combine.count(f'"{name.removesuffix(".cuh")}");') == 1, name
    headers = [
        f"#include <deep_ep/impls/{name}>"
        for name in ("afdep_fp8_combine.cuh", rk.FP8_COMBINE_REDUCE)
    ]
    shared = (
        "hybrid_combine.cuh",
        "combine_reduce_epilogue.cuh",
        "combine_utils.cuh",
        "afdep_hybrid_collect.cuh",
        rk.SPARSE_HYBRID_COLLECT,
        rk.FFN_SPARSE_HYBRID_COMBINE,
        rk.MIXED_HYBRID_COMBINE,
    )
    for header in headers:
        assert header not in combine and header not in dispatch, header
        for name in shared:
            assert header not in (rk.IMPLS / name).read_text(), name
    # Each copy includes its FP8 header once; the sparse copy keeps its sparse edits.
    reduce_header, fp8_header = headers[1], headers[0]
    for name in copies[:2]:
        assert (rk.IMPLS / name).read_text().count(reduce_header) == 1, name
    epilogue = (rk.IMPLS / rk.FP8_COMBINE_EPILOGUE).read_text()
    assert epilogue.count(fp8_header) == 1
    assert (rk.IMPLS / rk.FP8_COMBINE_REDUCE).read_text().count(fp8_header) == 1
    sparse_fp8 = (rk.IMPLS / rk.FFN_SPARSE_FP8_HYBRID_COMBINE).read_text()
    sparse_header = "#include <deep_ep/impls/afdep_sparse_hybrid.cuh>"
    assert sparse_fp8.count(sparse_header) == 1
    assert "afdep_sparse_done_ptr" in sparse_fp8
    # Scale-up partials stay BF16: only the scale-out reply rows use the FP8 layout.
    for name in copies[:2]:
        text = (rk.IMPLS / name).read_text()
        assert text.count("afd_fp8_layout") == 6, name
        assert "token_layout, kNumTokensInScaleupLayout" in text, name


def test_fp8_combine_switch_is_latched_and_agreement_checked():
    """The native buffer latches the switch once and reports it in the hybrid tuple."""
    elastic = VENDOR_ROOT / "csrc/kernels/elastic"
    suites = (elastic / "afdep_suites.hpp").read_text()
    assert (
        "return afdep_role_suites() and "
        'deep_ep::get_env<int>("AFDEP_HYBRID_FP8_COMBINE", 0) != 0;'
    ) in suites
    buffer = (VENDOR_ROOT / "csrc/elastic/buffer.hpp").read_text()
    # Read once, in the constructor latch, gated on the directional hybrid.
    assert buffer.count("afdep_hybrid_fp8_combine()") == 1
    latch = buffer[buffer.index("afd_fp8_combine = afd_ffn_ranks > 0") :]
    latch = latch[: latch.index(";")]
    for condition in (
        "not deterministic",
        "allow_multiple_reduction",
        "kEnableSM90Features",
        "afdep_directional_hybrid(",
    ):
        assert condition in latch, condition
    assert buffer.index("afd_fp8_combine = afd_ffn_ranks > 0") < buffer.index(
        "// Timeout\n"
    )
    assert (
        "std::tuple<int, bool, bool, int, bool, int> get_afd_sparse_hybrid_config()"
        in (buffer)
    )
    assert "return {0, false, false, mixed, false, 0};" in buffer
    assert "return {0, false, false, 0, false, 0};" in buffer
    # The minimum bucket is latched with the switch (0 when it is off, so ranks that
    # differ only in an unused minimum still agree), clamped at 0, reported next to it.
    assert (
        "afd_fp8_min_bucket = afd_fp8_combine ? "
        "afdep_hybrid_fp8_combine_min_bucket() : 0;"
    ) in buffer
    assert buffer.count("afdep_hybrid_fp8_combine_min_bucket()") == 1
    assert buffer.index("afd_fp8_min_bucket = afd_fp8_combine") < buffer.index(
        "// Timeout\n"
    )
    assert (
        'std::max(deep_ep::get_env<int>("AFDEP_HYBRID_FP8_COMBINE_MIN_BUCKET", 0), 0)'
    ) in suites
    # The layout assert holds at every bucket, not only the FP8 ones.
    assert buffer.index("EP_HOST_ASSERT(not afd_fp8_combine or") < buffer.index(
        "const bool afd_fp8_replies"
    )
    assert "afd_fp8_combine, afd_fp8_min_bucket};" in buffer
    # combine() passes the latched value, gated by this exchange's bucket, to both
    # launches; the kernels never re-read it.
    assert (
        "const bool afd_fp8_replies = afd_fp8_combine and "
        "num_max_tokens_per_rank >= afd_fp8_min_bucket;"
    ) in buffer
    assert "afd_mixed_ffn_ranks,\n            afd_fp8_replies);" in buffer
    assert "comm_stream, afd_fp8_replies);" in buffer
    combine = (elastic / "combine.hpp").read_text()
    assert "afdep_hybrid_fp8_combine" not in combine


def test_fp8_combine_exponent_rule():
    """Every positive finite BF16 amax gets the smallest e with amax <= 448 * 2^e."""
    header = (
        VENDOR_ROOT / "deep_ep/include/deep_ep/impls/afdep_fp8_combine.cuh"
    ).read_text()

    def constant(name):
        match = re.search(rf"\b{name} = (-?(?:0x)?[0-9a-fA-F]+);", header)
        return int(match.group(1), 0)

    bias, threshold = (
        constant("kAfdFp8ExponentBias"),
        constant("kAfdFp8MantissaThreshold"),
    )
    min_exponent, group = constant("kAfdFp8MinExponent"), constant("kAfdFp8GroupSize")
    assert group == 128
    e4m3_max, e4m3_min_subnormal = Fraction(448), Fraction(1, 2**9)

    def exponent(bf16_bits):
        bits = bf16_bits << 16  # the FP32 value the kernel's amax holds
        e = ((bits >> 23) & 0xFF) - bias + int((bits & 0x7FFFFF) > threshold)
        return max(e, min_exponent)

    def value(bf16_bits):
        field, mantissa = bf16_bits >> 7, bf16_bits & 0x7F
        if field == 0:
            return Fraction(mantissa, 2**133)
        return Fraction(128 + mantissa, 128) * Fraction(2) ** (field - 127)

    for bf16_bits in range(1, 0x7F80):  # 32,639 positive finite BF16 values
        amax, e = value(bf16_bits), exponent(bf16_bits)
        assert amax <= e4m3_max * Fraction(2) ** e, bf16_bits
        assert e == min_exponent or amax > e4m3_max * Fraction(2) ** (e - 1), bf16_bits
        # The UE8M0 byte and both powers of two are FP32 normals.
        assert 1 <= 127 + e <= 254 and 1 <= 127 - e <= 254, bf16_bits
        # The largest decoded magnitude is finite below the top scale; at 2^120 it
        # is not, as the header and docs/design.md section 19 state.
        assert e == 120 or e4m3_max * Fraction(2) ** e < Fraction(2) ** 128, bf16_bits
    assert exponent(0x7F7F) == 120  # the largest finite BF16
    assert exponent(0x7F80) == 120  # an infinity keeps the same, finite scale
    # Every nonzero dequantized value is an FP32 normal, so products are exact.
    assert e4m3_min_subnormal * Fraction(2) ** min_exponent >= Fraction(1, 2**126)


def test_count_first_host_flag_geometry_mirrors_the_header():
    """The host's flag bound uses the device header's wire constants."""

    def constant(text, name):
        return int(re.search(rf"\b{name} = (\d+);", text).group(1))

    impls = VENDOR_ROOT / "deep_ep/include/deep_ep/impls"
    header = (impls / "afdep_count_first.cuh").read_text()
    suites = (VENDOR_ROOT / "csrc/kernels/elastic/afdep_suites.hpp").read_text()
    assert "kAfdCfFlags = kAfdCfBase + 128;" in header
    flags = constant(header, "kAfdCfBase") + 128
    assert constant(suites, "kAfdepCountFirstFlagBase") == flags
    for host, device in (
        ("kAfdepCountFirstFlagStride", "kAfdepFlagStride"),
        ("kAfdepCountFirstMaxSources", "kAfdCfMaxSources"),
        ("kAfdepCountFirstMaxFfnRanks", "kAfdCfMaxFfnRanks"),
    ):
        assert constant(suites, host) == constant(header, device), host
    # kNumMaxRanks x kNumMaxChannels int64 words (1024 x 1024).
    assert "kAfdepCountFirstRegionWords = 1 << 20;" in suites


def test_runtime_header_contents_invalidate_build(tmp_path):
    header = tmp_path / "include/deep_ep/impls/dispatch.cuh"
    header.parent.mkdir(parents=True)
    header.write_text("first")
    os.utime(header, ns=(1_000_000_000, 1_000_000_000))
    first = source_digest(tmp_path)
    header.write_text("other")
    os.utime(header, ns=(1_000_000_000, 1_000_000_000))
    assert source_digest(tmp_path) != first


def test_loader_inspection_does_not_import_runtime_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import afdep; import afdep._build; "
            "api = (afdep.Config, afdep.Buffer); "
            "unexpected = {'torch', 'vllm', 'deep_gemm', 'afd_plugin', 'deep_ep'}; "
            "assert not unexpected & sys.modules.keys()",
        ],
        check=True,
    )
