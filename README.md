# AFDEP

AFDEP is a GPU communication library for attention–FFN disaggregation. Attention
GPUs send routed tokens to a separate FFN GPU domain; the FFN GPUs return one
reduced result per source token. Model execution, expert weights, routing policy,
and serving control remain in the caller.

The initial implementation provides a directional request/reply protocol over
NCCL GPU-Initiated Networking (GIN) and local NVLink peer access. It does not call
DeepEP dispatch/combine, create dummy experts, or require the DeepEP or vLLM
Python packages. Minimal upstream transport helpers are vendored with their
attribution and licenses; AFDEP owns its extension build and bootstrap wrapper.

**Status:** experimental extraction of the Hopper directional prototype. That
prototype passed eight-GPU correctness and communication benchmarks on October
15, 2026. Those results are historical evidence for the protocol, not validation
or performance measurements of every subsequent AFDEP build. See
[performance and validation](docs/performance.md) for the distinction.

## Supported configuration

The first backend supports one specific placement:

| Process-group ranks | Role | Placement |
|---|---|---|
| 0–3 | FFN workers | One four-GPU NVLink domain |
| 4–7 | Attention sources | A second four-GPU NVLink domain |

It targets Hopper SM90, one in-flight request/reply lane, BF16 activations or
prequantized FP8 E4M3 activations with FP32 block-128 scales, and BF16 replies.
The rank numbers above are relative to the supplied process group. Four workers
own contiguous, equal-sized ranges of real expert IDs. All eight ranks must
participate in every exchange, including ranks with no tokens or expert work.

Arbitrary N/M ratios, mixed-role domains, additional domains, concurrent lanes,
Blackwell, and other GPU architectures are not supported by this first backend.
The library name describes its intended scope; it is not a claim that those
topologies already work.

## Installation

GPU execution requires Linux, Hopper SM90 GPUs, PyTorch with CUDA 13, a compatible
CUDA 13 `nvcc` compiler, and NCCL 2.30.7 or later with device GIN headers and a
working native RDMA backend. A CPU Gloo process group bootstraps the native
communicator. AFDEP uses PyTorch's C++ extension tooling to compile its standalone
extension on first use; it does not use DeepEP's JIT. The CUDA compiler and
development headers must therefore remain available at runtime. Installing the
Python package, importing it, and testing configuration do not require a GPU.

From a source checkout, install into an environment with the required PyTorch
and CUDA/NCCL development dependencies already installed:

```bash
python -m pip install --no-build-isolation -e .
python -m afdep doctor
# On a supported GPU host, also compile and load the extension:
python -m afdep doctor --build
```

Install `torch`, `setuptools`, and `ninja` in the execution environment. Set
`CUDA_HOME` or `CUDA_NVCC_EXECUTABLE` if compiler discovery needs an override.
`AFDEP_NCCL_ROOT` selects NCCL headers/libraries; otherwise AFDEP looks for the
installed NVIDIA NCCL Python distribution. `AFDEP_CACHE_DIR` overrides the
extension build cache location.

The loaded NCCL runtime version must match the headers used to compile the
extension. An override cannot safely mix a different NCCL version with the one
already loaded by PyTorch; AFDEP rejects that mismatch.
NCCL's [GIN compatibility rules](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/usage/deviceapi.html#cross-version-compatibility)
require rebuilding device kernels when changing NCCL versions; AFDEP includes
the installed NCCL headers in its build-cache fingerprint.

Use the repository's validation and benchmark entry points before relying on a
new toolchain or network setup. The historical tested software configuration
and limitations are listed in [docs/performance.md](docs/performance.md).

Run the following from the source checkout on both four-GPU hosts, with
`NODE_RANK=0` on the FFN host and `NODE_RANK=1` on the Attention host. Set
`MASTER_ADDR` to the rendezvous address and configure network/NIC affinity for
the selected RDMA rail before launch.

```bash
torchrun --nnodes=2 --nproc-per-node=4 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  --module tests.gpu.correctness --payload bf16
```

Repeat with `--payload fp8`. To run the communication-only CUDA-graph benchmark,
replace the module and its options with:

```text
--module benchmarks.round_trip --payload bf16 --graph --iterations 100
```

Use `--nvtx` in a separate profiling run for annotated Nsight Systems traces.
Tracing changes timing; do not mix profiled samples into latency results. The
[round-trip example](examples/round_trip.py) demonstrates caller-owned weighted
expert computation without a model-serving framework.

## API

Create one buffer on every rank after selecting that process's CUDA device.
Construction and `close()` are collective lifecycle operations on the supplied
CPU Gloo group. Exchange methods enqueue work on the current PyTorch CUDA stream.

```python
import afdep

config = afdep.Config(
    capacity_tokens_per_rank=8192,
    hidden_size=2048,
    num_experts=128,
    top_k=8,
    payload="bf16",
)
buffer = afdep.Buffer(group=group, config=config)
```

For each exchange, all ranks agree on a positive `bucket` no greater than the
configured capacity. Each source may have between zero and `bucket` tokens.
The roles then follow separate APIs:

```python
# Attention source, on each rank 4–7:
buffer.source_dispatch(x, expert_ids, topk_weights, bucket=bucket)
buffer.source_collect(out)

# FFN worker, on each rank 0–3:
received = buffer.worker_receive(bucket=bucket, read_host_counts=False)
weighted_rows = run_local_experts(received)  # Caller-owned computation.
buffer.worker_reply(weighted_rows, received.receipt)
```

`x` is BF16 `[tokens, hidden_size]`, `expert_ids` is INT64 `[tokens, top_k]`,
`topk_weights` is FP32 with the same shape, and `out` is BF16 with the same shape
as `x`. FP8 dispatch uses `payload="fp8"`, FP8 E4M3 values, and
`x_scale=...` containing FP32 scales of shape `[tokens, hidden_size // 128]`.
AFDEP transports those values; it does not quantize them.

**Apply routing weights exactly once in the expert computation before
`worker_reply`.** AFDEP transports the weights in `recv_topk_weights` and sums
the supplied expert rows. It does not multiply by those weights during reply
reduction. `run_local_experts` above is an application placeholder, not an AFDEP
function.

The receive tensors include conservative padding. Use the device expert
offsets/count metadata when scheduling GPU expert work; the physical tensor
length is not the number of valid rows. Retain the opaque receipt unchanged for
the matching reply. See [the API contract](docs/api.md) for tensor layouts,
ownership, stream ordering, CUDA graphs, and shutdown.

```python
# After every rank has completed its final exchange:
buffer.close()
```

## Design and evidence

- [Architecture](docs/architecture.md): library boundaries, roles, and transport.
- [Protocol](docs/protocol.md): epochs, completion signals, and reader credits.
- [API contract](docs/api.md): shapes, weighted results, streams, and graphs.
- [Performance](docs/performance.md): inherited measurements and open questions.
- [Contributing](CONTRIBUTING.md): changes and validation expectations.

AFDEP is licensed under Apache-2.0. Upstream-derived components retain their
original notices; consult the repository's third-party attribution alongside
the source. The directional protocol originated in the AFD plugin experiments
comparing a dedicated transport against
[AFD plugin PR #23](https://github.com/ms-llmd/afd-plugin/pull/23).
