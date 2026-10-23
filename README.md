<div align="center">

<img src="docs/assets/afdep-logo.png" alt="afdEP" width="60%"/>

<p><b>Expert-parallel communication for Attention–FFN Disaggregation (AFD).</b></p>

</div>

afdEP connects N Attention GPUs, which route tokens, with M FFN GPUs, which hold the
experts: it dispatches routed tokens to the experts and combines their outputs back at
the source. It vendors DeepEP V2's kernels and specializes them for the two roles.
Routing, quantization, expert compute and serving scheduling stay with the caller.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/communication-dark.svg">
  <img alt="Masked DeepEP and afdEP v1: who sends, receives and waits in each phase" src="docs/assets/communication.svg">
</picture>

## What v1 changes

The baseline is **Masked DeepEP**: the same vendored DeepEP V2 kernels with
`role_aware=False`. Every rank then joins a union world, Attention ranks hold dummy
expert slots, and every rank sends, receives and waits in both phases. v1
(`role_aware=True`, the default) keeps the transport and removes only what the roles
make unnecessary; a change that would speed up Masked DeepEP as well is out of scope.
Every change keeps a control switch that restores the Masked behaviour.

In one NVLink domain:
- no entry barriers [§1](docs/v1.md#1-entry-barriers); counts flow Attention → FFN only [§2](docs/v1.md#2-directional-count-exchange);
  only receivers wait at exit, per sending CTA [§3](docs/v1.md#3-directed-exit);
- receive buffers sized per role, 50% of the union registration at 2A2F and about 75%
  at 3:1 and 1:3 [§4](docs/v1.md#4-role-sized-registered-buffers);
- each role launches only the warps it uses, and the FFN starts its copy epilogue at the
  counts [§14](docs/v1.md#14-role-suites-launch-only-what-each-role-uses), [§15](docs/v1.md#15-ffn-receive-copy-epilogue-trigger-at-the-counts-one-nvlink-domain);
- with one FFN rank, a single-FFN reduction and a zero-copy collect view
  [§10](docs/v1.md#10-single-ffn-reduction), [§13](docs/v1.md#13-single-ffn-collect-view).

Across NVLink domains joined by NCCL GPU-Initiated Networking (GIN):
- a directional protocol when every domain holds one role: no cross-domain barriers,
  and counts, payloads, finish tails and combine signals each flow one way
  [§7](docs/v1.md#7-directional-hybrid-protocol), with route checks in the kernel
  [§8](docs/v1.md#8-route-checks-inside-the-directional-hybrid-dispatch) and staging kept out of remotely written inboxes
  [§11](docs/v1.md#11-staging-outside-remotely-written-inboxes-fix);
- domains that hold both roles (FFN GPUs sharing a node with Attention GPUs) run
  DeepEP's union hybrid, which performs exactly like Masked.

Opt-in and off by default: count-first dispatch [§16](docs/v1.md#16-count-first-dispatch-with-a-streaming-ffn-expansion-opt-in-one-nvlink-domain). Every
change, its control and its evidence: [docs/v1.md](docs/v1.md).

## Results

Communication only: one dispatch plus combine round trip under CUDA graphs, in µs,
lower is better, maximum over Attention ranks, paired fresh processes. These are not
serving results. Data: [benchmarks.json](docs/assets/benchmarks.json); figures:
[plot_benchmarks.py](docs/assets/plot_benchmarks.py).

### One NVLink domain

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/results-in-node-h100-dark.svg">
  <img alt="afdEP v1 against Masked DeepEP on 4 H100, six layouts" src="docs/assets/results-in-node-h100.svg">
</picture>

On 4 × H100, v1 cuts the one-token round trip by 27–35% against
Masked and the 2,048-token one by 1–5%, and is faster in every
repeat of every cell. The Masked arm matches the Masked DeepEP that afd-plugin serves
within −0.4% to +1.2%.

<details><summary>4 × H100 numbers</summary>

4 × H100 in one NVLink domain, E128 / top-8 (E96 for 1A:3F), capacity 2,048, BF16 unless noted, 24 transport SMs; median of three fresh-process repeats, max over Attention ranks.

| 2A2F, hidden 2,048, random routing | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP v1, µs | 20.7 | 24.8 | 26.3 | 35.3 | 70.6 | 192.7 |
| Masked DeepEP, µs | 31.7 | 34.4 | 36.0 | 44.3 | 78.7 | 201.0 |
| afdEP v1 vs Masked DeepEP | −34.7% | −27.9% | −26.9% | −20.3% | −10.3% | −4.1% |

| 2A2F, hidden 4,096, random routing | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP v1, µs | 28.4 | 39.2 | 46.3 | 61.5 | 127.3 | 383.2 |
| Masked DeepEP, µs | 39.1 | 47.9 | 55.7 | 68.7 | 136.2 | 392.6 |
| afdEP v1 vs Masked DeepEP | −27.4% | −18.2% | −16.9% | −10.5% | −6.5% | −2.4% |

| 2A2F, hidden 2,048, skew routing | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP v1, µs | 24.1 | 25.7 | 29.5 | 41.2 | 88.7 | 269.9 |
| Masked DeepEP, µs | 34.3 | 35.7 | 38.9 | 50.1 | 95.9 | 274.1 |
| afdEP v1 vs Masked DeepEP | −29.7% | −28.0% | −24.2% | −17.8% | −7.5% | −1.5% |

| 3A:1F, hidden 2,048, random routing | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP v1, µs | 24.8 | 27.4 | 33.0 | 45.9 | 109.4 | 367.8 |
| Masked DeepEP, µs | 36.4 | 38.6 | 44.8 | 55.8 | 119.5 | 372.1 |
| afdEP v1 vs Masked DeepEP | −31.9% | −29.0% | −26.3% | −17.7% | −8.5% | −1.2% |

| 1A:3F, hidden 2,048, random routing | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP v1, µs | 19.9 | 21.5 | 23.1 | 29.4 | 59.6 | 188.3 |
| Masked DeepEP, µs | 30.5 | 31.3 | 32.8 | 39.7 | 67.7 | 197.7 |
| afdEP v1 vs Masked DeepEP | −34.8% | −31.3% | −29.6% | −25.9% | −12.0% | −4.8% |

| 2A2F FP8, hidden 2,048, random routing | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP v1, µs | 21.3 | 25.9 | 26.8 | 34.5 | 62.5 | 159.1 |
| Masked DeepEP, µs | 32.3 | 35.6 | 36.7 | 43.5 | 71.8 | 167.8 |
| afdEP v1 vs Masked DeepEP | −34.1% | −27.2% | −27.0% | −20.7% | −13.0% | −5.2% |

</details>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/results-in-node-h200-dark.svg">
  <img alt="afdEP v1 against Masked DeepEP on 8 H200, three layouts" src="docs/assets/results-in-node-h200.svg">
</picture>

On 8 × H200 the one-token reduction grows with the Attention:FFN ratio:
22% at 4A:4F, 25% at 6A:2F and 39% at 7A:1F, where
v1 is still 13% faster at 2,048 tokens.

<details><summary>8 × H200 numbers</summary>

8 × H200 in one NVLink domain (one pod), BF16, hidden 2048, top-8 of 128 experts, capacity 2,048, 24 SMs; 2 fresh processes per arm, max over Attention ranks.

| 6A:2F | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP v1, µs | 24.3 | 29.3 | 35.5 | 53.2 | 152.0 | 502.0 |
| Masked DeepEP, µs | 32.2 | 38.1 | 43.7 | 61.2 | 157.9 | 507.9 |
| afdEP v1 vs Masked DeepEP | −24.5% | −23.1% | −18.8% | −13.1% | −3.7% | −1.2% |

| 7A:1F | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP v1, µs | 23.8 | 27.2 | 34.1 | 70.3 | 202.0 | 736.8 |
| Masked DeepEP, µs | 39.3 | 46.4 | 53.4 | 97.3 | 244.6 | 847.4 |
| afdEP v1 vs Masked DeepEP | −39.4% | −41.4% | −36.1% | −27.7% | −17.4% | −13.1% |

| 4A:4F | T=1 | 8 | 32 | 128 | 512 | 2,048 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP v1, µs | 24.3 | 26.6 | 27.8 | 39.6 | 96.0 | 296.3 |
| Masked DeepEP, µs | 31.1 | 35.1 | 36.1 | 47.4 | 102.8 | 302.9 |
| afdEP v1 vs Masked DeepEP | −21.9% | −24.2% | −23.0% | −16.5% | −6.6% | −2.2% |

</details>

### Across NVLink domains

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/results-cross-node-dark.svg">
  <img alt="afdEP v1 against Masked DeepEP across NVLink domains, four layouts" src="docs/assets/results-cross-node.svg">
</picture>

With one role per NVLink domain the directional protocol cuts the one-token round
trip by 30% at 11:1 over 12 domains and by 26% at 5:1 over 6 domains. When the
FFN GPUs share their node with Attention GPUs, the union hybrid runs and v1 equals
Masked (within 1.3%). On real 8-GPU nodes that is the common case at high ratios: an
11:1 deployment puts 2 FFN and 6 Attention GPUs on one node.

<details><summary>Cross-node numbers</summary>

H200 joined by InfiniBand (NCCL GIN), FP8, hidden 2048, top-8 of 128 experts, capacity 2,048, 24 SMs; 2 fresh processes per arm, max over Attention ranks.

| 11:1, 12 single-role domains (12 × 2 H200) | T=1 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP v1, µs | 236.1 | 264.9 | 272.7 | 287.3 | 329.9 | 473.1 |
| Masked DeepEP, µs | 339.4 | 354.3 | 352.2 | 361.8 | 399.7 | 494.5 |
| afdEP v1 vs Masked DeepEP | −30.4% | −25.2% | −22.6% | −20.6% | −17.5% | −4.3% |

| 5:1, 6 single-role domains (6 × 4 H200) | T=1 | 8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|---:|
| afdEP v1, µs | 137.8 | 155.3 | 165.0 | 235.1 | 604.8 |
| Masked DeepEP, µs | 185.4 | 193.6 | 210.7 | 277.9 | 636.0 |
| afdEP v1 vs Masked DeepEP | −25.7% | −19.8% | −21.7% | −15.4% | −4.9% |

| 11:1, mixed-role FFN node (3 × 8 H200) | T=1 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|
| afdEP v1, µs | 166.8 | 189.7 | 200.4 | 214.5 | 230.7 | 257.7 |
| Masked DeepEP, µs | 168.9 | 189.9 | 201.0 | 213.9 | 228.1 | 255.6 |
| afdEP v1 vs Masked DeepEP | −1.2% | −0.1% | −0.3% | +0.3% | +1.1% | +0.8% |

| 11:1, mixed-role FFN domain (6 × 4 H200) | T=1 | 8 | 32 | 128 | 512 |
|---|---:|---:|---:|---:|---:|
| afdEP v1, µs | 212.1 | 235.4 | 265.0 | 333.5 | 763.8 |
| Masked DeepEP, µs | 211.3 | 235.7 | 264.6 | 332.4 | 760.4 |
| afdEP v1 vs Masked DeepEP | +0.4% | −0.1% | +0.2% | +0.3% | +0.4% |

</details>

## Use

Linux, PyTorch ≥2.6, a CUDA toolkit with `nvcc`, and NCCL ≥2.30.7 with device headers
are required. The validated toolchain is in [validation](docs/validation.md).

```bash
pip install .
# Set these before starting Python; select the same NCCL headers and runtime.
export AFDEP_NCCL_ROOT=/path/to/nccl
export LD_LIBRARY_PATH="$AFDEP_NCCL_ROOT/lib:$LD_LIBRARY_PATH"
# For a single NVLink domain, without network communication:
export EP_DISABLE_GIN=1
python -m afdep doctor --build
```

```python
from afdep import Buffer, Config

# Set the CUDA device and initialize a CPU Gloo group first.
# Group ranks [0, 2) own experts; [2, 4) originate tokens.
comm = Buffer(
    group=group,
    config=Config(
        capacity_tokens_per_rank=2048,
        hidden_size=4096,
        num_experts=128,
        top_k=8,
        ffn_ranks=2,
        attention_ranks=2,
    ),
)
if comm.role == "attention":
    comm.source_dispatch(x, expert_ids, weights, bucket=2048)
    output = comm.source_collect()
else:
    received = comm.worker_receive(bucket=2048)
    weighted_rows = run_experts(received)  # Caller applies each routing weight once.
    comm.worker_reply(weighted_rows, received.receipt)
comm.close()  # Collective, after all exchanges and graph replays finish.
```

Routing, quantization and expert computation belong to the caller. BF16 and
prequantized FP8 dispatch return BF16 combined outputs. See the
[tensor, graph and lifecycle contract](docs/api.md).

## Support

| Architecture | Evidence | Scope |
|---|---|---|
| Hopper (SM90) | H100 GPU suite; H200 GPU suites over two to five NVLink domains | BF16/FP8; eager and CUDA graphs; one NVLink domain, or several joined by GIN |
| Blackwell (SM100) | Not validated | Port tracked in [#2](https://github.com/afdEP/afdEP/issues/2) |
| Other architectures | Not validated | No support claim yet |

v1 is validated with 2A2F, 3A:1F and 1A:3F in one four-GPU NVLink domain
([evidence](docs/validation-v1-clean.json)), and across two to four NVLink domains of
two H200 GPUs (up to 8 ranks) and four and five single-GPU domains joined by
InfiniBand GIN ([evidence](docs/validation-v1-cross-node.json),
[directional](docs/validation-v1-directional-hybrid.json)). Other layouts and larger
multi-domain worlds require validation. `allow_hybrid_mode` keeps DeepEP's meaning; it
does not enable an unvalidated topology.

[v1 changes and evidence](docs/v1.md) · [Validation](docs/validation.md) ·
[Contributing](CONTRIBUTING.md) · [Licenses and provenance](THIRD_PARTY_NOTICES.md)
