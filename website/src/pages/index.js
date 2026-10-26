import React from 'react';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import CodeBlock from '@theme/CodeBlock';
import useBaseUrl from '@docusaurus/useBaseUrl';

const stats = [
  ['0', 'entry barriers in one NVLink domain'],
  ['50%', 'registered buffers vs the union, at 2A2F'],
  ['-68%', 'small-batch cross-node round trip, best case'],
];

export default function Home() {
  return (
    <Layout description="Expert-parallel communication for Attention-FFN Disaggregation">
      <header className="hero-afdep">
        <img src={useBaseUrl('/afdep-logo.png')} alt="afdEP-comm" />
        <p>
          Expert-parallel communication for Attention-FFN Disaggregation.
          DeepEP V2 kernels specialized for the two roles: Attention GPUs only send
          in dispatch, FFN GPUs only send in combine.
        </p>
        <Link className="button button--primary" to="/docs/api">API</Link>
        <Link className="button button--secondary button--outline" to="/docs/design">Design</Link>
      </header>
      <section className="stats">
        {stats.map(([value, label]) => (
          <div key={label}><b>{value}</b><span>{label}</span></div>
        ))}
      </section>
      <main className="quick">
        <h2>Quick start</h2>
        <CodeBlock language="bash">{`pip install .
export AFDEP_NCCL_ROOT=/path/to/nccl
export LD_LIBRARY_PATH="$AFDEP_NCCL_ROOT/lib:$LD_LIBRARY_PATH"
python -m afdep doctor --build`}</CodeBlock>
        <CodeBlock language="python">{`comm = Buffer(group=group, config=Config(
    capacity_tokens_per_rank=2048, hidden_size=4096, num_experts=128,
    top_k=8, ffn_ranks=2, attention_ranks=2))
if comm.role == "attention":
    comm.source_dispatch(x, expert_ids, weights, bucket=2048)
    output = comm.source_collect()
else:
    received = comm.worker_receive(bucket=2048)
    comm.worker_reply(run_experts(received), received.receipt)
comm.close()`}</CodeBlock>
      </main>
    </Layout>
  );
}
