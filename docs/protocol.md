# Directional protocol

The protocol implements one logical exchange between four Attention sources and
four FFN workers. All persistent counters are device state. They advance on
each exchange, including CUDA graph replays, without host counter updates.

## Request and reply path

1. Source `Aj` waits for the previous request credit, advances its epoch, and
   packs a header, activations, routing IDs, weights, and optional FP8 scales.
2. It transfers the request to matching ingress worker `Fj` using registered
   NCCL GIN windows. Each active GIN context produces a strong readiness signal.
3. Each worker waits for its matching request and publishes local readiness.
   All FFN workers wait for the four ingress requests, count their own real
   expert rows, compute aligned offsets, and pack rows through NVLink reads.
4. Workers publish packing completion. Only after all four readers finish does
   each ingress return its request-buffer credit to the matching source.
5. The caller computes expert outputs and applies the received routing weights.
   Each worker reduces its local selected expert rows into one partial per
   source token, using the receive receipt to restore token association.
6. Workers publish partial readiness. Each ingress gathers and sums the four
   workers' partials for its matching source, then publishes partial consumption.
7. The ingress waits for the previous reply-buffer credit and sends its reduced
   BF16 reply to the source. Every active GIN context signals completion.
8. The source waits for all reply signals, copies the result into the caller's
   output tensor, and returns the reply-buffer credit.

Source and worker epochs must refer to the same logical exchange. Actual source
counts travel in headers; a positive bucket bounds all of them. Empty requests
and zero-byte replies still advance completion signals. An empty rank cannot
omit the exchange.

## Why credits are required

| Shared storage | Readers | Safe-reuse condition |
|---|---|---|
| FFN ingress request | All four FFN packing stages | Every worker published packing completion |
| Per-worker partials | All four ingress reply reductions | Every ingress published partial consumption |
| Attention reply window | Attention output-copy grid | The full copy grid completed and credit was returned |

A notification that data arrived is not permission to overwrite it. These
reader acknowledgments are separate from readiness. In particular, a fast
worker must not overwrite its partials while another ingress still gathers the
previous exchange. A source must not publish its next request before the FFN
readers release the previous request window.

Whole-grid completion boundaries precede publication kernels. Local flags use
system-scope release/acquire ordering so remote NVLink readers observe payload
and metadata before a published phase. Do not replace those publications with
an ordinary store or assume one CUDA block's completion covers the entire grid.

## GIN context ordering

The request/reply bytes are divided across active transport stripes. A stripe
uses its own GIN context. Receivers wait for a strong signal on **every** active
context before reading the message. A signal on context zero does not order
transfers issued on other contexts.

Each transfer stripe signals even if its assigned byte range is empty. GIN
flush establishes local source consumption for the issuing transfer; the strong
remote readiness signal establishes the receive-side handoff. Local completion
and remote readiness are different obligations.

Request and reply credits use context zero after the relevant complete reader
phase. Credits do not remove the per-context readiness requirement for payloads.

## Differences from masked DeepEP

The comparison baseline in AFD plugin PR #23 gives both roles membership in one
general expert-parallel group and masks the unused work. That produces the
desired useful payload flow, while retaining general dispatch/combine setup and
control work, including dummy expert slots.

AFDEP gives the roles different operations and layouts. Workers do not originate
empty token dispatches; sources do not own dummy experts or perform empty expert
combines. It retains FFN-local fan-out, completion handshakes, and reuse credits
because they protect real readers. Avoid describing the protocol as
“synchronization-free” or attributing a measured gain to removed bookkeeping
without a corresponding phase-level measurement.

## Change review requirements

Every change to transfer ordering, graph replay, or buffer reuse must establish:

- Which GPU can write each region, and when each reader is finished.
- Which release/acquire or GIN handoff orders the relevant payload.
- Whether completion is required from one block, every block, or every context.
- How empty messages and different actual source counts advance the phase.
- How a delayed peer behaves across consecutive epochs and interleaved buckets.

The single-lane protocol does not cover arbitrary N/M placement, multiple
in-flight exchanges, rank failure recovery, or concurrent use of one buffer.
Those extensions require new layouts and ownership proofs as well as tests.
