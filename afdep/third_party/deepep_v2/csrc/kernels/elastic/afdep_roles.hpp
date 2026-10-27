// Copyright (c) 2026 afdEP contributors. SPDX-License-Identifier: MIT
#pragma once

#include "../../jit/compiler.hpp"
#include "../../jit/launch_runtime.hpp"

namespace deep_ep::elastic {

// Role-path entry barriers (dispatch Tag0, combine Tag0) are implied by order.
// Each rank runs complete dispatch->combine pairs on one ordered stream; the
// epilogues are PDL dependents that finish before the next main kernel starts.
// Every remote write of exchange N+1 (FFN payload/count inboxes, Attention
// count inboxes, atomic slot counters) is issued after its writer passes the
// combine-N exit barrier, which each rank reaches only after finishing all
// exchange-N inbox reads and counter clears. Combine-N writes into Attention
// inboxes follow the dispatch-N exit barrier, which an Attention rank reaches
// only after its combine-(N-1) reduce epilogue. Every rank skips together, so
// the shared NVLink barrier phase stays aligned. Construction synchronizes all
// ranks before the first exchange. AFDEP_KEEP_ENTRY_BARRIERS=1 restores them as
// a measurement control.
// The direct GIN path (hybrid off across NVLink domains) skips them the same way:
// its exit barriers stay union (system fence, flush of every QP, then a signal to
// every rank and a wait for all of them), so every exchange-(N+1) write follows the
// writer's exit wait of exchange N, which saw each peer's signal, sent after that
// peer's count clears, its epilogue's inbox reads (stream order) and its flush (its
// NIC finished reading staging, even staging inside an inbox). Same-node writes are
// ordered by the acquire on that signal, remote ones by being posted after it.
// Every rank runs the same sequence of world barriers on one set of signal slots,
// whose targets are cumulative, so skipping requires every rank to skip, which
// construction latches and checks. A per-domain FFN prefix skips only combine's,
// since its dispatch keeps the union template (translated routes).
static bool afdep_skip_entry_barriers() {
    return deep_ep::get_env<int>("AFDEP_KEEP_ENTRY_BARRIERS", 0) == 0;
}

// Role-path dispatch sends counts Attention->FFN only; FFN ranks never wait on or
// reduce their own (zero) counts. AFDEP_UNION_NOTIFY=1 restores the all-to-all
// count exchange as a measurement control.
static bool afdep_directional_notify() {
    return deep_ep::get_env<int>("AFDEP_UNION_NOTIFY", 0) == 0;
}

// Directed exit: dispatch sources and combine FFN ranks release per-peer arrival
// counters; only receivers wait (ordering argument in afdep_skip_entry_barriers).
// AFDEP_ALL_RANK_EXIT=1 restores the all-rank exit barriers as a measurement control.
static bool afdep_directed_exit() {
    return deep_ep::get_env<int>("AFDEP_ALL_RANK_EXIT", 0) == 0;
}

// Per-CTA exit: a sending role receives nothing in its phase, so with directed
// exit each sending CTA releases its own arrival once its TMA stores complete,
// without the sender grid sync; receivers wait for round x sender CTAs (every
// rank uses the same transport SM count). The last source CTA resets the slot
// counters. AFDEP_GRID_SYNC_EXIT=1 keeps the grid-synced release as a control.
// Per-domain FFN prefix (ffn_ranks_per_domain f > 0): every NVLink domain holds f FFN ranks,
// its local ranks [0, f), and U - f Attention ranks, so FFN index j is rank (j / f) * U + j % f.
// Every domain then holds both roles, which neither protocol below covers (single-role domains;
// one FFN-holding domain), so these layouts run the union hybrid with per-rank roles: both
// selectors return 0. Sources still validate their real routes, and translate them to DeepEP's
// logical experts (afdep_domain_routes.cuh) for dispatch and combine.
// Silent FFN rails (per-domain FFN prefix): every rail u < f joins FFN GPUs only, none of them a
// source, so no cross-domain write ever happens on it (no counts, payload, tails, replies or
// signals) and the rail halves of its barriers order nothing; DeepEP's rail barrier spans rail
// peers only (ncclTeamTagRail). FFN GPUs then run afdep_domain_hybrid_{dispatch,combine}.cuh, which
// skip every scale-out action and keep their NVLink work (scale-up counts, list ends, partials) and
// all three scale-up barriers; Attention rails keep the union protocol unchanged. One protection
// goes: the FFN GPUs' combine-entry rail barrier held their partials until every domain's dispatch
// had completed, so an Attention forwarder's NIC had finished reading its dispatch staging, which
// dispatch kernels end without flushing. Holders now write the forwarder's combine scale-up inbox
// [0, L * D * b * Bc) right after their scale-up barrier, while that staging, at U * D * b * Bd
// (hybrid_dispatch.cuh), may still be read; L is the regions holders write (f in the rank layout,
// top-k otherwise). The two are disjoint when L * Bc <= U * Bd, which 2L <= U guarantees for every
// payload (a combine row is at most two dispatch rows), so the gate requires it per call and the
// call runs the union kernels otherwise. Every FFN GPU of a rail decides alike from identical
// arguments, and the protocols differ only on FFN rails, so switching per call is safe.
// AFDEP_UNION_HYBRID=1 keeps the union protocol as a measurement control.
static int afdep_domain_hybrid(const int afd_ffn_ranks, const int num_scaleout_ranks, const int num_scaleup_ranks,
                               const int afd_ffn_ranks_per_domain) {
    return afd_ffn_ranks_per_domain > 0 and num_scaleout_ranks > 1 and
        afd_ffn_ranks == num_scaleout_ranks * afd_ffn_ranks_per_domain and
        2 * afd_ffn_ranks_per_domain <= num_scaleup_ranks and deep_ep::get_env<int>("AFDEP_UNION_HYBRID", 0) == 0;
}

// The per-call gate above: the combine scale-up regions FFN GPUs write fit below the dispatch staging.
static bool afdep_domain_silent_rails(const int afd_ffn_ranks, const int num_scaleout_ranks,
                                      const int num_scaleup_ranks, const int afd_ffn_ranks_per_domain,
                                      const int num_topk, const bool allow_multiple_reduction) {
    const int regions = allow_multiple_reduction and num_scaleup_ranks <= num_topk ? afd_ffn_ranks_per_domain : num_topk;
    return afdep_domain_hybrid(afd_ffn_ranks, num_scaleout_ranks, num_scaleup_ranks, afd_ffn_ranks_per_domain) > 0 and
        2 * regions <= num_scaleup_ranks;
}

// Directional hybrid protocol: when every scale-out (NVLink) domain holds a single
// role, dispatch traffic (counts, payloads, finish tails) flows Attention -> FFN
// and combine traffic (payloads, finish signals) FFN -> Attention only; FFN
// domains skip scale-out sends, Attention domains skip forwarding and combine
// reduction. Every exchange-(N+1) cross-domain write then follows its reader's
// exchange-N consumption: Attention writes into an FFN domain only after its
// combine-N wait received that FFN rank's per-channel signals, sent after the
// FFN's dispatch-N forwarders cleaned their tails and the notify cleaned its
// counts; an FFN rank writes into an Attention domain only after its dispatch-
// (N+1) forwarders saw that source's finish tails, sent after the source's
// combine-N wait and reduce epilogue. The cross-domain half of both entry
// barriers is dropped; the scale-up half still orders reuse inside a domain.
// Returns the FFN scale-out domain count, or 0 (union protocol).
// AFDEP_UNION_HYBRID=1 keeps the union protocol as a measurement control.
static int afdep_directional_hybrid(const int afd_ffn_ranks, const int num_scaleout_ranks,
                                    const int num_scaleup_ranks, const int afd_ffn_ranks_per_domain) {
    if (afd_ffn_ranks <= 0 or afd_ffn_ranks_per_domain > 0 or num_scaleout_ranks <= 1 or
        afd_ffn_ranks % num_scaleup_ranks != 0 or deep_ep::get_env<int>("AFDEP_UNION_HYBRID", 0) != 0)
        return 0;
    return afd_ffn_ranks / num_scaleup_ranks;
}

// Mixed-role directional hybrid (opt-in, AFDEP_MIXED_HYBRID=1): FFN ranks [0, F) with
// 0 < F < U fill part of NVLink domain 0, which they share with U - F Attention ranks
// (A0), and every other domain holds only Attention ranks (AP). The directional
// protocol above applies with one FFN-holding domain and roles per rank: every domain-0
// GPU stays its rail's forwarder, so remote tokens still enter through all U NICs; only
// ranks below F hold experts; A0 GPUs are also sources, whose own tokens take the local
// bypass into their own forwarder (rail lane 0) and whose replies their own forward
// warps write locally. Its orderings are the directional ones plus:
// - Domain 0 keeps all three scale-up barriers (dispatch entry and exit, combine entry)
//   across all U GPUs, A0 included, so state shared inside domain 0 (scale-up inboxes,
//   tails, counts, list ends) is ordered as in union.
// - An AP writes dispatch N+1 into forwarder (0,u) only after its collect N saw (0,u)'s
//   signal on every channel, each sent after that channel's combine-N reads; the one
//   region (0,u) still reads afterwards is A0's reply inbox (its reduce epilogue), which
//   therefore moves, on every rank, into a slice reserved before the combine staging
//   slice at the window end, past both layouts (calculate_buffer_size).
// - Forwarders stage AP replies in the section-11 staging slice with two or more AP
//   domains; A0 never stages dispatch tokens (all are local) and nothing is written into
//   an AP window except its reply slice and signal words.
// Returns F, or 0 (union, or the directional protocol when F % U == 0).
static int afdep_mixed_hybrid(const int afd_ffn_ranks, const int num_scaleout_ranks,
                              const int num_scaleup_ranks, const int afd_ffn_ranks_per_domain) {
    if (afd_ffn_ranks <= 0 or afd_ffn_ranks_per_domain > 0 or num_scaleout_ranks <= 1 or
        afd_ffn_ranks >= num_scaleup_ranks or
        deep_ep::get_env<int>("AFDEP_UNION_HYBRID", 0) != 0 or deep_ep::get_env<int>("AFDEP_MIXED_HYBRID", 0) == 0)
        return 0;
    return afd_ffn_ranks;
}

// Directional hybrid dispatch, and direct dispatch over GIN (hybrid off across
// NVLink domains), validate each route in their notify and payload warps before
// the first route-derived index, store or put, as the direct NVLink path does,
// instead of a separate route-guard kernel ahead of every dispatch (the guard
// exists only because the union world gives Attention ranks dummy expert slots).
// AFDEP_HYBRID_ROUTE_GUARD=1 keeps the guard kernel as a control on the directional
// hybrid; the direct GIN dispatch (afdep_direct_gin_dispatch.cuh) always checks, and
// translates, its routes in-kernel, for global and per-domain FFN prefixes.
static bool afdep_hybrid_routes_in_kernel() {
    return deep_ep::get_env<int>("AFDEP_HYBRID_ROUTE_GUARD", 0) == 0;
}

// Directional hybrid: Attention domains run neither half of the dispatch entry
// barrier, so its grid sync only holds their CTAs (and the first put) behind the
// slowest CTA. AFDEP_ATTENTION_GRID_SYNC=1 keeps it as a measurement control.
static bool afdep_skip_attention_entry_sync() {
    return deep_ep::get_env<int>("AFDEP_ATTENTION_GRID_SYNC", 0) == 0;
}

// Single-FFN reduction: DeepEP's FFN combine reduces a token's local top-k rows
// with a branch-free prefix of ceil(top_k / ranks) slots, an expected count for
// tokens spread over every rank. On the role path the Attention ranks hold no
// experts; with one FFN rank every token's top-k rows are local, so all top_k slots
// are valid on every token and the whole reduction takes the branch-free path.
// AFDEP_UNION_REDUCE_PREFIX=1 keeps the union prefix as a measurement control.
static bool afdep_single_ffn_reduce() {
    return deep_ep::get_env<int>("AFDEP_UNION_REDUCE_PREFIX", 0) == 0;
}

static bool afdep_per_cta_exit() {
    return deep_ep::get_env<int>("AFDEP_GRID_SYNC_EXIT", 0) == 0;
}

// Role-sized registered buffers: in one NVLink domain only Attention sources
// write FFN dispatch inboxes and only FFN ranks write Attention combine inboxes,
// so the union window's FFN-source dispatch regions and Attention-source combine
// regions are never written. Dispatch regions are indexed by source - FFN ranks.
// AFDEP_UNION_BUFFERS=1 keeps upstream union sizing as a measurement control.
static bool afdep_role_buffers() {
    return deep_ep::get_env<int>("AFDEP_UNION_BUFFERS", 0) == 0;
}

class AfdEPRouteGuardRuntime final : public jit::LaunchRuntime<AfdEPRouteGuardRuntime> {
public:
    struct Args {
        int real_experts, top_k;
        topk_idx_t* ids;
        int num_tokens;
        jit::LaunchArgs launch_args;
    };
    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/afdep_roles.cuh>
using namespace deep_ep::elastic;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&afdep_validate_source_routes_impl<{}, {}>);
}}
)", args.real_experts, args.top_k);
    }
    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config, args.ids, args.num_tokens));
    }
};

static void launch_afdep_route_guard(topk_idx_t* ids, const int num_tokens,
                                     const int real_experts, const int top_k,
                                     const int num_sms, const at::cuda::CUDAStream& stream) {
    if (num_tokens == 0)
        return;
    const AfdEPRouteGuardRuntime::Args args = {
        real_experts, top_k, ids, num_tokens,
        jit::LaunchArgs(std::min(num_sms, (num_tokens + 7) / 8), 256)
    };
    const auto runtime = jit::compiler->build("afdep_route_guard", AfdEPRouteGuardRuntime::generate(args));
    AfdEPRouteGuardRuntime::launch(runtime, args, stream);
}

class AfdEPDomainRoutesRuntime final : public jit::LaunchRuntime<AfdEPDomainRoutesRuntime> {
public:
    struct Args {
        int real_experts, top_k, ffn_ranks_per_domain, scaleup_ranks, experts_per_rank;
        const topk_idx_t* ids;
        topk_idx_t* logical_ids;
        int num_tokens;
        jit::LaunchArgs launch_args;
    };
    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_ep/impls/afdep_domain_routes.cuh>
using namespace deep_ep::elastic;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&afdep_domain_routes_impl<{}, {}, {}, {}, {}>);
}}
)", args.real_experts, args.top_k, args.ffn_ranks_per_domain, args.scaleup_ranks, args.experts_per_rank);
    }
    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config, args.ids, args.logical_ids, args.num_tokens));
    }
};

// The route guard for a per-domain FFN prefix: validates real routes and writes their logical IDs.
static void launch_afdep_domain_routes(const topk_idx_t* ids, topk_idx_t* logical_ids, const int num_tokens,
                                       const int real_experts, const int top_k,
                                       const int ffn_ranks_per_domain, const int scaleup_ranks,
                                       const int experts_per_rank,
                                       const int num_sms, const at::cuda::CUDAStream& stream) {
    if (num_tokens == 0)
        return;
    const AfdEPDomainRoutesRuntime::Args args = {
        real_experts, top_k, ffn_ranks_per_domain, scaleup_ranks, experts_per_rank,
        ids, logical_ids, num_tokens,
        jit::LaunchArgs(std::min(num_sms, (num_tokens + 7) / 8), 256)
    };
    const auto runtime = jit::compiler->build("afdep_domain_routes", AfdEPDomainRoutesRuntime::generate(args));
    AfdEPDomainRoutesRuntime::launch(runtime, args, stream);
}

class AfdEPCompletionRuntime final : public jit::LaunchRuntime<AfdEPCompletionRuntime> {
public:
    struct Args {
        bool source;
        int scaleout_ranks, scaleup_ranks, experts, channels;
        void* workspace;
        int* channel_linked_list;
        int* rank_psum;
        jit::LaunchArgs launch_args;
    };
    static std::string generate_impl(const Args& args) {
        const auto function = args.source ? fmt::format(
            "afdep_source_finalize_impl<{}, {}, {}, {}>",
            args.scaleout_ranks, args.scaleup_ranks, args.experts, args.channels)
            : "afdep_ffn_complete_impl<>";
        return fmt::format(R"(
#include <deep_ep/impls/afdep_roles.cuh>
using namespace deep_ep::elastic;
static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&{});
}}
)", function);
    }
    static void launch_impl(const jit::KernelHandle& kernel, const jit::LaunchConfigHandle& config, Args args) {
        if (args.source) {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(
                kernel, config, args.workspace, args.channel_linked_list, args.rank_psum));
        } else {
            EP_CUDA_UNIFIED_CHECK(jit::launch_kernel(kernel, config, args.workspace));
        }
    }
};

static void launch_afdep_completion(const bool source, void* workspace,
                                    int* channel_linked_list, int* rank_psum,
                                    const int scaleout_ranks, const int scaleup_ranks,
                                    const int experts, const int channels,
                                    const at::cuda::CUDAStream& stream) {
    const AfdEPCompletionRuntime::Args args = {
        source, scaleout_ranks, scaleup_ranks, experts, channels,
        workspace, channel_linked_list, rank_psum,
        jit::LaunchArgs(1, 128, 0, 1, false, true)
    };
    const auto runtime = jit::compiler->build(
        source ? "afdep_source_finalize" : "afdep_ffn_complete",
        AfdEPCompletionRuntime::generate(args));
    AfdEPCompletionRuntime::launch(runtime, args, stream);
}

}  // namespace deep_ep::elastic
