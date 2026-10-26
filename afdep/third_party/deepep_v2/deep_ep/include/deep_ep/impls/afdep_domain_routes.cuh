// Copyright (c) 2026 afdEP contributors. SPDX-License-Identifier: MIT
#pragma once

#include <deep_ep/impls/afdep_roles.cuh>

namespace deep_ep::elastic {

// Per-domain FFN prefix (afdep_roles.hpp): FFN index j is rank (j / f) * U + j % f and owns
// real experts [j * E, (j + 1) * E), DeepEP's logical experts [rank * E, (rank + 1) * E).
// Real expert e is therefore logical e + (e / (f * E)) * (U - f) * E: each earlier domain's
// Attention ranks add their U - f empty blocks. Sources check each real route as the route guard
// does, then write the logical IDs that dispatch reads and copies for combine.
template <int kNumRealExperts, int kNumTopk, int kFfnRanksPerDomain, int kNumScaleupRanks, int kNumExpertsPerRank>
__global__ void afdep_domain_routes_impl(const topk_idx_t* ids, topk_idx_t* logical_ids, const int num_tokens) {
    EP_STATIC_ASSERT(0 < kFfnRanksPerDomain and kFfnRanksPerDomain < kNumScaleupRanks, "Invalid per-domain FFN prefix");
    constexpr int kDomainExperts = kFfnRanksPerDomain * kNumExpertsPerRank;
    constexpr int kAttentionExperts = (kNumScaleupRanks - kFfnRanksPerDomain) * kNumExpertsPerRank;
    const auto lane = ptx::get_lane_idx();
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int warps = gridDim.x * blockDim.x / 32;
    for (int token = warp; token < num_tokens; token += warps) {
        const topk_idx_t expert = lane < kNumTopk ? ids[token * kNumTopk + lane] : -1;
        const int real = afdep_checked_expert<kNumRealExperts>(expert, lane);
        if (lane < kNumTopk)
            logical_ids[token * kNumTopk + lane] = real < 0 ? -1 : real + real / kDomainExperts * kAttentionExperts;
    }
}

}  // namespace deep_ep::elastic
