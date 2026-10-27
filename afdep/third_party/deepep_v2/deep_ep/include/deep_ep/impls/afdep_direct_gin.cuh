// Copyright (c) 2026 afdEP contributors. SPDX-License-Identifier: MIT
#pragma once

#include <deep_ep/impls/afdep_roles.cuh>

namespace deep_ep::elastic {

// afdEP direct GIN roles (afdep_roles.hpp): hybrid off folds the NVLink domains into one logical
// domain of W ranks, whose FFN ranks are the global prefix [0, F) (kFfnRanksPerDomain = 0) or the
// first f ranks of every U-rank domain. FFN index j owns real experts [j * E, (j + 1) * E), DeepEP's
// logical experts [rank * E, (rank + 1) * E): real expert e is logical e + e / (f * E) * (U - f) * E,
// the identity for the global prefix (afdep_domain_routes.cuh).
template <int kNumRanks, int kNumFfnRanks, int kFfnRanksPerDomain, int kDomainRanks>
struct AfdepGinRoles {
    static constexpr int kLocalFfn = kFfnRanksPerDomain > 0 ? kFfnRanksPerDomain : kNumFfnRanks;
    static constexpr int kLocalRanks = kFfnRanksPerDomain > 0 ? kDomainRanks : kNumRanks;
    EP_STATIC_ASSERT(kLocalRanks > 0 and kNumRanks % kLocalRanks == 0 and
                     kNumFfnRanks == kNumRanks / kLocalRanks * kLocalFfn and 0 < kLocalFfn and kLocalFfn < kLocalRanks,
                     "Invalid direct GIN FFN layout");

    __device__ __forceinline__ static bool is_ffn(const int& rank) { return rank % kLocalRanks < kLocalFfn; }

    // The rank of FFN index j and of source (Attention) index s.
    __device__ __forceinline__ static int ffn_rank(const int& j) { return j / kLocalFfn * kLocalRanks + j % kLocalFfn; }
    __device__ __forceinline__ static int source_rank(const int& s) {
        return s / (kLocalRanks - kLocalFfn) * kLocalRanks + kLocalFfn + s % (kLocalRanks - kLocalFfn);
    }

    template <int kNumExpertsPerRank>
    __device__ __forceinline__ static int logical(const int& expert) {
        return expert < 0 ? expert :
            expert + expert / (kLocalFfn * kNumExpertsPerRank) * ((kLocalRanks - kLocalFfn) * kNumExpertsPerRank);
    }
};

}  // namespace deep_ep::elastic
