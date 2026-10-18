// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
// SPDX-FileCopyrightText: Copyright contributors to the AFDEP project
#pragma once

#include "layout.hpp"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nccl_device.h>

namespace afdep {

__device__ __forceinline__ Control* control(uint8_t* base) {
    return reinterpret_cast<Control*>(base);
}
__device__ __forceinline__ uint64_t acquire(const uint64_t* p) {
    uint64_t value;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) : "l"(p) : "memory");
    return value;
}
__device__ __forceinline__ void release(uint64_t* p, uint64_t value) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(p), "l"(value) : "memory");
}
__device__ __forceinline__ void fail(const char* reason, const DeviceArgs& a) {
    printf("AFDEP timeout/bounds: %s rank=%d epoch=%llu\n", reason,
           a.rank, static_cast<unsigned long long>(control(a.base)->epoch));
    asm volatile("trap;");
}
__device__ __forceinline__ void wait_local(const uint64_t* p, uint64_t target,
                                         const DeviceArgs& a) {
    const uint64_t start = clock64();
    while (acquire(p) < target) {
        if (clock64() - start >= a.timeout_cycles) fail("local phase", a);
    }
}
__device__ __forceinline__ void wait_signal(int context, int signal, uint64_t target,
                                          const DeviceArgs& a) {
    ncclGin gin(a.comm, context, NCCL_GIN_RESOURCE_SHARING_CTA);
    const uint64_t start = clock64();
    while (gin.readSignal(static_cast<ncclGinSignal_t>(signal)) < target) {
        if (clock64() - start >= a.timeout_cycles) fail("network phase", a);
    }
}
__device__ __forceinline__ void send_credit(int signal, int peer, const DeviceArgs& a) {
    ncclGin gin(a.comm, 0, NCCL_GIN_RESOURCE_SHARING_CTA);
    gin.signal(ncclTeamRail(a.comm), peer,
               ncclGin_StrongSignalInc{static_cast<ncclGinSignal_t>(signal)});
    gin.flush(ncclCoopThread());
}
__device__ __forceinline__ int packet_tokens(uint8_t* base, const Layout& l) {
    return static_cast<int>(reinterpret_cast<volatile Packet*>(base + l.request)->tokens);
}
__device__ __forceinline__ uint4 load_remote_vector(const void* pointer) {
    uint4 value;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(pointer) : "memory");
    return value;
}

// One final strong signal per active context. Each receiver waits every one;
// a signal on context zero cannot order transfers issued on other contexts.
__device__ __forceinline__ void transfer(const DeviceArgs& a, size_t src, size_t dst,
                                       size_t bytes, int peer, int signal) {
    if (threadIdx.x != 0) return;
    const int q = blockIdx.x;
    const size_t units = (bytes + 15) / 16;
    const size_t begin = (units * q / a.stripes) * 16;
    const size_t end = min((units * (q + 1) / a.stripes) * 16, bytes);
    ncclGin gin(a.comm, q, NCCL_GIN_RESOURCE_SHARING_CTA);
    const auto action = ncclGin_StrongSignalInc{static_cast<ncclGinSignal_t>(signal)};
    if (end > begin) {
        gin.put(ncclTeamRail(a.comm), peer,
                a.window, dst + begin, a.window, src + begin, end - begin,
                action, ncclGin_None(), ncclCoopThread(), ncclGin_None(),
                cuda::thread_scope_thread, cuda::thread_scope_device,
                ncclGinOptFlagsDefault);
    } else {
        // Empty sources still advance their phase so workers cannot deadlock.
        gin.signal(ncclTeamRail(a.comm), peer, action);
    }
    // Local source consumption only. Remote readiness is the signal above.
    gin.flush(ncclCoopThread());
}

template <int kOperation>
__global__ void directional_kernel(DeviceArgs a) {
    const Layout l = a.layout;
    Control* state = control(a.base);
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const int worker = a.rank;

    if constexpr (kOperation == BeginSource) {
        if (tid == 0) {
            const uint64_t next = state->epoch + 1;
            wait_signal(0, RequestCredit, next - 1, a);
            state->epoch = next;
            state->source_tokens = a.tokens;
        }
    } else if constexpr (kOperation == PackSource) {
        uint8_t* packet = a.base + l.request;
        if (tid == 0) {
            reinterpret_cast<Packet*>(packet)->tokens = a.tokens;
            reinterpret_cast<Packet*>(packet)->reserved = 0;
        }
        const size_t xbytes = size_t(a.tokens) * l.hidden * l.element_bytes();
        for (size_t i = tid; i < xbytes / 16; i += stride)
            reinterpret_cast<uint4*>(packet + l.packet_x())[i] =
                reinterpret_cast<const uint4*>(a.x)[i];
        for (int i = tid; i < a.tokens * l.topk; i += stride) {
            const int64_t expert = a.ids[i];
            if (expert < -1 || expert >= l.experts) fail("invalid real expert", a);
            for (int previous = 0; expert >= 0 && previous < i % l.topk; ++previous)
                if (a.ids[i - i % l.topk + previous] == expert)
                    fail("duplicate expert for one token", a);
            reinterpret_cast<int64_t*>(packet + l.packet_ids(a.tokens))[i] = a.ids[i];
            reinterpret_cast<float*>(packet + l.packet_weights(a.tokens))[i] = a.weights[i];
        }
        for (int i = tid; i < a.tokens * l.scale_groups(); i += stride) {
            const int token = i / l.scale_groups(), group = i % l.scale_groups();
            reinterpret_cast<float*>(packet + l.packet_scales(a.tokens))[i] =
                a.scales[token * a.scale_stride0 + group * a.scale_stride1];
        }
    } else if constexpr (kOperation == SendRequest) {
        transfer(a, l.request, l.request, l.packet_bytes(a.tokens), 0, RequestReady);
    } else if constexpr (kOperation == BeginWorker) {
        if (tid == 0) {
            ++state->epoch;
            for (int q = 0; q < a.stripes; ++q)
                wait_signal(q, RequestReady, state->epoch, a);
            release(&state->request_ready, state->epoch);
        }
    } else if constexpr (kOperation == Histogram) {
        if (threadIdx.x == 0) {
            for (int source = 0; source < 4; ++source)
                wait_local(&control(a.peers[source])->request_ready, state->epoch, a);
        }
        __syncthreads();
        if (tid == 0) {
            for (int source = 0; source < 4; ++source) {
                const int n = packet_tokens(a.peers[source], l);
                if (n < 0 || n > a.bucket) fail("source tokens exceed bucket", a);
                state->received_tokens[source] = n;
            }
        }
        for (int i = tid; i < 4 * a.bucket * l.topk; i += stride) {
            const int source = i / (a.bucket * l.topk);
            const int token = (i / l.topk) % a.bucket, k = i % l.topk;
            const int n = packet_tokens(a.peers[source], l);
            if (token >= n) continue;
            const uint8_t* packet = a.peers[source] + l.request;
            const int64_t expert = reinterpret_cast<const volatile int64_t*>(packet + l.packet_ids(n))[token * l.topk + k];
            if (expert < -1 || expert >= l.experts) fail("invalid real expert", a);
            if (expert >= worker * l.local_experts() && expert < (worker + 1) * l.local_experts())
                atomicAdd(a.counts + expert - worker * l.local_experts(), 1);
        }
    } else if constexpr (kOperation == Prefix) {
        if (tid == 0) {
            int offset = 0;
            for (int e = 0; e < l.local_experts(); ++e) {
                offset = (offset + l.alignment - 1) / l.alignment * l.alignment;
                a.starts[e] = offset;
                offset += a.counts[e];
                a.psum[e] = offset;
                a.cursors[e] = 0;
            }
        }
    } else if constexpr (kOperation == PackRows) {
        __shared__ int slots[32];
        for (int index = blockIdx.x; index < 4 * a.bucket; index += gridDim.x) {
            const int source = index / a.bucket, token = index % a.bucket;
            const int n = state->received_tokens[source];
            if (token >= n) continue;
            const uint8_t* packet = a.peers[source] + l.request;
            if (threadIdx.x < l.topk) {
                const int k = threadIdx.x;
                const int64_t expert = reinterpret_cast<const volatile int64_t*>(packet + l.packet_ids(n))[token * l.topk + k];
                int row = -1;
                if (expert >= worker * l.local_experts() && expert < (worker + 1) * l.local_experts()) {
                    const int e = expert - worker * l.local_experts();
                    row = a.starts[e] + atomicAdd(a.cursors + e, 1);
                    const int max_rows = 4 * a.bucket * min(l.topk, l.local_experts()) +
                                         l.local_experts() * (l.alignment - 1);
                    if (row < 0 || row >= max_rows) fail("expert row overflow", a);
                    a.recv_weights[row] = reinterpret_cast<const volatile float*>(packet + l.packet_weights(n))[token * l.topk + k];
                    a.receipt[(source * a.bucket + token) * l.topk + k] = row;
                }
                slots[k] = row;
            }
            __syncthreads();
            for (int k = 0; k < l.topk; ++k) {
                const int row = slots[k];
                if (row < 0) continue;
                const int vectors = l.hidden * l.element_bytes() / 16;
                for (int h = threadIdx.x; h < vectors; h += blockDim.x)
                    reinterpret_cast<uint4*>(a.recv_x)[row * vectors + h] =
                        load_remote_vector(packet + l.packet_x() + size_t(token * vectors + h) * sizeof(uint4));
                for (int group = threadIdx.x; group < l.scale_groups(); group += blockDim.x)
                    a.recv_scales[row + group * a.recv_scale_stride1] =
                        reinterpret_cast<const volatile float*>(packet + l.packet_scales(n))[token * l.scale_groups() + group];
            }
            __syncthreads();
        }
    } else if constexpr (kOperation == PublishPackAndCredit) {
        if (tid == 0) {
            // A distinct kernel launch follows the entire payload-copy grid.
            release(&state->pack_done, state->epoch);
            for (int i = 0; i < 4; ++i)
                wait_local(&control(a.peers[i])->pack_done, state->epoch, a);
            send_credit(RequestCredit, 1, a);
        }
    } else if constexpr (kOperation == MakePartials) {
        if (threadIdx.x == 0) {
            for (int i = 0; i < 4; ++i)
                wait_local(&control(a.peers[i])->partials_consumed, state->epoch - 1, a);
        }
        __syncthreads();
        for (size_t i = tid; i < size_t(4) * a.bucket * l.hidden; i += stride) {
            const int source = i / (size_t(a.bucket) * l.hidden);
            const int token = (i / l.hidden) % a.bucket, h = i % l.hidden;
            float sum = 0;
            for (int k = 0; k < l.topk; ++k) {
                const int row = a.receipt[(source * a.bucket + token) * l.topk + k];
                if (row >= 0)
                    sum += __bfloat162float(reinterpret_cast<const nv_bfloat16*>(a.rows)[size_t(row) * l.hidden + h]);
            }
            reinterpret_cast<nv_bfloat16*>(a.base + l.partial(source, token))[h] = __float2bfloat16(sum);
        }
    } else if constexpr (kOperation == PublishPartialAndWait) {
        if (tid == 0) {
            release(&state->partial_done, state->epoch);
            for (int i = 0; i < 4; ++i)
                wait_local(&control(a.peers[i])->partial_done, state->epoch, a);
        }
    } else if constexpr (kOperation == ReduceReply) {
        const int n = state->received_tokens[worker];
        for (size_t i = tid; i < size_t(n) * l.hidden; i += stride) {
            const int token = i / l.hidden, h = i % l.hidden;
            float sum = 0;
            for (int iworker = 0; iworker < 4; ++iworker)
                sum += __bfloat162float(__ushort_as_bfloat16(
                    reinterpret_cast<const volatile uint16_t*>(
                        a.peers[iworker] + l.partial(worker, token))[h]));
            reinterpret_cast<nv_bfloat16*>(a.base + l.worker_reply)[i] = __float2bfloat16(sum);
        }
    } else if constexpr (kOperation == PublishConsumedAndWaitCredit) {
        if (tid == 0) {
            // All reduction-grid readers finished before partials may be reused.
            release(&state->partials_consumed, state->epoch);
            wait_signal(0, ReplyCredit, state->epoch - 1, a);
        }
    } else if constexpr (kOperation == SendReply) {
        const size_t bytes = size_t(state->received_tokens[worker]) * l.hidden * sizeof(uint16_t);
        transfer(a, l.worker_reply, l.source_reply, bytes, 1, ReplyReady);
    } else if constexpr (kOperation == CollectReply) {
        if (threadIdx.x == 0) {
            for (int q = 0; q < a.stripes; ++q)
                wait_signal(q, ReplyReady, state->epoch, a);
        }
        __syncthreads();
        const size_t bytes = size_t(state->source_tokens) * l.hidden * sizeof(uint16_t);
        for (size_t i = tid; i < bytes / 16; i += stride)
            reinterpret_cast<uint4*>(a.out)[i] = load_remote_vector(
                a.base + l.source_reply + i * sizeof(uint4));
    } else if constexpr (kOperation == ReplyConsumedCredit) {
        if (tid == 0) send_credit(ReplyCredit, 0, a);
    }
}

}  // namespace afdep
