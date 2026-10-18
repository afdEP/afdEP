// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the AFDEP project
#pragma once

#include <c10/util/Exception.h>
#include <cuda_runtime_api.h>
#include <nccl.h>

static_assert(NCCL_VERSION_CODE >= 23007, "AFDEP requires NCCL 2.30.7 or newer");

#define AFDEP_CUDA_CHECK(expression)                                                                         \
    do {                                                                                                     \
        const cudaError_t status = (expression);                                                             \
        TORCH_CHECK(status == cudaSuccess, "AFDEP CUDA error: ", cudaGetErrorString(status), " (",           \
                    #expression, ")");                                                                       \
    } while (false)

#define AFDEP_NCCL_CHECK(expression)                                                                         \
    do {                                                                                                     \
        const ncclResult_t status = (expression);                                                            \
        TORCH_CHECK(status == ncclSuccess, "AFDEP NCCL error: ", ncclGetErrorString(status), " (",           \
                    #expression, ")");                                                                       \
    } while (false)
