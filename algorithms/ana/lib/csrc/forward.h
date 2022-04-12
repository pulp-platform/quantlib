/*
 *  forward.h
 *
 *  Author(s):
 *  Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
 *
 *  Copyright (c) 2020-2021 ETH Zurich.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <cuda.h>
#include <cuda_runtime.h>


#define PLUS_1(x) (x + 1)


template <typename scalar_t>
__global__ void forward_expectation_cuda_kernel(
    scalar_t * const __restrict__ x_out,
    scalar_t * const __restrict__ pmf,
    const int64_t len_x,
    const scalar_t * __restrict__ q,
    const int64_t len_t
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    if (ix < len_x)
    {
        // pre-compute row offset from the beginning of the `pmf` array
        int row_offset = ix * PLUS_1(len_t);

        scalar_t sum = 0.0f;
        for (int iq = 0; iq < PLUS_1(len_t); ++iq)
        {
            sum += q[iq] * pmf[row_offset + iq];
        }

        x_out[ix] = sum;
    }
    else  // I am out of bounds!
    {
        return;
    }
}


template <typename scalar_t>
__global__ void forward_mode_cuda_kernel(
    scalar_t * const __restrict__ x_out,
    scalar_t * const __restrict__ pmf,
    const int64_t len_x,
    const scalar_t * __restrict__ q,
    const int64_t len_t
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    if (ix < len_x)
    {
        // pre-compute row offset from the beginning of the `pmf` array
        int row_offset = ix * PLUS_1(len_t);

        // find the bin that contains the greatest probability mass
        int32_t argmax = 0;
        scalar_t max = pmf[row_offset + argmax];
        for (int iq = 1; iq < PLUS_1(len_t); ++iq)
        {
            if (max <= pmf[row_offset + iq])
            {
                argmax = iq;
                max = pmf[row_offset + argmax];
            }
        }

        x_out[ix] = q[argmax];
    }
    else  // I am out of bounds!
    {
        return;
    }
}


template <typename scalar_t>
__global__ void forward_random_cuda_kernel(
    scalar_t * const __restrict__ x_out,
    scalar_t * const __restrict__ us,     // samples from the uniform over [0, 1)
    scalar_t * const __restrict__ pmf,
    const int64_t len_x,
    const scalar_t * __restrict__ q,
    const int64_t len_t
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    if (ix < len_x)
    {
        // pre-compute row offset from the beginning of the `pmf` array
        int row_offset = ix * PLUS_1(len_t);

        // Each row in `pmf` sums to 1 due to the normalisation property.
        // I imagine to have a segment for each bin, the length of the
        // segment being proportional to the probability mass in the bin. If I
        // glue the segments in a row, selecting a random number in [0, 1)
        // will generate a point falling in exactly one of the segments, i.e.,
        // in one of the bins.
        scalar_t u = us[ix];
        scalar_t cum_prob = 0.0f;
        int idx = -1;
        for (int iq = 0; iq < PLUS_1(len_t); ++iq)
        {
            cum_prob += pmf[row_offset + iq];
            if ((idx < 0) && (u < cum_prob))  // I work under the assumption that the cumulative probability is monotone
            {
                idx = iq;  // setting this integer to positive acts as a flag signaling that the sampled bin has been found
            }
        }

        x_out[ix] = q[idx];
    }
    else  // I am out of bounds!
    {
        return;
    }
}
