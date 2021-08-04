/*  
 *  uniform_cuda_kernel.cu
 *  
 *  Author(s):
 *  Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
 *  
 *  Copyright (c) 2020-2021 ETH Zurich. All rights reserved.
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

#include <torch/extension.h>
#include <vector>

// #include <stdio.h>  // for debugging

#include <cuda.h>
#include <cuda_runtime.h>


#define THREADS_PER_BLOCK 1024

#define PLUS_1(x) (x + 1)
#define ABS(x) ((x < 0.0f) ? -x : x)
#define CLAMP_0_1(x) ((x > 1.0f) ? 1.0f : ((x < 0.0f) ?  0.0f : x))


// definitions of CUDA kernels (executed on: GPU)


template <typename scalar_t>
__global__ void uniform_forward_cuda_kernel_pmf(
    scalar_t * const __restrict__ pmf,
    scalar_t * const __restrict__ x_in,
    const int64_t len_x,
    const scalar_t * __restrict__ t,      // thresholds
    const int64_t len_t,
    const scalar_t * __restrict__ mu,     // mean
    const scalar_t * __restrict__ sigma,  // standard deviation
    const scalar_t * __restrict__ training
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    if (ix < len_x)
    {
        // pre-compute row offset from the beginning of the `pmf` array
        int row_offset = ix * PLUS_1(len_t);

        // compute shifted thresholds
        for (int it = 0; it < PLUS_1(len_t); ++it)
        {
            pmf[row_offset + it + 1] = x_in[ix] - *mu - t[it];
        }

        // compute CDF
        for (int it = 0; it < PLUS1(len_t); ++it)
        {
            if (it == 0)
            {
                pmf[row_offset + it] = 1.0f;
            }
            else
            {
                if (*training && (*sigma != 0.0f))
                {
                    scalar_t sigma_inv = 1.0 / (*sigma);
                    pmf[row_offset + it] = CLAMP_0_1(0.5f * (temp[row_offset + it] * sigma_inv + 1.0f));
                }
                else
                {
                    pmf[row_offset + it] = (scalar_t) (pmf[row_offset + it] >= 0.0f);
                }
            }
        }

        // compute the probability mass in each bin
        for (int iq = 0; iq < PLUS_1(len_t) - 1; ++iq)
        {
            pmf[row_offset + iq] = pmf[row_offset + iq] - pmf[row_offset + iq + 1];
        }
        // the last bin (with index `row_offset + len_t`) would have mass `pmf[row_offset + len_t] - 0.0f`, so it's not necessary to compute it!
    }
    else  // I am out of bounds!
    {
        return;
    }
}


template <typename scalar_t>
__global__ void uniform_forward_cuda_kernel_expectation(
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
__global__ void uniform_forward_cuda_kernel_mode(
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
            if (max < temp[row_offset + iq])
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
__global__ void uniform_forward_cuda_kernel_random(
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
        for (int iq = 0; iq < PLUS_1(iq); ++iq)
        {
            cum_prob += pmf[row_offset + iq];
            if ((idx < 0) && (u < cum_prob))  // I work under the assumption that the cumulative probability is monotone
            {
                idx = it;  // setting this integer to positive acts as a flag signaling that the sampled bin has been found
            }
        }

        x_out[ix] = q[idx];
    }
    else  // I am out of bounds!
    {
        return;
    }
}


template <typename scalar_t>
__global__ void uniform_backward_cuda_kernel(
    scalar_t * const __restrict__ grad_out,
    const scalar_t * __restrict__ grad_in,
    const scalar_t * __restrict__ x_in,
    const int64_t len_x,
    const scalar_t * __restrict__ q,
    const scalar_t * __restrict__ t,
    const int64_t len_t,
    const scalar_t * __restrict__ mu,
    const scalar_t * __restrict__ sigma
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < len_x)
    {
        scalar_t sum = 0.0f;

        for (int it = 0; it < len_t; ++it)
        {
            // input position relative to the threshold
            scalar_t x_minus_t = x_in[ix] - t[it] - *mu;

            // the derivative of the expected (i.e., regularised) step function is the PDF of the uniform distribution
            scalar_t pdf;
            if (*sigma != 0.0f)
            {
                scalar_t sigma_inv = 1.0f / (*sigma);
                scalar_t local_derivative = (scalar_t) (ABS(x_minus_t) <= (*sigma));
                pdf = 0.5f * sigma_inv * local_derivative;
            }
            else
            {
                pdf = 0.0f;  // no noise, no gradient!
            }

            // dilate and accumulate expected derivative
            scalar_t dq = q[it + 1] - q[it];
            sum += dq * pdf;
        }

        // compose gradients
        grad_out[ix] = sum * grad_in[ix];
    }
    else  // I am out of bounds!
    {
        return;
    }
}


// definitions of C++\CUDA interface (executed on: CPU)
// goals:
//   * allocate GPU memory for the output;
//   * define the parameters for the GPU kernel;
//   * call the kernel;

torch::Tensor uniform_forward_cuda_dispatch(
    torch::Tensor x_in,
    torch::Tensor q,
    torch::Tensor t,
    torch::Tensor mu,
    torch::Tensor sigma,
    torch::Tensor strategy,
    torch::Tensor training
)
{
    auto x_out = torch::zeros_like(x_in);
    auto pmf = torch::zeros({x_in.numel(), PLUS2(t.numel())}, torch::TensorOptions().dtype(x_in.dtype()).device(x_in.device()));

    const dim3 blocks((x_in.numel() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // compute PMF over bins (i.e., the quantization levels)
    AT_DISPATCH_FLOATING_TYPES(
        x_in.type(),
        "uniform_forward_cuda_kernel_pmf",
        ([&] {
            uniform_forward_cuda_kernel_pmf<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
                pmf.data_ptr<scalar_t>(),
                x_in.data_ptr<scalar_t>(),
                x_in.numel(),
                t.data_ptr<scalar_t>(),
                t.numel(),
                mu.data_ptr<scalar_t>(),
                sigma.data_ptr<scalar_t>(),
                training.data_ptr<scalar_t>()
            );
        })
    );

    switch(strategy.item())
    {
        case 0:  // expectation
            AT_DISPATCH_FLOATING_TYPES(
                x_in.type(),
                "uniform_forward_cuda_kernel_expectation",
                ([&] {
                    uniform_forward_cuda_kernel_expectation<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
                        x_out.data_ptr<scalar_t>(),
                        pmf.data_ptr<scalar_t>(),
                        x_in.numel(),
                        q.data_ptr<scalar_t>(),
                        t.numel()
                    );
                })
            );

        case 1:  // argmax sampling (i.e., mode)
            AT_DISPATCH_FLOATING_TYPES(
                x_in.type(),
                "uniform_forward_cuda_kernel_mode",
                ([&] {
                    uniform_forward_cuda_kernel_mode<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
                        x_out.data_ptr<scalar_t>(),
                        pmf.data_ptr<scalar_t>(),
                        x_in.numel(),
                        q.data_ptr<scalar_t>(),
                        t.numel()
                    );
                })
            );

        case 2:  // random sampling
            auto us = torch::rand_like(x_in);
            AT_DISPATCH_FLOATING_TYPES(
                x_in.type(),
                "uniform_forward_cuda_kernel_random",
                ([&] {
                    uniform_forward_cuda_kernel_random<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
                        x_out.data_ptr<scalar_t>(),
                        us.data_ptr<scalar_t>(),
                        pmf.data_ptr<scalar_t>(),
                        x_in.numel(),
                        q.data_ptr<scalar_t>(),
                        t.numel()
                    );
                })
            );

    }



    return x_out;
}


torch::Tensor uniform_backward_cuda_dispatch(
    torch::Tensor grad_in,
    torch::Tensor x_in,
    torch::Tensor q,
    torch::Tensor t,
    torch::Tensor mu,
    torch::Tensor sigma
)
{
    auto grad_out = torch::zeros_like(x_in);
    const dim3 blocks((x_in.numel() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // see PyTorch's ATen ("A TENsor") library; the full path from PyTorch GitHub repo's main is `aten/src/ATen/Dispatch.h`
    AT_DISPATCH_FLOATING_TYPES(
        x_in.type(),
        "uniform_backward_cuda",
        ([&] {
            uniform_backward_cuda_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
                grad_out.data_ptr<scalar_t>(),
                grad_in.data_ptr<scalar_t>(),
                x_in.data_ptr<scalar_t>(),
                x_in.numel(),
                q.data_ptr<scalar_t>(),
                t.data_ptr<scalar_t>(),
                t.numel(),
                mu.data_ptr<scalar_t>(),
                sigma.data_ptr<scalar_t>()
            );
        })
    );

    return grad_out;
}

