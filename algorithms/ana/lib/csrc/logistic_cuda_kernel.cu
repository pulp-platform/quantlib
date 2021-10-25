/*  
 *  logistic_cuda_kernel.cu
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

#include <torch/extension.h>
#include <vector>

// #include <stdio.h>  // for debug

#include <cuda.h>
#include <cuda_runtime.h>

#include "forward.h"


#define THREADS_PER_BLOCK 1024


// definitions of CUDA kernels (executed on: GPU)

template <typename scalar_t>
__global__ void logistic_forward_pmf_cuda_kernel(
    scalar_t * const __restrict__ pmf,
    const scalar_t * __restrict__ x_in,
    const int64_t len_x,
    const scalar_t * __restrict__ t,
    const int64_t len_t,
    const scalar_t * __restrict__ mi,
    const scalar_t * __restrict__ sigma,
    const scalar_t * __restrict__ training
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;

    if (ix < len_x)
    {
        // pre-compute row offset from the beginning of the `pmf` array
        int row_offset = ix * PLUS_1(len_t);

        // compute shifted thresholds
        for (int it = 0; it < len_t; ++it)
        {
            pmf[row_offset + it + 1] = x_in[ix] - *mi - t[it];
        }

        // compute CDF
        for (int it = 0; it < PLUS_1(len_t); ++it)
        {
            if (it == 0)
            {
                pmf[row_offset + it] = 1.0f;
            }
            else
            {
                if (*training && (*sigma != 0.0f))
                {
                    scalar_t sigma_inv = 1.0f / (*sigma);
                    scalar_t shifted_x_minus_t_over_s = pmf[row_offset + it] * sigma_inv;
                    scalar_t exp_shifted_x_minus_t_over_s = expf(-1.0f * shifted_x_minus_t_over_s);
                    pmf[row_offset + it] = 1.0f / (1.0f + exp_shifted_x_minus_t_over_s);
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
__global__ void logistic_backward_cuda_kernel(
    scalar_t * const __restrict__ grad_out,
    const scalar_t * __restrict__ grad_in,
    const scalar_t * __restrict__ x_in,
    const int64_t len_x,
    const scalar_t * __restrict__ q,
    const scalar_t * __restrict__ t,
    const int64_t len_t,
    const scalar_t * __restrict__ mi,
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
            scalar_t shifted_x_minus_t  = x_in[ix] - *mi - t[it];

            // the derivative of the expected (i.e., regularised) step function is the PDF of the logistic distribution
            scalar_t pdf;
            if (*sigma != 0.0f)
            {
                scalar_t sigma_inv = 1.0f / (*sigma);
                scalar_t shifted_x_minus_t_over_s = shifted_x_minus_t * sigma_inv;
                scalar_t exp_shifted_x_minus_t_over_s = expf(-1.0f * shifted_x_minus_t_over_s);
                scalar_t cdf = 1.0f / (1.0f + exp_shifted_x_minus_t_over_s);
                pdf = cdf * (1.0f - cdf) * sigma_inv;
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

torch::Tensor logistic_forward_cuda_dispatch(
    torch::Tensor x_in,
    torch::Tensor q,
    torch::Tensor t,
    torch::Tensor mi,
    torch::Tensor sigma,
    torch::Tensor strategy,
    torch::Tensor training
)
{
    auto x_out = torch::zeros_like(x_in);
    auto pmf = torch::zeros({x_in.numel(), PLUS_1(t.numel())}, torch::TensorOptions().dtype(x_in.dtype()).device(x_in.device()));

    const dim3 blocks((x_in.numel() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    // compute PMF over bins (i.e., the quantization levels)
    AT_DISPATCH_FLOATING_TYPES(
        x_in.type(),
        "logistic_forward_pmf_cuda_kernel",
        ([&] {
            logistic_forward_pmf_cuda_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
                pmf.data_ptr<scalar_t>(),
                x_in.data_ptr<scalar_t>(),
                x_in.numel(),
                t.data_ptr<scalar_t>(),
                t.numel(),
                mi.data_ptr<scalar_t>(),
                sigma.data_ptr<scalar_t>(),
                training.data_ptr<scalar_t>()
            );
        })
    );

    switch(strategy.item<int32_t>())  // how to read tensor's content using the C++ API: https://stackoverflow.com/a/54208912
    {
        case 0:  // expectation
            AT_DISPATCH_FLOATING_TYPES(
                x_in.type(),
                "logistic_forward_expectation_cuda_kernel",
                ([&] {
                    forward_expectation_cuda_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
                        x_out.data_ptr<scalar_t>(),
                        pmf.data_ptr<scalar_t>(),
                        x_in.numel(),
                        q.data_ptr<scalar_t>(),
                        t.numel()
                    );
                })
            );
            break;

        case 1:  // argmax sampling (i.e., mode)
            AT_DISPATCH_FLOATING_TYPES(
                x_in.type(),
                "logistic_forward_mode_cuda_kernel",
                ([&] {
                    forward_mode_cuda_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
                        x_out.data_ptr<scalar_t>(),
                        pmf.data_ptr<scalar_t>(),
                        x_in.numel(),
                        q.data_ptr<scalar_t>(),
                        t.numel()
                    );
                })
            );
            break;

        case 2:  // random sampling
            auto us = torch::rand_like(x_in);
            AT_DISPATCH_FLOATING_TYPES(
                x_in.type(),
                "logistic_forward_random_cuda_kernel",
                ([&] {
                    forward_random_cuda_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
                        x_out.data_ptr<scalar_t>(),
                        us.data_ptr<scalar_t>(),
                        pmf.data_ptr<scalar_t>(),
                        x_in.numel(),
                        q.data_ptr<scalar_t>(),
                        t.numel()
                    );
                })
            );
            break;

    }

    return x_out;
}


torch::Tensor logistic_backward_cuda_dispatch(
    torch::Tensor grad_in,
    torch::Tensor x_in,
    torch::Tensor q,
    torch::Tensor t,
    torch::Tensor mi,
    torch::Tensor sigma
)
{
    auto grad_out = torch::zeros_like(x_in);
    const dim3 blocks((x_in.numel() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(
        x_in.type(),
        "logistic_backward_cuda",
        ([&] {
            logistic_backward_cuda_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
                grad_out.data_ptr<scalar_t>(),
                grad_in.data_ptr<scalar_t>(),
                x_in.data_ptr<scalar_t>(),
                x_in.numel(),
                q.data_ptr<scalar_t>(),
                t.data_ptr<scalar_t>(),
                t.numel(),
                mi.data_ptr<scalar_t>(),
                sigma.data_ptr<scalar_t>()
            );
        })
    );

    return grad_out;
}
