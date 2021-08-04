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

// #include <stdio.h>  // for debug

#include <cuda.h>
#include <cuda_runtime.h>


#define THREADS_PER_BLOCK 1024

#define PLUS2(x) (x + 2)
#define ABS(x) ((x < 0.0f) ? -x : x)
#define CLAMP_0_1(x) ((x > 1.0f) ? 1.0f : ((x < 0.0f) ?  0.0f : x))


// definitions of CUDA kernels (executed on: GPU)

template <typename scalar_t>
__global__ void uniform_forward_cuda_kernel(
    scalar_t * const __restrict__ x_out,
    scalar_t * const __restrict__ seeds,
    scalar_t * const __restrict__ temp,
    const scalar_t * __restrict__ x_in,
    const int64_t len_x,
    const scalar_t * __restrict__ q,
    const scalar_t * __restrict__ t,
    const int64_t len_t,
    const scalar_t * __restrict__ fmu,
    const scalar_t * __restrict__ fsigma,
    const int32_t * __restrict__ strategy,
    const scalar_t * __restrict__ training
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < len_x)
    {
        // precompute row offset
        int row_offset = ix * PLUS2(len_t);

        // compute shifted thresholds
        for (int it = (0 + 1); it < (len_t + 1); ++it)
        {
            temp[row_offset + it] = t[it] - (x_in[ix] - *fmu);
        }

        // compute CDF
        for (int it = 0; it < PLUS2(len_t); ++it)
        {
            if (it == 0)
            {
                temp[row_offset + it] = 0.0f;
            }
            else if (it == (PLUS2(len_t) - 1))
            {
                temp[row_offset + it] = 1.0f;
            }
            else
            {
                if (*training && (*fsigma != 0.0f))
                {
                    scalar_t sigma_inv = 1.0 / (*fsigma);
                    temp[row_offset + it] = CLAMP_0_1(0.5f * (temp[row_offset + it] * sigma_inv + 1.0f));
                }
                else
                {
                    temp[row_offset + it] = (scalar_t) (temp[row_offset + it] >= 0.0f);
                }
            }
        }

        // compute probability mass in each bin
        for (int it = 0; it < len_t + 1; ++it)
        {
            temp[row_offset + it] = temp[row_offset + it + 1] - temp[row_offset + it];
        }

        // compute outputs
        if (*strategy == 0)  // expectation
        {
            scalar_t sum = 0.0f;

            for (int it = 0; it < len_t + 1; ++it)
            {
                sum += q[it] * temp[row_offset + it];
            }

            x_out[ix] = sum;
        }
        else if (*strategy == 1)  // argmax sampling (i.e., mode)
        {
            int argmax = 0;
            scalar_t max = temp[row_offset + argmax];

            for (int it = 1; it < (len_t + 1); ++it)
            {
                if (max < temp[row_offset + it])
                {
                    argmax = it;
                    max = temp[row_offset + argmax];
                }
            }

            x_out[ix] = q[argmax];
        }
        else if (*strategy == 2)  // stochastic sampling
        {
            scalar_t cum_prob = 0.0f;
            scalar_t u = seeds[ix];
            bool found = false;
            int idx = -1;

            for (int it = 0; it < (len_t + 1); ++it)
            {
                cum_prob += temp[row_offset + it];
                if ((!found) && (u < cum_prob))
                {
                    idx = it;
                    found = true;
                }
            }

            x_out[ix] = q[idx];
        }
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
    const scalar_t * __restrict__ bmu,
    const scalar_t * __restrict__ bsigma
)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    if (ix < len_x)
    {
        scalar_t sum = 0.0f;

        for (int it = 0; it < len_t; ++it)
        {
            // input position relative to the threshold
            scalar_t x_minus_t = x_in[ix] - t[it] - *bmu;

            // the derivative of the expected (i.e., regularised) step function is the PDF of the uniform distribution
            scalar_t pdf;
            if (*bsigma != 0.0f)
            {
                scalar_t sigma_inv = 1.0f / (*bsigma);
                scalar_t local_derivative = (scalar_t) (ABS(x_minus_t) <= (*bsigma));
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
    torch::Tensor fmu,
    torch::Tensor fsigma,
    torch::Tensor strategy,
    torch::Tensor training
)
{
    auto x_out = torch::zeros_like(x_in);

    auto temp = torch::zeros({x_in.numel(), t.numel() + 2}, torch::TensorOptions().dtype(x_in.dtype()).device(x_in.device()));
    auto seeds = torch::rand_like(x_in);

    const dim3 blocks((x_in.numel() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    AT_DISPATCH_FLOATING_TYPES(
        x_in.type(),
        "uniform_forward_cuda",
        ([&] {
            uniform_forward_cuda_kernel<scalar_t><<<blocks, THREADS_PER_BLOCK>>>(
                x_out.data_ptr<scalar_t>(),
                seeds.data_ptr<scalar_t>(),
                temp.data_ptr<scalar_t>(),
                x_in.data_ptr<scalar_t>(),
                x_in.numel(),
                q.data_ptr<scalar_t>(),
                t.data_ptr<scalar_t>(),
                t.numel(),
                fmu.data_ptr<scalar_t>(),
                fsigma.data_ptr<scalar_t>(),
                strategy.data_ptr<int32_t>(),
                training.data_ptr<scalar_t>()
            );
        })
    );

    return x_out;
}


torch::Tensor uniform_backward_cuda_dispatch(
    torch::Tensor grad_in,
    torch::Tensor x_in,
    torch::Tensor q,
    torch::Tensor t,
    torch::Tensor bmu,
    torch::Tensor bsigma
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
                bmu.data_ptr<scalar_t>(),
                bsigma.data_ptr<scalar_t>()
            );
        })
    );

    return grad_out;
}

