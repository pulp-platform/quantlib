/*  
 *  normal_cuda.cpp
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


// declarations of C++\CUDA interface (executed on: CPU)

torch::Tensor normal_forward_cuda_dispatch(
    torch::Tensor x_in,
    torch::Tensor q,
    torch::Tensor t,
    torch::Tensor mi,
    torch::Tensor sigma,
    torch::Tensor strategy,
    torch::Tensor training
);

torch::Tensor normal_backward_cuda_dispatch(
    torch::Tensor grad_in,
    torch::Tensor x_in,
    torch::Tensor q,
    torch::Tensor t,
    torch::Tensor mi,
    torch::Tensor sigma
);


// definitions of C++ wrappers (executed on: CPU)
// goals:
//   * check that the memory layout of tensors allocated on GPU memory is correct
//   * call the dispatcher

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor normal_forward_cuda(
    torch::Tensor x_in,
    torch::Tensor q,
    torch::Tensor t,
    torch::Tensor mi,
    torch::Tensor sigma,
    torch::Tensor strategy,
    torch::Tensor training
)
{
    CHECK_INPUT(x_in);
    CHECK_INPUT(q);
    CHECK_INPUT(t);
    CHECK_INPUT(mi);
    CHECK_INPUT(sigma);
    CHECK_INPUT(strategy);
    CHECK_INPUT(training);

    return normal_forward_cuda_dispatch(x_in, q, t, mi, sigma, strategy, training);
}


torch::Tensor normal_backward_cuda(
    torch::Tensor grad_in,
    torch::Tensor x_in,
    torch::Tensor q,
    torch::Tensor t,
    torch::Tensor mi,
    torch::Tensor sigma
)
{
    CHECK_INPUT(grad_in);
    CHECK_INPUT(x_in);
    CHECK_INPUT(q);
    CHECK_INPUT(t);
    CHECK_INPUT(mi);
    CHECK_INPUT(sigma);

    return normal_backward_cuda_dispatch(grad_in, x_in, q, t, mi, sigma);
}


// compile into a Python module

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &normal_forward_cuda, "ANA normal noise forward (CUDA)");
    m.def("backward", &normal_backward_cuda, "ANA normal noise backward (CUDA)");
}
