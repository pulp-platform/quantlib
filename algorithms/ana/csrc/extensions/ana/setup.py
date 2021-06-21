from setuptools import setup
from torch.utils import cpp_extension

setup(name='ana',
      ext_modules=[
          cpp_extension.CUDAExtension(name='ana_uniform_cuda',    sources=['uniform_cuda.cpp',    'uniform_cuda_kernel.cu']),
          cpp_extension.CUDAExtension(name='ana_triangular_cuda', sources=['triangular_cuda.cpp', 'triangular_cuda_kernel.cu']),
          cpp_extension.CUDAExtension(name='ana_normal_cuda',     sources=['normal_cuda.cpp',     'normal_cuda_kernel.cu']),
          cpp_extension.CUDAExtension(name='ana_logistic_cuda',   sources=['logistic_cuda.cpp',   'logistic_cuda_kernel.cu']),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
