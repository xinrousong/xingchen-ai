# -*- coding: utf-8 -*-
"""工具链冒烟测试：先编一个 C++ 扩展，再编一个 CUDA 扩展，确认整条链路可用。

python_embeded 是 embeddable 发行版，没有 include/ 和 libs/，
所以显式借用同版本系统 CPython（3.13.14）的头文件和 import lib。
"""
import sys
import time

import torch
from torch.utils.cpp_extension import load_inline

PY_INC = r"C:\Users\Xinrou\AppData\Local\Programs\Python\Python313\include"
PY_LIB = r"C:\Users\Xinrou\AppData\Local\Programs\Python\Python313\libs"
COMMON = dict(
    extra_include_paths=[PY_INC],
    extra_ldflags=[f"/LIBPATH:{PY_LIB}"],
    # CUDA 13.4 的 CCCL 头要求 MSVC 用「标准一致」的预处理器，否则 #error
    extra_cflags=["/Zc:preprocessor"],
    extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"],
    verbose=False,
)

t0 = time.time()
cpp = load_inline(
    name="xc_smoke_cpp",
    cpp_sources="torch::Tensor twice(torch::Tensor x){ return x * 2; }",
    functions=["twice"],
    with_cuda=False,
    **COMMON,
)
print(f"[C++]  build OK in {time.time()-t0:.1f}s -> {cpp.twice(torch.ones(3)).tolist()}", flush=True)

cuda_src = r"""
#include <cuda_runtime.h>

__global__ void add_one_kernel(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}

torch::Tensor add_one(torch::Tensor x) {
    auto y = x.contiguous();
    int n = (int)y.numel();
    int threads = 256, blocks = (n + threads - 1) / threads;
    add_one_kernel<<<blocks, threads>>>(y.data_ptr<float>(), n);
    return y;
}
"""

t0 = time.time()
cu = load_inline(
    name="xc_smoke_cuda",
    cpp_sources="torch::Tensor add_one(torch::Tensor x);",
    cuda_sources=cuda_src,
    functions=["add_one"],
    with_cuda=True,
    **COMMON,
)
x = torch.zeros(8, device="cuda")
print(f"[CUDA] build OK in {time.time()-t0:.1f}s -> {cu.add_one(x).tolist()}", flush=True)
print("TOOLCHAIN READY", flush=True)
