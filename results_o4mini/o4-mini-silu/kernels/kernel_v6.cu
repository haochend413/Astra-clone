#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// ----- math -----
__device__ __forceinline__ float silu_f(float x) {
    float e = __expf(-x);
    float logistic = __frcp_rn(1.0f + e);
    return __fmaf_rn(x, logistic, 0.0f);
}

// ----- vector kernel -----
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void silu_vec_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int total) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total2 = total >> 1;

    const float2* __restrict__ x2 = reinterpret_cast<const float2*>(in);
    const float2* __restrict__ g2 = reinterpret_cast<const float2*>(in + total);
    float2* __restrict__ o2 = reinterpret_cast<float2*>(out);

    int unroll_stride = stride * 2;
    for (int idx = tid; idx < total2; idx += unroll_stride) {
        // first vector
        float2 xv = __ldg(&x2[idx]);
        float2 gv = __ldg(&g2[idx]);
        float2 rv;
        float x0 = xv.x;
        rv.x = silu_f(x0) * gv.x;
        float x1 = xv.y;
        rv.y = silu_f(x1) * gv.y;
        o2[idx] = rv;
        // unrolled second vector
        int idx2 = idx + stride;
        if (idx2 < total2) {
            float2 xv1 = __ldg(&x2[idx2]);
            float2 gv1 = __ldg(&g2[idx2]);
            float2 rv1;
            float y0 = xv1.x;
            rv1.x = silu_f(y0) * gv1.x;
            float y1 = xv1.y;
            rv1.y = silu_f(y1) * gv1.y;
            o2[idx2] = rv1;
        }
    }

    // handle tail element if odd
    if (total & 1) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            int idx = total - 1;
            float x = in[idx];
            float g = in[total + idx];
            out[idx] = silu_f(x) * g;
        }
    }
}

// ----- C++ entry -----
torch::Tensor silu_and_mul(torch::Tensor input, int64_t D) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    auto B = input.size(0);
    TORCH_CHECK(input.size(1) == 2 * D, "input.shape[-1] must be 2*D");
    auto output = torch::empty({B, D}, input.options());
    int total = static_cast<int>(B * D);
    dim3 block(BLOCK_SIZE);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int grid_size = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(grid_size);
    silu_vec_kernel<<<grid, block, 0, stream>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total
    );
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kernel launch failed");
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgl_silu_mul", &silu_and_mul, "SiLU(x) * gate",
          py::arg("input"), py::arg("dim"));
}