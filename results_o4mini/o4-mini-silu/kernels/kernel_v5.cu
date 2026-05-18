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
  float logistic = __fdividef(1.0f, 1.0f + e);
  return __fmaf_rn(x, logistic, 0.0f);
}

// ----- vector kernel -----
__global__ __launch_bounds__(BLOCK_SIZE, 4)
void silu_vec_kernel(
    const float* __restrict__ in,
    float* __restrict__ out,
    int B,
    int D) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int total = B * D;
  for (int idx = tid; idx < total; idx += stride) {
    int b = idx / D;
    int d = idx % D;
    float x = in[b * (2 * D) + d];
    float g = in[b * (2 * D) + D + d];
    out[b * D + d] = silu_f(x) * g;
  }
}

// ----- C++ entry -----
torch::Tensor silu_and_mul(torch::Tensor input, int64_t D) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  auto B = input.size(0);
  TORCH_CHECK(input.size(1) == 2 * D, "input.shape[-1] must be 2*D");
  auto output = torch::empty({B, D}, input.options());
  dim3 block(BLOCK_SIZE);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid((B * D + BLOCK_SIZE - 1) / BLOCK_SIZE);
  silu_vec_kernel<<<grid, block, 0, stream>>>(
      input.data_ptr<float>(), output.data_ptr<float>(), (int)B, (int)D);
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kernel launch failed");
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sgl_silu_mul", &silu_and_mul, "SiLU(x) * gate",
        py::arg("input"), py::arg("dim"));
}