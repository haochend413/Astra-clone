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
__device__ __forceinline__ double silu_d(double x) {
  double e = exp(-x);
  double logistic = 1.0/(1.0 + e);
  return fma(x, logistic, 0.0);
}

// ----- dtype convert helpers -----
template <typename T> __device__ __forceinline__ float to_float(T x);
template <> __device__ __forceinline__ float to_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float to_float<at::Half>(at::Half x) {
  return __half2float(*(const __half*)&x);
}
template <> __device__ __forceinline__ float to_float<at::BFloat16>(at::BFloat16 x) {
  return __bfloat162float(*(const __nv_bfloat16*)&x);
}
template <> __device__ __forceinline__ float to_float<double>(double x) { return (float)x; }

template <typename T> __device__ __forceinline__ T from_float(float x);
template <> __device__ __forceinline__ float from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ at::Half from_float<at::Half>(float x) {
  __half h = __float2half_rn(x);
  return *(at::Half*)&h;
}
template <> __device__ __forceinline__ at::BFloat16 from_float<at::BFloat16>(float x) {
  __nv_bfloat16 b = __float2bfloat16(x);
  return *(at::BFloat16*)&b;
}
template <> __device__ __forceinline__ double from_float<double>(float x) { return (double)x; }

// ----- vector type mapping -----
template <typename T> struct VecType;
template <> struct VecType<float> { using type = float2; };
template <> struct VecType<double> { using type = double2; };
template <> struct VecType<at::Half> { using type = __half2; };

// ----- Apply SiLU and multiply for vectors -----
__device__ __forceinline__ float2 apply_silu_mul(float2 x, float2 g) {
  float y0 = silu_f(x.x) * g.x;
  float y1 = silu_f(x.y) * g.y;
  return make_float2(y0, y1);
}
__device__ __forceinline__ double2 apply_silu_mul(double2 x, double2 g) {
  double y0 = silu_d(x.x) * g.x;
  double y1 = silu_d(x.y) * g.y;
  return make_double2(y0, y1);
}
__device__ __forceinline__ __half2 apply_silu_mul(__half2 x, __half2 g) {
  float2 xf = __half22float2(x);
  float2 gf = __half22float2(g);
  float2 yf;
  yf.x = silu_f(xf.x) * gf.x;
  yf.y = silu_f(xf.y) * gf.y;
  return __float22half2_rn(yf);
}

// ----- vector kernel -----
template <typename scalar_t>
__global__ __launch_bounds__(BLOCK_SIZE)
void silu_and_mul_vec_kernel(
    const scalar_t* __restrict__ in,
    scalar_t* __restrict__ out,
    int64_t B,
    int64_t nvec) {
  using Vec = typename VecType<scalar_t>::type;
  const Vec* in_vec = reinterpret_cast<const Vec*>(in);
  Vec* out_vec = reinterpret_cast<Vec*>(out);
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  int64_t vec_per_batch = nvec;
  int64_t total = B * vec_per_batch;
  #pragma unroll
  for (int64_t idx = tid; idx < total; idx += stride) {
    int64_t b = idx / vec_per_batch;
    int64_t v = idx % vec_per_batch;
    int64_t base = b * (2 * vec_per_batch);
    Vec x = in_vec[base + v];
    Vec g = in_vec[base + vec_per_batch + v];
    out_vec[b * vec_per_batch + v] = apply_silu_mul(x, g);
  }
}

// ----- scalar kernel for BFloat16 -----
template <typename scalar_t>
__global__ __launch_bounds__(BLOCK_SIZE)
void silu_and_mul_scalar_kernel(
    const scalar_t* __restrict__ in,
    scalar_t* __restrict__ out,
    int64_t B,
    int64_t D) {
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  int64_t total = B * D;
  #pragma unroll
  for (int64_t idx = tid; idx < total; idx += stride) {
    int64_t b = idx / D;
    int64_t d = idx % D;
    float xv = to_float<scalar_t>(in[b * (2 * D) + d]);
    float gv = to_float<scalar_t>(in[b * (2 * D) + D + d]);
    float y = silu_f(xv) * gv;
    out[b * D + d] = from_float<scalar_t>(y);
  }
}

// ----- C++ entry -----
void silu_and_mul(torch::Tensor input, torch::Tensor output, int64_t D) {
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(input.dim() == 2 && output.dim() == 2, "input/out must be 2D");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "tensors must be contiguous");
  TORCH_CHECK(D > 0 && D % 2 == 0, "dim must be positive even");
  TORCH_CHECK(input.size(0) == output.size(0), "batch size mismatch");
  TORCH_CHECK(input.size(1) == 2 * D, "input.shape[-1] must be 2*D");
  TORCH_CHECK(output.size(1) == D, "output.shape[-1] must be D");

  int64_t B = input.size(0);
  int64_t nvec = D / 2;
  dim3 block(BLOCK_SIZE);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (input.scalar_type()) {
    case torch::kFloat: {
      dim3 grid((B * nvec + BLOCK_SIZE - 1) / BLOCK_SIZE);
      silu_and_mul_vec_kernel<float><<<grid, block, 0, stream>>>(
          input.data_ptr<float>(), output.data_ptr<float>(), B, nvec);
      break;
    }
    case torch::kHalf: {
      dim3 grid((B * nvec + BLOCK_SIZE - 1) / BLOCK_SIZE);
      silu_and_mul_vec_kernel<at::Half><<<grid, block, 0, stream>>>(
          input.data_ptr<at::Half>(), output.data_ptr<at::Half>(), B, nvec);
      break;
    }
    case torch::kDouble: {
      dim3 grid((B * nvec + BLOCK_SIZE - 1) / BLOCK_SIZE);
      silu_and_mul_vec_kernel<double><<<grid, block, 0, stream>>>(
          input.data_ptr<double>(), output.data_ptr<double>(), B, nvec);
      break;
    }
    case torch::kBFloat16: {
      dim3 grid((B * D + BLOCK_SIZE - 1) / BLOCK_SIZE);
      silu_and_mul_scalar_kernel<at::BFloat16><<<grid, block, 0, stream>>>(
          input.data_ptr<at::BFloat16>(), output.data_ptr<at::BFloat16>(), B, D);
      break;
    }
    default:
      TORCH_CHECK(false, "sgl_silu_mul: unsupported dtype");
  }
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "kernel launch failed");
}

// ----- PyBind11 -----
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sgl_silu_mul", &silu_and_mul, "SiLU(x) * gate",
        py::arg("input"), py::arg("output"), py::arg("dim"));
}