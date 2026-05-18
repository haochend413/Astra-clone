#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/util/Exception.h>
#include <cmath>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while (0)
#define CHECK_DIM(D, x) TORCH_CHECK((x).dim() == (D), #x " must have dim " #D)
#define CHECK_SHAPE(a, b) TORCH_CHECK((a).sizes() == (b).sizes(), #a " and " #b " must have the same shape")
#define CHECK_EQ(x, y) TORCH_CHECK((x) == (y), #x " must equal " #y)

#define NEXT_MULTIPLE(x, y) (((x) + (y) - 1) / (y) * (y))
static const uint32_t MAX_THREADS = 256;

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_float(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T> __device__ __forceinline__ T cast_from_float(float x);
template <> __device__ __forceinline__ float cast_from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __half cast_from_float<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 cast_from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// Kernels unchanged from v2
__global__ __launch_bounds__(MAX_THREADS) void merge_state_kernel_v3_float(
    const float* __restrict__ v_a,
    const float* __restrict__ s_a,
    const float* __restrict__ v_b,
    const float* __restrict__ s_b,
    float* __restrict__ v_out,
    float* __restrict__ s_out,
    uint32_t head_dim) {
  unsigned mask = 0xffffffffu;
  uint32_t seq = blockIdx.x;
  uint32_t head = blockIdx.y;
  uint32_t lane = threadIdx.x;
  uint32_t stride = blockDim.x;
  uint64_t sh_index = uint64_t(seq) * gridDim.y + head;
  uint64_t base = sh_index * head_dim;
  float sa = s_a[sh_index];
  float sb = s_b[sh_index];
  float smax = fmaxf(sa, sb);
  float wa = exp2f(sa - smax);
  float wb = exp2f(sb - smax);
  float denom = wa + wb + 1e-12f;
  float a_scale = wa / denom;
  float b_scale = wb / denom;
  if (lane == 0) {
    s_out[sh_index] = log2f(denom) + smax;
  }
  a_scale = __shfl_sync(mask, a_scale, 0);
  b_scale = __shfl_sync(mask, b_scale, 0);
  if (head_dim % 4 == 0) {
    uint32_t vecCount = head_dim / 4;
    const float4* va4 = reinterpret_cast<const float4*>(v_a + base);
    const float4* vb4 = reinterpret_cast<const float4*>(v_b + base);
    float4* vo4 = reinterpret_cast<float4*>(v_out + base);
    for (uint32_t i = lane; i < vecCount; i += stride) {
      float4 a = va4[i];
      float4 b = vb4[i];
      float4 o;
      o.x = fmaf(a_scale, a.x, b_scale * b.x);
      o.y = fmaf(a_scale, a.y, b_scale * b.y);
      o.z = fmaf(a_scale, a.z, b_scale * b.z);
      o.w = fmaf(a_scale, a.w, b_scale * b.w);
      vo4[i] = o;
    }
  } else {
    for (uint32_t d = lane; d < head_dim; d += stride) {
      float va_f = v_a[base + d];
      float vb_f = v_b[base + d];
      v_out[base + d] = fmaf(a_scale, va_f, b_scale * vb_f);
    }
  }
}

__global__ __launch_bounds__(MAX_THREADS) void merge_state_kernel_v3_half(
    const __half* __restrict__ v_a,
    const float* __restrict__ s_a,
    const __half* __restrict__ v_b,
    const float* __restrict__ s_b,
    __half* __restrict__ v_out,
    float* __restrict__ s_out,
    uint32_t head_dim) {
  unsigned mask = 0xffffffffu;
  uint32_t seq = blockIdx.x;
  uint32_t head = blockIdx.y;
  uint32_t lane = threadIdx.x;
  uint32_t stride = blockDim.x;
  uint64_t sh_index = uint64_t(seq) * gridDim.y + head;
  uint64_t base = sh_index * head_dim;
  float sa = s_a[sh_index];
  float sb = s_b[sh_index];
  float smax = fmaxf(sa, sb);
  float wa = exp2f(sa - smax);
  float wb = exp2f(sb - smax);
  float denom = wa + wb + 1e-12f;
  float a_scale = wa / denom;
  float b_scale = wb / denom;
  if (lane == 0) {
    s_out[sh_index] = log2f(denom) + smax;
  }
  a_scale = __shfl_sync(mask, a_scale, 0);
  b_scale = __shfl_sync(mask, b_scale, 0);
  if (head_dim % 2 == 0) {
    uint32_t vecCount = head_dim / 2;
    const __half2* va2 = reinterpret_cast<const __half2*>(v_a + base);
    const __half2* vb2 = reinterpret_cast<const __half2*>(v_b + base);
    __half2* vo2 = reinterpret_cast<__half2*>(v_out + base);
    for (uint32_t i = lane; i < vecCount; i += stride) {
      float2 fa = __half22float2(va2[i]);
      float2 fb = __half22float2(vb2[i]);
      float2 fo;
      fo.x = fmaf(a_scale, fa.x, b_scale * fb.x);
      fo.y = fmaf(a_scale, fa.y, b_scale * fb.y);
      vo2[i] = __float22half2_rn(fo);
    }
  } else {
    for (uint32_t d = lane; d < head_dim; d += stride) {
      float va_f = to_float(v_a[base + d]);
      float vb_f = to_float(v_b[base + d]);
      v_out[base + d] = cast_from_float<__half>(fmaf(a_scale, va_f, b_scale * vb_f));
    }
  }
}

__global__ __launch_bounds__(MAX_THREADS) void merge_state_kernel_v3_bf16(
    const __nv_bfloat16* __restrict__ v_a,
    const float* __restrict__ s_a,
    const __nv_bfloat16* __restrict__ v_b,
    const float* __restrict__ s_b,
    __nv_bfloat16* __restrict__ v_out,
    float* __restrict__ s_out,
    uint32_t head_dim) {
  unsigned mask = 0xffffffffu;
  uint32_t seq = blockIdx.x;
  uint32_t head = blockIdx.y;
  uint32_t lane = threadIdx.x;
  uint32_t stride = blockDim.x;
  uint64_t sh_index = uint64_t(seq) * gridDim.y + head;
  uint64_t base = sh_index * head_dim;
  float sa = s_a[sh_index];
  float sb = s_b[sh_index];
  float smax = fmaxf(sa, sb);
  float wa = exp2f(sa - smax);
  float wb = exp2f(sb - smax);
  float denom = wa + wb + 1e-12f;
  float a_scale = wa / denom;
  float b_scale = wb / denom;
  if (lane == 0) {
    s_out[sh_index] = log2f(denom) + smax;
  }
  a_scale = __shfl_sync(mask, a_scale, 0);
  b_scale = __shfl_sync(mask, b_scale, 0);
  for (uint32_t d = lane; d < head_dim; d += stride) {
    float va_f = to_float(v_a[base + d]);
    float vb_f = to_float(v_b[base + d]);
    v_out[base + d] = cast_from_float<__nv_bfloat16>(fmaf(a_scale, va_f, b_scale * vb_f));
  }
}

void merge_state(at::Tensor v_a, at::Tensor s_a,
                 at::Tensor v_b, at::Tensor s_b,
                 at::Tensor v_out, at::Tensor s_out) {
  CHECK_INPUT(v_a);
  CHECK_INPUT(s_a);
  CHECK_INPUT(v_b);
  CHECK_INPUT(s_b);
  auto device = v_a.device();
  CHECK_EQ(s_a.device(), device);
  CHECK_EQ(v_b.device(), device);
  CHECK_EQ(s_b.device(), device);
  CHECK_DIM(3, v_a);
  CHECK_DIM(2, s_a);
  CHECK_DIM(3, v_b);
  CHECK_DIM(2, s_b);
  CHECK_SHAPE(v_a, v_b);
  CHECK_SHAPE(s_a, s_b);
  CHECK_EQ(v_a.size(0), s_a.size(0));
  CHECK_EQ(v_a.size(1), s_b.size(1));
  CHECK_SHAPE(v_a, v_out);
  CHECK_SHAPE(s_a, s_out);

  const uint32_t seq_len   = (uint32_t)v_a.size(0);
  const uint32_t num_heads = (uint32_t)v_a.size(1);
  const uint32_t head_dim  = (uint32_t)v_a.size(2);

  c10::cuda::OptionalCUDAGuard guard(v_a.device());
  auto stream = at::cuda::getCurrentCUDAStream();

  dim3 blocks(seq_len, num_heads);
  uint32_t threads = std::min((uint32_t)NEXT_MULTIPLE(head_dim, 32), MAX_THREADS);

  bool ok = false;
  if (v_a.scalar_type() == at::kFloat) {
    merge_state_kernel_v3_float<<<blocks, threads, 0, stream>>>(
      v_a.data_ptr<float>(),
      s_a.data_ptr<float>(),
      v_b.data_ptr<float>(),
      s_b.data_ptr<float>(),
      v_out.data_ptr<float>(),
      s_out.data_ptr<float>(),
      head_dim);
    ok = true;
  } else if (v_a.scalar_type() == at::kHalf) {
    merge_state_kernel_v3_half<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __half*>(v_a.data_ptr()),
      s_a.data_ptr<float>(),
      reinterpret_cast<const __half*>(v_b.data_ptr()),
      s_b.data_ptr<float>(),
      reinterpret_cast<__half*>(v_out.data_ptr()),
      s_out.data_ptr<float>(),
      head_dim);
    ok = true;
  } else if (v_a.scalar_type() == at::kBFloat16) {
    merge_state_kernel_v3_bf16<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(v_a.data_ptr()),
      s_a.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(v_b.data_ptr()),
      s_b.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(v_out.data_ptr()),
      s_out.data_ptr<float>(),
      head_dim);
    ok = true;
  }

  auto err = cudaGetLastError();
  TORCH_CHECK(ok && err == cudaSuccess, "merge_state kernel launch v3 failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("merge_state", &merge_state, "merge_state (v3)");
}