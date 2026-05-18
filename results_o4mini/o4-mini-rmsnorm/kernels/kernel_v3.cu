#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

// Fused Add + RMSNorm kernel
__global__ __launch_bounds__(BLOCK_SIZE)
void fused_add_rmsnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ input_out,
    const float* __restrict__ weight,
    int32_t batch_size,
    int32_t hidden_size,
    int32_t stride_input,
    int32_t stride_residual,
    float eps) {
  int b = blockIdx.x;
  if (b >= batch_size) return;
  const float* in_row = input + int64_t(b) * stride_input;
  const float* res_row = residual + int64_t(b) * stride_residual;
  float* out_row = input_out + int64_t(b) * stride_input;

  int tid = threadIdx.x;
  int vec_size = hidden_size >> 2;
  int tail = hidden_size & 3;
  const float4* in4  = reinterpret_cast<const float4*>(in_row);
  const float4* res4 = reinterpret_cast<const float4*>(res_row);
  const float4* w4   = reinterpret_cast<const float4*>(weight);
  float4* out4       = reinterpret_cast<float4*>(out_row);

  // 1) compute sum of squares
  float sum = 0.0f;
#pragma unroll 2
  for (int idx = tid; idx < vec_size; idx += BLOCK_SIZE) {
    float4 iv = in4[idx];
    float4 rv = res4[idx];
    float4 s = {iv.x + rv.x, iv.y + rv.y, iv.z + rv.z, iv.w + rv.w};
    sum = fmaf(s.x, s.x, sum);
    sum = fmaf(s.y, s.y, sum);
    sum = fmaf(s.z, s.z, sum);
    sum = fmaf(s.w, s.w, sum);
  }
#pragma unroll 2
  for (int i = (vec_size << 2) + tid; i < hidden_size; i += BLOCK_SIZE) {
    float v = __ldg(in_row + i) + __ldg(res_row + i);
    sum = fmaf(v, v, sum);
  }

  // 2) warp-level reduction
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  }
  __shared__ float warp_sums[WARPS_PER_BLOCK];
  int warp_id = tid / WARP_SIZE;
  if ((tid & (WARP_SIZE - 1)) == 0) {
    warp_sums[warp_id] = sum;
  }
  __syncthreads();
  float block_sum = 0.0f;
  if (warp_id == 0) {
    block_sum = (tid < WARPS_PER_BLOCK) ? warp_sums[tid] : 0.0f;
#pragma unroll 2
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
    }
  }
  __syncthreads();
  float inv_rms = rsqrtf(block_sum / hidden_size + eps);

  // 3) apply normalization and weight
#pragma unroll 2
  for (int idx = tid; idx < vec_size; idx += BLOCK_SIZE) {
    float4 iv = in4[idx];
    float4 rv = res4[idx];
    float4 sum4 = {iv.x + rv.x, iv.y + rv.y, iv.z + rv.z, iv.w + rv.w};
    float4 wf = w4[idx];
    float4 outv;
    outv.x = fmaf(sum4.x * inv_rms, wf.x, 0.0f);
    outv.y = fmaf(sum4.y * inv_rms, wf.y, 0.0f);
    outv.z = fmaf(sum4.z * inv_rms, wf.z, 0.0f);
    outv.w = fmaf(sum4.w * inv_rms, wf.w, 0.0f);
    out4[idx] = outv;
  }
#pragma unroll 2
  for (int i = (vec_size << 2) + tid; i < hidden_size; i += BLOCK_SIZE) {
    float v = in_row[i] + res_row[i];
    out_row[i] = fmaf(v * inv_rms, weight[i], 0.0f);
  }
}

} // namespace

void sgl_fused_add_rmsnorm(torch::Tensor input,
                           torch::Tensor residual,
                           torch::Tensor weight,
                           double eps,
                           bool /*enable_pdl*/) {
  TORCH_CHECK(input.is_cuda(),    "input must be CUDA");
  TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
  TORCH_CHECK(weight.is_cuda(),   "weight must be CUDA");

  TORCH_CHECK(input.dim() == 2 && residual.dim() == 2, "input and residual must be (B, D)");
  TORCH_CHECK(weight.dim() == 1,                        "weight must be (D)");
  TORCH_CHECK(input.size(0) == residual.size(0),       "batch size mismatch");
  TORCH_CHECK(input.size(1) == residual.size(1),       "hidden size mismatch");
  TORCH_CHECK(input.size(1) == weight.size(0),         "weight size mismatch");

  const int32_t B = static_cast<int32_t>(input.size(0));
  const int32_t D = static_cast<int32_t>(input.size(1));

  auto input_f32    = input.contiguous().to(torch::kFloat);
  auto residual_f32 = residual.contiguous().to(torch::kFloat);
  auto weight_f32   = weight.contiguous().to(torch::kFloat);

  const int32_t stride_input    = static_cast<int32_t>(input_f32.stride(0));
  const int32_t stride_residual = static_cast<int32_t>(residual_f32.stride(0));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(B);
  dim3 block(BLOCK_SIZE);

  fused_add_rmsnorm_kernel<<<grid, block, 0, stream>>>(
      input_f32.data_ptr<float>(),
      residual_f32.data_ptr<float>(),
      input_f32.data_ptr<float>(),
      weight_f32.data_ptr<float>(),
      B, D,
      stride_input,
      stride_residual,
      static_cast<float>(eps));

  input.copy_(input_f32.to(input.scalar_type()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sgl_fused_add_rmsnorm", &sgl_fused_add_rmsnorm,
        "Simplified fused Add + RMSNorm (in-place on input)");
}