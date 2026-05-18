#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

constexpr int MAX_BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int ROWS_PER_BLOCK = 4;
constexpr int MAX_WARPS_PER_BLOCK = MAX_BLOCK_SIZE / WARP_SIZE;

namespace {
// Fused Add + RMSNorm kernel with custom block-wide reduction and multi-row per block
__global__ __launch_bounds__(MAX_BLOCK_SIZE, 4)
void fused_add_rmsnorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ input_out,
    const float* __restrict__ weight,
    int32_t batch_size,
    int32_t hidden_size,
    int32_t stride_input,
    int32_t stride_residual,
    float invN,
    float eps) {
  __shared__ float warp_sums[MAX_WARPS_PER_BLOCK];
  __shared__ float inv_rms_s;

  int tid = threadIdx.x;
  int block_size = blockDim.x;
  int warps_per_block = (block_size + WARP_SIZE - 1) / WARP_SIZE;
  int row_start = blockIdx.x * ROWS_PER_BLOCK;

  for (int r = 0; r < ROWS_PER_BLOCK; ++r) {
    int b = row_start + r;
    if (b >= batch_size) break;

    const float* __restrict__ in_row = input + int64_t(b) * stride_input;
    const float* __restrict__ res_row = residual + int64_t(b) * stride_residual;
    float* __restrict__ out_row = input_out + int64_t(b) * stride_input;

    const float4* __restrict__ in4 = reinterpret_cast<const float4*>(in_row);
    const float4* __restrict__ res4 = reinterpret_cast<const float4*>(res_row);
    const float4* __restrict__ w4 = reinterpret_cast<const float4*>(weight);
    float4* __restrict__ out4 = reinterpret_cast<float4*>(out_row);

    int vec_size = hidden_size >> 2;
    float sum = 0.0f;

    #pragma unroll 8
    for (int idx = tid; idx < vec_size; idx += block_size) {
      float4 iv = in4[idx];
      float4 rv = res4[idx];
      float4 s = {iv.x + rv.x, iv.y + rv.y, iv.z + rv.z, iv.w + rv.w};
      sum = fmaf(s.x, s.x, sum);
      sum = fmaf(s.y, s.y, sum);
      sum = fmaf(s.z, s.z, sum);
      sum = fmaf(s.w, s.w, sum);
    }

    #pragma unroll 8
    for (int i = (vec_size << 2) + tid; i < hidden_size; i += block_size) {
      float v = __ldg(in_row + i) + __ldg(res_row + i);
      sum = fmaf(v, v, sum);
    }

    // warp-level reduction
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
      float tmp = __shfl_xor_sync(0xffffffff, sum, offset);
      sum += tmp;
    }
    if ((tid & (WARP_SIZE - 1)) == 0) {
      warp_sums[tid / WARP_SIZE] = sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    if (tid == 0) {
      for (int i = 0; i < warps_per_block; ++i) {
        block_sum += warp_sums[i];
      }
      inv_rms_s = rsqrtf(block_sum * invN + eps);
    }
    __syncthreads();
    float inv_rms = inv_rms_s;

    // normalize + weight
    #pragma unroll 8
    for (int idx = tid; idx < vec_size; idx += block_size) {
      float4 iv = in4[idx];
      float4 rv = res4[idx];
      float4 s = {iv.x + rv.x, iv.y + rv.y, iv.z + rv.z, iv.w + rv.w};
      float4 wf = w4[idx];
      float4 outv;
      outv.x = fmaf(s.x * inv_rms, wf.x, 0.0f);
      outv.y = fmaf(s.y * inv_rms, wf.y, 0.0f);
      outv.z = fmaf(s.z * inv_rms, wf.z, 0.0f);
      outv.w = fmaf(s.w * inv_rms, wf.w, 0.0f);
      out4[idx] = outv;
    }

    #pragma unroll 8
    for (int i = (vec_size << 2) + tid; i < hidden_size; i += block_size) {
      float v = __ldg(in_row + i) + __ldg(res_row + i);
      out_row[i] = fmaf(v * inv_rms, __ldg(weight + i), 0.0f);
    }
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

  int32_t B = input.size(0);
  int32_t D = input.size(1);

  auto input_f32    = input.contiguous().to(torch::kFloat);
  auto residual_f32 = residual.contiguous().to(torch::kFloat);
  auto weight_f32   = weight.contiguous().to(torch::kFloat);

  int32_t stride_input    = input_f32.stride(0);
  int32_t stride_residual = residual_f32.stride(0);

  float invN = 1.0f / static_cast<float>(D);

  const int rows_per_block = ROWS_PER_BLOCK;
  int grid_x = (B + rows_per_block - 1) / rows_per_block;

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(grid_x);
  dim3 block(MAX_BLOCK_SIZE);

  fused_add_rmsnorm_kernel<<<grid, block, 0, stream>>>(
      input_f32.data_ptr<float>(),
      residual_f32.data_ptr<float>(),
      input_f32.data_ptr<float>(),
      weight_f32.data_ptr<float>(),
      B, D,
      stride_input,
      stride_residual,
      invN,
      static_cast<float>(eps));

  input.copy_(input_f32.to(input.scalar_type()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sgl_fused_add_rmsnorm", &sgl_fused_add_rmsnorm,
        "Simplified fused Add + RMSNorm (in-place on input)");
}