#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int BLOCK_SIZE = 256;

// Kernel 1: compute per-row inverse RMS with vectorized loads and shared-memory reduction
__global__ __launch_bounds__(BLOCK_SIZE)
void compute_inv_rms_kernel(const float* __restrict__ input,
                            const float* __restrict__ residual,
                            int32_t batch_size,
                            int32_t hidden_size,
                            int32_t stride_input,
                            int32_t stride_residual,
                            float eps,
                            float* __restrict__ inv_rms_out) {
  int b = blockIdx.x;
  if (b >= batch_size) return;
  const float* in_row = input + static_cast<int64_t>(b) * stride_input;
  const float* res_row = residual + static_cast<int64_t>(b) * stride_residual;

  int vec_size = hidden_size >> 2;
  int tail = hidden_size & 3;
  const float4* in4 = reinterpret_cast<const float4*>(in_row);
  const float4* res4 = reinterpret_cast<const float4*>(res_row);

  float sum = 0.0f;
  // Vectorized accumulation
  for (int idx = threadIdx.x; idx < vec_size; idx += BLOCK_SIZE) {
    float4 iv = in4[idx];
    float4 rv = res4[idx];
    sum = fmaf(iv.x + rv.x, iv.x + rv.x, sum);
    sum = fmaf(iv.y + rv.y, iv.y + rv.y, sum);
    sum = fmaf(iv.z + rv.z, iv.z + rv.z, sum);
    sum = fmaf(iv.w + rv.w, iv.w + rv.w, sum);
  }
  // Remainder elements
  int offset = vec_size << 2;
  for (int i = offset + threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
    float x = in_row[i] + res_row[i];
    sum = fmaf(x, x, sum);
  }

  // Shared-memory reduction
  __shared__ float sdata[BLOCK_SIZE];
  sdata[threadIdx.x] = sum;
  __syncthreads();
  for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      sdata[threadIdx.x] += sdata[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float mean = sdata[0] / float(hidden_size);
    inv_rms_out[b] = rsqrtf(mean + eps);
  }
}

// Kernel 2: apply normalization and weight in-place with vectorized stores
__global__ __launch_bounds__(BLOCK_SIZE)
void apply_rms_weight_kernel(float* __restrict__ input_out,
                             const float* __restrict__ residual,
                             const float* __restrict__ weight,
                             const float* __restrict__ inv_rms,
                             int32_t batch_size,
                             int32_t hidden_size,
                             int32_t stride_input,
                             int32_t stride_residual) {
  int b = blockIdx.x;
  if (b >= batch_size) return;
  float scale = inv_rms[b];
  float* in_row = input_out + static_cast<int64_t>(b) * stride_input;
  const float* res_row = residual + static_cast<int64_t>(b) * stride_residual;

  int vec_size = hidden_size >> 2;
  int tail = hidden_size & 3;
  float4* in4 = reinterpret_cast<float4*>(in_row);
  const float4* res4 = reinterpret_cast<const float4*>(res_row);
  const float4* w4 = reinterpret_cast<const float4*>(weight);

  for (int idx = threadIdx.x; idx < vec_size; idx += BLOCK_SIZE) {
    float4 iv = in4[idx];
    float4 rv = res4[idx];
    float4 sum4{iv.x + rv.x, iv.y + rv.y, iv.z + rv.z, iv.w + rv.w};
    float4 wf = w4[idx];
    float4 y4;
    y4.x = fmaf(sum4.x * scale, wf.x, 0.0f);
    y4.y = fmaf(sum4.y * scale, wf.y, 0.0f);
    y4.z = fmaf(sum4.z * scale, wf.z, 0.0f);
    y4.w = fmaf(sum4.w * scale, wf.w, 0.0f);
    in4[idx] = y4;
  }
  int offset = vec_size << 2;
  for (int i = offset + threadIdx.x; i < hidden_size; i += BLOCK_SIZE) {
    float x = in_row[i] + res_row[i];
    float w = weight[i];
    in_row[i] = fmaf(x * scale, w, 0.0f);
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

  auto inv_rms = torch::empty({B}, input_f32.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(B);
  dim3 block(BLOCK_SIZE);

  compute_inv_rms_kernel<<<grid, block, 0, stream>>>(
      input_f32.data_ptr<float>(),
      residual_f32.data_ptr<float>(),
      B, D,
      stride_input,
      stride_residual,
      static_cast<float>(eps),
      inv_rms.data_ptr<float>());

  apply_rms_weight_kernel<<<grid, block, 0, stream>>>(
      input_f32.data_ptr<float>(),
      residual_f32.data_ptr<float>(),
      weight_f32.data_ptr<float>(),
      inv_rms.data_ptr<float>(),
      B, D,
      stride_input,
      stride_residual);

  input.copy_(input_f32.to(input.scalar_type()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sgl_fused_add_rmsnorm", &sgl_fused_add_rmsnorm,
        "Simplified fused Add + RMSNorm (in-place on input)");
}