#include <vector>
#include <cfloat>
#include <cuda_fp16.h>

#include "../tester/utils.h"

namespace {

constexpr int kThreadsPerBlock = 128;

__device__ inline float toFloat(float value) { return value; }
__device__ inline float toFloat(half value) { return __half2float(value); }

template <typename T>
__device__ T fromFloat(float value);

template <>
__device__ float fromFloat<float>(float value) {
  return value;
}

template <>
__device__ half fromFloat<half>(float value) {
  return __float2half_rn(value);
}

template <typename T>
__global__ void rmsNormKernel(const T* input, const T* weight, T* output,
                              size_t rows, size_t hidden_dim, float eps) {
  const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  const size_t row_offset = row * hidden_dim;
  float sum_square = 0.0f;
  for (size_t col = 0; col < hidden_dim; ++col) {
    const float value = toFloat(input[row_offset + col]);
    sum_square += value * value;
  }

  const float scale = rsqrtf(sum_square / static_cast<float>(hidden_dim) + eps);
  for (size_t col = 0; col < hidden_dim; ++col) {
    const float value = toFloat(input[row_offset + col]);
    const float gamma = toFloat(weight[col]);
    output[row_offset + col] = fromFloat<T>(value * scale * gamma);
  }
}

template <typename T>
__global__ void attentionKernel(const T* q, const T* k, const T* v, T* output,
                                float* accumulator, int batch_size,
                                int target_seq_len, int src_seq_len,
                                int query_heads, int kv_heads, int head_dim,
                                bool is_causal) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_rows = batch_size * target_seq_len * query_heads;
  if (row >= total_rows) {
    return;
  }

  const int query_head = row % query_heads;
  const int target_pos = (row / query_heads) % target_seq_len;
  const int batch = row / (target_seq_len * query_heads);

  // Consecutive groups of query heads share one key/value head in GQA.
  const int heads_per_group = query_heads / kv_heads;
  const int kv_head = query_head / heads_per_group;
  const int valid_src_len =
      is_causal ? min(src_seq_len, target_pos + 1) : src_seq_len;
  const float softmax_scale = rsqrtf(static_cast<float>(head_dim));

  const size_t q_offset =
      ((static_cast<size_t>(batch) * target_seq_len + target_pos) *
           query_heads +
       query_head) *
      head_dim;
  const size_t output_offset = static_cast<size_t>(row) * head_dim;

  // Pass 1: find the largest scaled QK score for stable softmax.
  float max_score = -FLT_MAX;
  for (int src_pos = 0; src_pos < valid_src_len; ++src_pos) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + src_pos) * kv_heads +
         kv_head) *
        head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot += toFloat(q[q_offset + dim]) * toFloat(k[kv_offset + dim]);
    }
    max_score = fmaxf(max_score, dot * softmax_scale);
  }

  // Pass 2: calculate the softmax denominator.
  float denominator = 0.0f;
  for (int src_pos = 0; src_pos < valid_src_len; ++src_pos) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + src_pos) * kv_heads +
         kv_head) *
        head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot += toFloat(q[q_offset + dim]) * toFloat(k[kv_offset + dim]);
    }
    denominator += expf(dot * softmax_scale - max_score);
  }

  for (int dim = 0; dim < head_dim; ++dim) {
    accumulator[output_offset + dim] = 0.0f;
  }

  // Pass 3: multiply each value vector by its softmax probability.
  for (int src_pos = 0; src_pos < valid_src_len; ++src_pos) {
    const size_t kv_offset =
        ((static_cast<size_t>(batch) * src_seq_len + src_pos) * kv_heads +
         kv_head) *
        head_dim;
    float dot = 0.0f;
    for (int dim = 0; dim < head_dim; ++dim) {
      dot += toFloat(q[q_offset + dim]) * toFloat(k[kv_offset + dim]);
    }
    const float probability = expf(dot * softmax_scale - max_score) /
                              denominator;
    for (int dim = 0; dim < head_dim; ++dim) {
      accumulator[output_offset + dim] +=
          probability * toFloat(v[kv_offset + dim]);
    }
  }

  for (int dim = 0; dim < head_dim; ++dim) {
    output[output_offset + dim] =
        fromFloat<T>(accumulator[output_offset + dim]);
  }
}

template <typename T>
void allocateAndCopy(T** device, const std::vector<T>& host) {
  const size_t bytes = host.size() * sizeof(T);
  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(device), bytes));
  RUNTIME_CHECK(cudaMemcpy(*device, host.data(), bytes,
                           cudaMemcpyHostToDevice));
}

}  // namespace

/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  h_output.resize(rows * hidden_dim);
  if (rows == 0 || hidden_dim == 0) {
    return;
  }

  T* d_input = nullptr;
  T* d_weight = nullptr;
  T* d_output = nullptr;
  allocateAndCopy(&d_input, h_input);
  allocateAndCopy(&d_weight, h_weight);
  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output),
                           h_output.size() * sizeof(T)));

  const int blocks =
      static_cast<int>((rows + kThreadsPerBlock - 1) / kThreadsPerBlock);
  rmsNormKernel<T><<<blocks, kThreadsPerBlock>>>(
      d_input, d_weight, d_output, rows, hidden_dim, eps);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());
  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output,
                           h_output.size() * sizeof(T),
                           cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_weight));
  RUNTIME_CHECK(cudaFree(d_output));
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  const size_t output_size = static_cast<size_t>(batch_size) * target_seq_len *
                             query_heads * head_dim;
  h_o.resize(output_size);
  if (output_size == 0 || src_seq_len == 0) {
    return;
  }

  T* d_q = nullptr;
  T* d_k = nullptr;
  T* d_v = nullptr;
  T* d_o = nullptr;
  float* d_accumulator = nullptr;
  allocateAndCopy(&d_q, h_q);
  allocateAndCopy(&d_k, h_k);
  allocateAndCopy(&d_v, h_v);
  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_o),
                           output_size * sizeof(T)));
  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_accumulator),
                           output_size * sizeof(float)));

  const int total_rows = batch_size * target_seq_len * query_heads;
  const int blocks =
      (total_rows + kThreadsPerBlock - 1) / kThreadsPerBlock;
  attentionKernel<T><<<blocks, kThreadsPerBlock>>>(
      d_q, d_k, d_v, d_o, d_accumulator, batch_size, target_seq_len,
      src_seq_len, query_heads, kv_heads, head_dim, is_causal);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, output_size * sizeof(T),
                           cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_q));
  RUNTIME_CHECK(cudaFree(d_k));
  RUNTIME_CHECK(cudaFree(d_v));
  RUNTIME_CHECK(cudaFree(d_o));
  RUNTIME_CHECK(cudaFree(d_accumulator));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
