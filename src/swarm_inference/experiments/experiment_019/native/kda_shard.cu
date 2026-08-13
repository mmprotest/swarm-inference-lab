#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

#if defined(_WIN32)
#define EXP019_EXPORT extern "C" __declspec(dllexport)
#else
#define EXP019_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

__device__ __forceinline__ float bf16_to_float(unsigned short bits) {
    return __uint_as_float(static_cast<unsigned int>(bits) << 16U);
}

__global__ void grouped_int4_quantize(
    const unsigned short *source,
    unsigned char *packed,
    float *scales,
    int rows,
    int columns) {
    const int groups_per_row = columns / 64;
    const int linear_group = blockIdx.x;
    const int row = linear_group / groups_per_row;
    const int group = linear_group - row * groups_per_row;
    const int lane = threadIdx.x;
    if (row >= rows || lane >= 64) {
        return;
    }
    __shared__ float values[64];
    __shared__ float maxima[64];
    __shared__ float scale;
    __shared__ float inverse;
    const int source_index = row * columns + group * 64 + lane;
    const float value = bf16_to_float(source[source_index]);
    values[lane] = value;
    maxima[lane] = fabsf(value);
    __syncthreads();
    for (int stride = 32; stride > 0; stride >>= 1) {
        if (lane < stride) {
            maxima[lane] = fmaxf(maxima[lane], maxima[lane + stride]);
        }
        __syncthreads();
    }
    if (lane == 0) {
        scale = fmaxf(maxima[0] / 7.0F, 1.0e-20F);
        inverse = 1.0F / scale;
        scales[linear_group] = scale;
    }
    __syncthreads();
    if (lane < 32) {
        int low = static_cast<int>(rintf(values[lane * 2] * inverse));
        int high = static_cast<int>(rintf(values[lane * 2 + 1] * inverse));
        low = max(-8, min(7, low));
        high = max(-8, min(7, high));
        const unsigned char encoded_low = static_cast<unsigned char>(low + 8);
        const unsigned char encoded_high = static_cast<unsigned char>(high + 8);
        packed[row * (columns / 2) + group * 32 + lane] =
            encoded_low | static_cast<unsigned char>(encoded_high << 4U);
    }
}

__global__ void row_int8_quantize(
    const unsigned short *source,
    signed char *quantized,
    float *scales,
    int rows,
    int columns) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= rows) {
        return;
    }
    __shared__ float maxima[256];
    __shared__ float scale;
    __shared__ float inverse;
    float local_maximum = 0.0F;
    for (int column = lane; column < columns; column += blockDim.x) {
        local_maximum = fmaxf(
            local_maximum,
            fabsf(bf16_to_float(source[row * columns + column])));
    }
    maxima[lane] = local_maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            maxima[lane] = fmaxf(maxima[lane], maxima[lane + stride]);
        }
        __syncthreads();
    }
    if (lane == 0) {
        scale = fmaxf(maxima[0] / 127.0F, 1.0e-20F);
        inverse = 1.0F / scale;
        scales[row] = scale;
    }
    __syncthreads();
    for (int column = lane; column < columns; column += blockDim.x) {
        int value = static_cast<int>(rintf(
            bf16_to_float(source[row * columns + column]) * inverse));
        value = max(-127, min(127, value));
        quantized[row * columns + column] = static_cast<signed char>(value);
    }
}

__global__ void row_absolute_maximum(
    const unsigned short *source,
    float *maxima_output,
    int rows,
    int columns) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= rows) {
        return;
    }
    __shared__ float maxima[256];
    float local_maximum = 0.0F;
    for (int column = lane; column < columns; column += blockDim.x) {
        local_maximum = fmaxf(
            local_maximum,
            fabsf(bf16_to_float(source[row * columns + column])));
    }
    maxima[lane] = local_maximum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            maxima[lane] = fmaxf(maxima[lane], maxima[lane + stride]);
        }
        __syncthreads();
    }
    if (lane == 0) {
        maxima_output[row] = maxima[0];
    }
}

__global__ void row_int8_quantize_with_scales(
    const unsigned short *source,
    const float *scales,
    signed char *quantized,
    int rows,
    int columns) {
    const size_t total = static_cast<size_t>(rows) * columns;
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int row = static_cast<int>(index / columns);
        const float inverse = 1.0F / scales[row];
        int value = static_cast<int>(rintf(bf16_to_float(source[index]) * inverse));
        value = max(-127, min(127, value));
        quantized[index] = static_cast<signed char>(value);
    }
}

__global__ void f32_rmsnorm(
    float *output,
    const float *input,
    const float *weight,
    int rows,
    int columns,
    float eps) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= rows) {
        return;
    }
    __shared__ double sums[256];
    __shared__ float inverse;
    double local = 0.0;
    for (int column = lane; column < columns; column += blockDim.x) {
        const double value = static_cast<double>(input[row * columns + column]);
        local += value * value;
    }
    sums[lane] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            sums[lane] += sums[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        inverse = rsqrtf(
            static_cast<float>(sums[0] / static_cast<double>(columns)) + eps);
    }
    __syncthreads();
    for (int column = lane; column < columns; column += blockDim.x) {
        output[row * columns + column] =
            input[row * columns + column] * inverse * weight[column];
    }
}

__global__ void kda_shard_short_window(
    float *output,
    const float *q,
    const float *k,
    const float *v,
    const float *gate,
    const float *decay,
    const float *beta_raw,
    const float *conv_q,
    const float *conv_k,
    const float *conv_v,
    float *window_q,
    float *window_k,
    float *window_v,
    float *state,
    const float *dt,
    const float *a,
    const float *output_norm,
    int rows,
    int heads,
    int head_dim,
    int conv_width,
    float gate_lower_bound,
    float eps) {
    const int head = blockIdx.x;
    const int lane = threadIdx.x;
    const int state_index = head * head_dim + lane;
    if (head >= heads || lane >= head_dim) {
        return;
    }

    __shared__ float qconv[128];
    __shared__ float kconv[128];
    __shared__ float vconv[128];
    __shared__ float qn[128];
    __shared__ float kn[128];
    __shared__ float alpha[128];
    __shared__ float vt[128];
    __shared__ float qsum[128];
    __shared__ float ksum[128];
    __shared__ double norm_sum[128];

    for (int row_index = 0; row_index < rows; ++row_index) {
        const int d = row_index * heads * head_dim + state_index;
        float *windows[3] = {window_q, window_k, window_v};
        const float *taps[3] = {conv_q, conv_k, conv_v};
        const float *values[3] = {q, k, v};
        for (int which = 0; which < 3; ++which) {
            float *window = windows[which] +
                            static_cast<size_t>(state_index) * conv_width;
            for (int tap = 0; tap < conv_width - 1; ++tap) {
                window[tap] = window[tap + 1];
            }
            window[conv_width - 1] = values[which][d];
            float accumulator = 0.0F;
            const float *filter = taps[which] +
                                  static_cast<size_t>(state_index) * conv_width;
            for (int tap = 0; tap < conv_width; ++tap) {
                accumulator += filter[tap] * window[tap];
            }
            const float convolved = accumulator / (1.0F + expf(-accumulator));
            if (which == 0) {
                qconv[lane] = convolved;
            } else if (which == 1) {
                kconv[lane] = convolved;
            } else {
                vconv[lane] = convolved;
            }
        }

        qsum[lane] = qconv[lane] * qconv[lane];
        ksum[lane] = kconv[lane] * kconv[lane];
        __syncthreads();
        for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                qsum[lane] += qsum[lane + stride];
                ksum[lane] += ksum[lane + stride];
            }
            __syncthreads();
        }

        const float qscale = rsqrtf(static_cast<float>(head_dim));
        qn[lane] = qconv[lane] * rsqrtf(qsum[0] + 1.0e-6F) * qscale;
        kn[lane] = kconv[lane] * rsqrtf(ksum[0] + 1.0e-6F);
        const float z = decay[d] + dt[state_index];
        alpha[lane] = expf(
            gate_lower_bound / (1.0F + expf(-a[head] * z)));
        __syncthreads();

        float *head_state = state +
                            static_cast<size_t>(head) * head_dim * head_dim;
        float k_state = 0.0F;
        for (int kk = 0; kk < head_dim; ++kk) {
            float *cell = head_state + static_cast<size_t>(kk) * head_dim + lane;
            const float state_value = *cell * alpha[kk];
            *cell = state_value;
            k_state += kn[kk] * state_value;
        }
        const float beta =
            1.0F / (1.0F + expf(-beta_raw[row_index * heads + head]));
        vt[lane] = (vconv[lane] - k_state) * beta;
        __syncthreads();

        float head_output = 0.0F;
        for (int kk = 0; kk < head_dim; ++kk) {
            float *cell = head_state + static_cast<size_t>(kk) * head_dim + lane;
            const float state_value = *cell + kn[kk] * vt[lane];
            *cell = state_value;
            head_output += qn[kk] * state_value;
        }
        norm_sum[lane] = static_cast<double>(head_output) *
                         static_cast<double>(head_output);
        __syncthreads();
        for (int stride = head_dim / 2; stride > 0; stride >>= 1) {
            if (lane < stride) {
                norm_sum[lane] += norm_sum[lane + stride];
            }
            __syncthreads();
        }

        const float inverse = rsqrtf(
            static_cast<float>(norm_sum[0] / static_cast<double>(head_dim)) +
            eps);
        const float gate_value = 1.0F / (1.0F + expf(-gate[d]));
        output[d] =
            head_output * inverse * output_norm[lane] * gate_value;
        __syncthreads();
    }
}

}  // namespace

EXP019_EXPORT int exp019_quantize_grouped_int4_host(
    int device,
    const unsigned short *source_host,
    int rows,
    int columns,
    unsigned char *packed_host,
    float *scales_host) {
    if (!source_host || !packed_host || !scales_host || rows < 1 ||
        columns < 64 || columns % 64 != 0 || cudaSetDevice(device) != cudaSuccess) {
        return 0;
    }
    unsigned short *source_device = nullptr;
    unsigned char *packed_device = nullptr;
    float *scales_device = nullptr;
    const size_t source_bytes =
        static_cast<size_t>(rows) * columns * sizeof(unsigned short);
    const size_t packed_bytes = static_cast<size_t>(rows) * columns / 2;
    const size_t scale_count = static_cast<size_t>(rows) * columns / 64;
    if (cudaMalloc(&source_device, source_bytes) != cudaSuccess ||
        cudaMalloc(&packed_device, packed_bytes) != cudaSuccess ||
        cudaMalloc(&scales_device, scale_count * sizeof(float)) != cudaSuccess) {
        cudaFree(scales_device);
        cudaFree(packed_device);
        cudaFree(source_device);
        return 0;
    }
    bool ok = cudaMemcpy(
                  source_device,
                  source_host,
                  source_bytes,
                  cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        grouped_int4_quantize<<<static_cast<unsigned int>(scale_count), 64>>>(
            source_device, packed_device, scales_device, rows, columns);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        ok = cudaMemcpy(
                 packed_host,
                 packed_device,
                 packed_bytes,
                 cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    if (ok) {
        ok = cudaMemcpy(
                 scales_host,
                 scales_device,
                 scale_count * sizeof(float),
                 cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(scales_device);
    cudaFree(packed_device);
    cudaFree(source_device);
    return ok ? 1 : 0;
}

EXP019_EXPORT int exp019_quantize_row_int8_host(
    int device,
    const unsigned short *source_host,
    int rows,
    int columns,
    signed char *quantized_host,
    float *scales_host) {
    if (!source_host || !quantized_host || !scales_host || rows < 1 ||
        columns < 1 || cudaSetDevice(device) != cudaSuccess) {
        return 0;
    }
    unsigned short *source_device = nullptr;
    signed char *quantized_device = nullptr;
    float *scales_device = nullptr;
    const size_t source_bytes =
        static_cast<size_t>(rows) * columns * sizeof(unsigned short);
    const size_t quantized_bytes = static_cast<size_t>(rows) * columns;
    if (cudaMalloc(&source_device, source_bytes) != cudaSuccess ||
        cudaMalloc(&quantized_device, quantized_bytes) != cudaSuccess ||
        cudaMalloc(&scales_device, static_cast<size_t>(rows) * sizeof(float)) !=
            cudaSuccess) {
        cudaFree(scales_device);
        cudaFree(quantized_device);
        cudaFree(source_device);
        return 0;
    }
    bool ok = cudaMemcpy(
                  source_device,
                  source_host,
                  source_bytes,
                  cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        row_int8_quantize<<<rows, 256>>>(
            source_device, quantized_device, scales_device, rows, columns);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        ok = cudaMemcpy(
                 quantized_host,
                 quantized_device,
                 quantized_bytes,
                 cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    if (ok) {
        ok = cudaMemcpy(
                 scales_host,
                 scales_device,
                 static_cast<size_t>(rows) * sizeof(float),
                 cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(scales_device);
    cudaFree(quantized_device);
    cudaFree(source_device);
    return ok ? 1 : 0;
}

EXP019_EXPORT int exp019_bf16_row_max_host(
    int device,
    const unsigned short *source_host,
    int rows,
    int columns,
    float *maxima_host) {
    if (!source_host || !maxima_host || rows < 1 || columns < 1 ||
        cudaSetDevice(device) != cudaSuccess) {
        return 0;
    }
    unsigned short *source_device = nullptr;
    float *maxima_device = nullptr;
    const size_t source_bytes =
        static_cast<size_t>(rows) * columns * sizeof(unsigned short);
    if (cudaMalloc(&source_device, source_bytes) != cudaSuccess ||
        cudaMalloc(&maxima_device, static_cast<size_t>(rows) * sizeof(float)) !=
            cudaSuccess) {
        cudaFree(maxima_device);
        cudaFree(source_device);
        return 0;
    }
    bool ok = cudaMemcpy(
                  source_device,
                  source_host,
                  source_bytes,
                  cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        row_absolute_maximum<<<rows, 256>>>(
            source_device, maxima_device, rows, columns);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        ok = cudaMemcpy(
                 maxima_host,
                 maxima_device,
                 static_cast<size_t>(rows) * sizeof(float),
                 cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(maxima_device);
    cudaFree(source_device);
    return ok ? 1 : 0;
}

EXP019_EXPORT int exp019_quantize_row_int8_with_scales_host(
    int device,
    const unsigned short *source_host,
    const float *scales_host,
    int rows,
    int columns,
    signed char *quantized_host) {
    if (!source_host || !scales_host || !quantized_host || rows < 1 ||
        columns < 1 || cudaSetDevice(device) != cudaSuccess) {
        return 0;
    }
    unsigned short *source_device = nullptr;
    float *scales_device = nullptr;
    signed char *quantized_device = nullptr;
    const size_t source_bytes =
        static_cast<size_t>(rows) * columns * sizeof(unsigned short);
    const size_t quantized_bytes = static_cast<size_t>(rows) * columns;
    if (cudaMalloc(&source_device, source_bytes) != cudaSuccess ||
        cudaMalloc(&scales_device, static_cast<size_t>(rows) * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc(&quantized_device, quantized_bytes) != cudaSuccess) {
        cudaFree(quantized_device);
        cudaFree(scales_device);
        cudaFree(source_device);
        return 0;
    }
    bool ok = cudaMemcpy(
                  source_device,
                  source_host,
                  source_bytes,
                  cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(
                  scales_device,
                  scales_host,
                  static_cast<size_t>(rows) * sizeof(float),
                  cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        const size_t total = static_cast<size_t>(rows) * columns;
        unsigned int blocks = static_cast<unsigned int>((total + 255) / 256);
        if (blocks > 4096) {
            blocks = 4096;
        }
        row_int8_quantize_with_scales<<<blocks, 256>>>(
            source_device, scales_device, quantized_device, rows, columns);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        ok = cudaMemcpy(
                 quantized_host,
                 quantized_device,
                 quantized_bytes,
                 cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(quantized_device);
    cudaFree(scales_device);
    cudaFree(source_device);
    return ok ? 1 : 0;
}

EXP019_EXPORT int exp019_rmsnorm_host(
    int device,
    const float *input_host,
    const float *weight_host,
    float *output_host,
    int rows,
    int columns,
    float eps) {
    if (!input_host || !weight_host || !output_host || rows < 1 || columns < 1 ||
        eps <= 0.0F || cudaSetDevice(device) != cudaSuccess) {
        return 0;
    }
    float *input_device = nullptr;
    float *weight_device = nullptr;
    float *output_device = nullptr;
    const size_t values_bytes =
        static_cast<size_t>(rows) * columns * sizeof(float);
    const size_t weight_bytes = static_cast<size_t>(columns) * sizeof(float);
    if (cudaMalloc(&input_device, values_bytes) != cudaSuccess ||
        cudaMalloc(&weight_device, weight_bytes) != cudaSuccess ||
        cudaMalloc(&output_device, values_bytes) != cudaSuccess) {
        cudaFree(output_device);
        cudaFree(weight_device);
        cudaFree(input_device);
        return 0;
    }
    bool ok = cudaMemcpy(
                  input_device,
                  input_host,
                  values_bytes,
                  cudaMemcpyHostToDevice) == cudaSuccess &&
              cudaMemcpy(
                  weight_device,
                  weight_host,
                  weight_bytes,
                  cudaMemcpyHostToDevice) == cudaSuccess;
    if (ok) {
        f32_rmsnorm<<<rows, 256>>>(
            output_device, input_device, weight_device, rows, columns, eps);
        ok = cudaGetLastError() == cudaSuccess;
    }
    if (ok) {
        ok = cudaMemcpy(
                 output_host,
                 output_device,
                 values_bytes,
                 cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(output_device);
    cudaFree(weight_device);
    cudaFree(input_device);
    return ok ? 1 : 0;
}

EXP019_EXPORT int exp019_kda_shard_short_window_dev(
    int device,
    float *output_dev,
    const float *q_dev,
    const float *k_dev,
    const float *v_dev,
    const float *gate_dev,
    const float *decay_dev,
    const float *beta_dev,
    const float *conv_q_dev,
    const float *conv_k_dev,
    const float *conv_v_dev,
    float *window_q_dev,
    float *window_k_dev,
    float *window_v_dev,
    float *state_dev,
    const float *dt_dev,
    const float *a_dev,
    const float *output_norm_dev,
    int rows,
    int heads,
    int head_dim,
    int conv_width,
    float gate_lower_bound,
    float eps) {
    if (!output_dev || !q_dev || !k_dev || !v_dev || !gate_dev ||
        !decay_dev || !beta_dev || !conv_q_dev || !conv_k_dev ||
        !conv_v_dev || !window_q_dev || !window_k_dev || !window_v_dev ||
        !state_dev || !dt_dev || !a_dev || !output_norm_dev || rows < 1 ||
        rows > 17 || heads < 1 || heads > 96 || head_dim != 128 ||
        conv_width != 4 || gate_lower_bound >= 0.0F || eps <= 0.0F) {
        return 0;
    }
    if (cudaSetDevice(device) != cudaSuccess) {
        return 0;
    }
    kda_shard_short_window<<<heads, head_dim>>>(
        output_dev,
        q_dev,
        k_dev,
        v_dev,
        gate_dev,
        decay_dev,
        beta_dev,
        conv_q_dev,
        conv_k_dev,
        conv_v_dev,
        window_q_dev,
        window_k_dev,
        window_v_dev,
        state_dev,
        dt_dev,
        a_dev,
        output_norm_dev,
        rows,
        heads,
        head_dim,
        conv_width,
        gate_lower_bound,
        eps);
    return cudaGetLastError() == cudaSuccess ? 1 : 0;
}
