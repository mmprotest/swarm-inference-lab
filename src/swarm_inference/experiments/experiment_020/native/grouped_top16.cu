// Experiment 020 exact grouped top-16 Kimi expert-stripe kernels.
//
// This DLL consumes Colibri's opaque resident tensor handles but intentionally
// depends only on their stable prefix.  It launches exactly two CUDA kernels:
// (1) grouped gate/up dot products with the exact SiTU epilogue and
// (2) grouped down projections with deterministic route-weight accumulation.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(_WIN32)
#define E020_EXPORT extern "C" __declspec(dllexport)
#else
#define E020_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr int kTopK = 16;
constexpr int kThreads = 256;
constexpr int kMaximumRows = 4;

// Stable prefix of third_party/colibri/c/backend_cuda.cu::ColiCudaTensor.
struct ColiCudaTensorPrefix {
    void* weights;
    float* scales;
    std::size_t weight_bytes;
    std::size_t scale_bytes;
    int fmt;
    int input;
    int output;
    int device;
    int group_size;
    int groups_per_row;
    std::size_t scale_count;
    int tracked;
    int weights_owned;
};

struct ExpertDescriptor {
    const std::uint8_t* gate_weights;
    const std::uint8_t* up_weights;
    const std::uint8_t* down_weights;
    const std::uint8_t* gate_scales;
    const std::uint8_t* up_scales;
    const std::uint8_t* down_scales;
};

__device__ __constant__ float kMxFp4[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

__device__ __forceinline__ float mx_scale(std::uint8_t exponent) {
    return __uint_as_float(static_cast<std::uint32_t>(exponent) << 23);
}

__device__ __forceinline__ float mx_value(
    const std::uint8_t* row,
    const std::uint8_t* scales,
    int index
) {
    const std::uint8_t packed = row[index >> 1];
    const int shift = (index & 1) * 4;
    return kMxFp4[(packed >> shift) & 15] * mx_scale(scales[index / 32]);
}

__global__ void grouped_gate_up_situ(
    float* intermediate,
    const float* activation,
    const ExpertDescriptor* descriptors,
    int rows,
    int hidden,
    int width,
    float beta,
    float linear_beta
) {
    const int output = static_cast<int>(blockIdx.x);
    const int fragment = static_cast<int>(blockIdx.y);
    if (fragment >= rows * kTopK || output >= width) return;
    const int row = fragment / kTopK;
    const ExpertDescriptor descriptor = descriptors[fragment];
    const std::size_t row_bytes = static_cast<std::size_t>(hidden + 1) / 2;
    const int groups = (hidden + 31) / 32;
    const std::uint8_t* gate_row = descriptor.gate_weights + output * row_bytes;
    const std::uint8_t* up_row = descriptor.up_weights + output * row_bytes;
    const std::uint8_t* gate_scales = descriptor.gate_scales + output * groups;
    const std::uint8_t* up_scales = descriptor.up_scales + output * groups;
    const float* input = activation + static_cast<std::size_t>(row) * hidden;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
        const float value = input[index];
        gate_sum += value * mx_value(gate_row, gate_scales, index);
        up_sum += value * mx_value(up_row, up_scales, index);
    }
    __shared__ float gate_partial[kThreads];
    __shared__ float up_partial[kThreads];
    gate_partial[threadIdx.x] = gate_sum;
    up_partial[threadIdx.x] = up_sum;
    __syncthreads();
    for (int count = blockDim.x >> 1; count; count >>= 1) {
        if (threadIdx.x < count) {
            gate_partial[threadIdx.x] += gate_partial[threadIdx.x + count];
            up_partial[threadIdx.x] += up_partial[threadIdx.x + count];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float gate = gate_partial[0];
        const float up = up_partial[0];
        const float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
        intermediate[static_cast<std::size_t>(fragment) * width + output] =
            beta * tanhf(gate / beta) * sigmoid_gate
            * linear_beta * tanhf(up / linear_beta);
    }
}

__global__ void grouped_down_weighted_reduce(
    float* output,
    const float* intermediate,
    const float* route_weights,
    const ExpertDescriptor* descriptors,
    int rows,
    int hidden,
    int width
) {
    const int hidden_output = static_cast<int>(blockIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    if (row >= rows || hidden_output >= hidden) return;

    float sums[kTopK] = {};
#pragma unroll
    for (int slot = 0; slot < kTopK; ++slot) {
        const int fragment = row * kTopK + slot;
        const ExpertDescriptor descriptor = descriptors[fragment];
        const std::size_t row_bytes = static_cast<std::size_t>(width + 1) / 2;
        const int groups = (width + 31) / 32;
        const std::uint8_t* down_row =
            descriptor.down_weights + static_cast<std::size_t>(hidden_output) * row_bytes;
        const std::uint8_t* down_scales =
            descriptor.down_scales + static_cast<std::size_t>(hidden_output) * groups;
        const float* values = intermediate + static_cast<std::size_t>(fragment) * width;
        for (int index = threadIdx.x; index < width; index += blockDim.x) {
            sums[slot] += values[index] * mx_value(down_row, down_scales, index);
        }
    }

    __shared__ float partial[kTopK][kThreads];
#pragma unroll
    for (int slot = 0; slot < kTopK; ++slot) {
        partial[slot][threadIdx.x] = sums[slot];
    }
    __syncthreads();
    for (int count = blockDim.x >> 1; count; count >>= 1) {
        if (threadIdx.x < count) {
#pragma unroll
            for (int slot = 0; slot < kTopK; ++slot) {
                partial[slot][threadIdx.x] += partial[slot][threadIdx.x + count];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float total = 0.0f;
#pragma unroll
        for (int slot = 0; slot < kTopK; ++slot) {
            total += partial[slot][0] * route_weights[row * kTopK + slot];
        }
        output[static_cast<std::size_t>(row) * hidden + hidden_output] = total;
    }
}

struct Workspace {
    ExpertDescriptor* descriptors = nullptr;
    float* intermediate = nullptr;
    std::size_t descriptor_capacity = 0;
    std::size_t intermediate_capacity = 0;
    int device = -1;
};

Workspace g_workspace;

bool reserve(void** pointer, std::size_t* capacity, std::size_t bytes) {
    if (*capacity >= bytes) return true;
    if (*pointer) cudaFree(*pointer);
    *pointer = nullptr;
    *capacity = 0;
    if (cudaMalloc(pointer, bytes) != cudaSuccess) return false;
    *capacity = bytes;
    return true;
}

bool valid_tensor(const ColiCudaTensorPrefix* tensor, int device, int input, int output) {
    return tensor && tensor->fmt == 7 && tensor->group_size == 32
        && tensor->device == device && tensor->input == input && tensor->output == output
        && tensor->weights && tensor->scales;
}

}  // namespace

E020_EXPORT int e020_kimi_grouped_top16(
    void* const* gates,
    void* const* ups,
    void* const* downs,
    float* output_device,
    const float* activation_device,
    const float* route_weights_device,
    int rows,
    float beta,
    float linear_beta,
    double* cuda_ms
) {
    if (!gates || !ups || !downs || !output_device || !activation_device
        || !route_weights_device || !cuda_ms || rows < 1 || rows > kMaximumRows
        || beta <= 0.0f || linear_beta <= 0.0f) return 0;
    const auto* first = static_cast<const ColiCudaTensorPrefix*>(gates[0]);
    if (!first) return 0;
    const int device = first->device;
    const int hidden = first->input;
    const int width = first->output;
    if (cudaSetDevice(device) != cudaSuccess) return 0;

    const int fragments = rows * kTopK;
    std::vector<ExpertDescriptor> host(static_cast<std::size_t>(fragments));
    for (int index = 0; index < fragments; ++index) {
        const auto* gate = static_cast<const ColiCudaTensorPrefix*>(gates[index]);
        const auto* up = static_cast<const ColiCudaTensorPrefix*>(ups[index]);
        const auto* down = static_cast<const ColiCudaTensorPrefix*>(downs[index]);
        if (!valid_tensor(gate, device, hidden, width)
            || !valid_tensor(up, device, hidden, width)
            || !valid_tensor(down, device, width, hidden)) return 0;
        host[static_cast<std::size_t>(index)] = {
            static_cast<const std::uint8_t*>(gate->weights),
            static_cast<const std::uint8_t*>(up->weights),
            static_cast<const std::uint8_t*>(down->weights),
            reinterpret_cast<const std::uint8_t*>(gate->scales),
            reinterpret_cast<const std::uint8_t*>(up->scales),
            reinterpret_cast<const std::uint8_t*>(down->scales),
        };
    }

    if (g_workspace.device != device) {
        if (g_workspace.descriptors) cudaFree(g_workspace.descriptors);
        if (g_workspace.intermediate) cudaFree(g_workspace.intermediate);
        g_workspace = {};
        g_workspace.device = device;
    }
    const std::size_t descriptor_bytes = host.size() * sizeof(ExpertDescriptor);
    const std::size_t intermediate_bytes =
        static_cast<std::size_t>(fragments) * width * sizeof(float);
    if (!reserve(reinterpret_cast<void**>(&g_workspace.descriptors),
                 &g_workspace.descriptor_capacity, descriptor_bytes)
        || !reserve(reinterpret_cast<void**>(&g_workspace.intermediate),
                    &g_workspace.intermediate_capacity, intermediate_bytes)) return 0;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) return 0;
    bool ok = cudaMemcpyAsync(
        g_workspace.descriptors,
        host.data(),
        descriptor_bytes,
        cudaMemcpyHostToDevice
    ) == cudaSuccess;
    if (ok) ok = cudaEventRecord(start) == cudaSuccess;
    if (ok) {
        grouped_gate_up_situ<<<dim3(width, fragments), kThreads>>>(
            g_workspace.intermediate,
            activation_device,
            g_workspace.descriptors,
            rows,
            hidden,
            width,
            beta,
            linear_beta
        );
        grouped_down_weighted_reduce<<<dim3(hidden, rows), kThreads>>>(
            output_device,
            g_workspace.intermediate,
            route_weights_device,
            g_workspace.descriptors,
            rows,
            hidden,
            width
        );
        ok = cudaGetLastError() == cudaSuccess && cudaEventRecord(stop) == cudaSuccess
            && cudaEventSynchronize(stop) == cudaSuccess;
    }
    float elapsed = 0.0f;
    if (ok) ok = cudaEventElapsedTime(&elapsed, start, stop) == cudaSuccess;
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    *cuda_ms = static_cast<double>(elapsed);
    return ok ? 1 : 0;
}

E020_EXPORT void e020_kimi_grouped_release() {
    if (g_workspace.device >= 0) cudaSetDevice(g_workspace.device);
    if (g_workspace.descriptors) cudaFree(g_workspace.descriptors);
    if (g_workspace.intermediate) cudaFree(g_workspace.intermediate);
    g_workspace = {};
}

E020_EXPORT int e020_kimi_grouped_physical_launches() { return 2; }

E020_EXPORT int e020_kimi_grouped_supports_compute_capability(int major, int minor) {
#if defined(__CUDA_ARCH__)
    (void)major;
    (void)minor;
    return 1;
#else
    return (major == 8 && minor == 6) || (major == 12 && minor == 0);
#endif
}
