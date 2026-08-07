#include "swarm_moe_runtime.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static float swarm_bf16_to_f32(uint16_t value) {
    uint32_t bits = ((uint32_t)value) << 16;
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float swarm_silu(float value) {
    if (value >= 0.0f) return value / (1.0f + expf(-value));
    float exponent = expf(value);
    return value * exponent / (1.0f + exponent);
}

static float swarm_int4_group_value(
    const int32_t *packed,
    const uint16_t *scales,
    size_t row,
    size_t column,
    size_t width,
    size_t group_size
) {
    size_t packed_width = (width + 7u) / 8u;
    uint32_t word = (uint32_t)packed[row * packed_width + column / 8u];
    int quantized = (int)((word >> (4u * (column % 8u))) & 0x0fu) - 8;
    size_t scale_width = (width + group_size - 1u) / group_size;
    float scale = swarm_bf16_to_f32(scales[row * scale_width + column / group_size]);
    return (float)quantized * scale;
}

const char *swarm_moe_abi_version(void) {
    return SWARM_MOE_ABI_VERSION;
}

int swarm_moe_swiglu_f32(
    const float *activation,
    size_t rows,
    size_t hidden,
    const float *gate,
    const float *up,
    size_t intermediate,
    const float *down,
    float *output
) {
    if (!activation || !gate || !up || !down || !output ||
        rows == 0 || hidden == 0 || intermediate == 0) return -1;
    float *temporary = (float *)malloc(intermediate * sizeof(float));
    if (!temporary) return -2;
    for (size_t row = 0; row < rows; ++row) {
        const float *input = activation + row * hidden;
        for (size_t middle = 0; middle < intermediate; ++middle) {
            float gate_value = 0.0f;
            float up_value = 0.0f;
            const float *gate_row = gate + middle * hidden;
            const float *up_row = up + middle * hidden;
            for (size_t column = 0; column < hidden; ++column) {
                gate_value += input[column] * gate_row[column];
                up_value += input[column] * up_row[column];
            }
            temporary[middle] = swarm_silu(gate_value) * up_value;
        }
        for (size_t column = 0; column < hidden; ++column) {
            float value = 0.0f;
            const float *down_row = down + column * intermediate;
            for (size_t middle = 0; middle < intermediate; ++middle)
                value += temporary[middle] * down_row[middle];
            output[row * hidden + column] = value;
        }
    }
    free(temporary);
    return 0;
}

int swarm_moe_swiglu_bf16(
    const float *activation,
    size_t rows,
    size_t hidden,
    const uint16_t *gate,
    const uint16_t *up,
    size_t intermediate,
    const uint16_t *down,
    float *output
) {
    if (!activation || !gate || !up || !down || !output ||
        rows == 0 || hidden == 0 || intermediate == 0) return -1;
    float *temporary = (float *)malloc(intermediate * sizeof(float));
    if (!temporary) return -2;
    for (size_t row = 0; row < rows; ++row) {
        const float *input = activation + row * hidden;
        for (size_t middle = 0; middle < intermediate; ++middle) {
            float gate_value = 0.0f;
            float up_value = 0.0f;
            const uint16_t *gate_row = gate + middle * hidden;
            const uint16_t *up_row = up + middle * hidden;
            for (size_t column = 0; column < hidden; ++column) {
                gate_value += input[column] * swarm_bf16_to_f32(gate_row[column]);
                up_value += input[column] * swarm_bf16_to_f32(up_row[column]);
            }
            temporary[middle] = swarm_silu(gate_value) * up_value;
        }
        for (size_t column = 0; column < hidden; ++column) {
            float value = 0.0f;
            const uint16_t *down_row = down + column * intermediate;
            for (size_t middle = 0; middle < intermediate; ++middle)
                value += temporary[middle] * swarm_bf16_to_f32(down_row[middle]);
            output[row * hidden + column] = value;
        }
    }
    free(temporary);
    return 0;
}

int swarm_moe_swiglu_int4_g32(
    const float *activation,
    size_t rows,
    size_t hidden,
    const int32_t *gate_packed,
    const uint16_t *gate_scales,
    const int32_t *up_packed,
    const uint16_t *up_scales,
    size_t intermediate,
    const int32_t *down_packed,
    const uint16_t *down_scales,
    size_t group_size,
    float *output
) {
    if (!activation || !gate_packed || !gate_scales || !up_packed || !up_scales ||
        !down_packed || !down_scales || !output || rows == 0 || hidden == 0 ||
        intermediate == 0 || group_size == 0 || group_size % 8u != 0) return -1;
    float *temporary = (float *)malloc(intermediate * sizeof(float));
    if (!temporary) return -2;
    for (size_t row = 0; row < rows; ++row) {
        const float *input = activation + row * hidden;
        for (size_t middle = 0; middle < intermediate; ++middle) {
            float gate_value = 0.0f;
            float up_value = 0.0f;
            for (size_t column = 0; column < hidden; ++column) {
                gate_value += input[column] * swarm_int4_group_value(
                    gate_packed, gate_scales, middle, column, hidden, group_size
                );
                up_value += input[column] * swarm_int4_group_value(
                    up_packed, up_scales, middle, column, hidden, group_size
                );
            }
            temporary[middle] = swarm_silu(gate_value) * up_value;
        }
        for (size_t column = 0; column < hidden; ++column) {
            float value = 0.0f;
            for (size_t middle = 0; middle < intermediate; ++middle) {
                value += temporary[middle] * swarm_int4_group_value(
                    down_packed, down_scales, column, middle, intermediate, group_size
                );
            }
            output[row * hidden + column] = value;
        }
    }
    free(temporary);
    return 0;
}
