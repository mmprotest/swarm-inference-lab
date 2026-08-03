/*
 * Experiment 010 downstream adapter for JustVugg/colibri.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * This file is part of the swarm-inference-lab correction pass.  It exposes a
 * deliberately narrow ABI around Colibri's header-only native MXFP4 kernel so
 * the dense Kimi K3-shaped fixture measures the same compiled arithmetic used
 * by Colibri.  It does not contain model weights and does not alter Colibri's
 * router, tokenizer, sampling, or model execution paths.
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "quant.h"

#if defined(_WIN32)
#define COLI_KIMI_API __declspec(dllexport)
#else
#define COLI_KIMI_API __attribute__((visibility("default")))
#endif

COLI_KIMI_API const char *coli_kimi_mxfp4_runtime_abi(void) {
    return "colibri-native-mxfp4-fixture-v1";
}

COLI_KIMI_API int coli_kimi_mxfp4_matmul(
    float *output,
    const float *activation,
    const uint8_t *packed,
    const uint8_t *scales,
    int batch_rows,
    int input_dimension,
    int output_dimension
) {
    if (!output || !activation || !packed || !scales || batch_rows <= 0 ||
        input_dimension <= 0 || output_dimension <= 0 || input_dimension % 32) {
        return -1;
    }
    matmul_mxfp4(
        output,
        activation,
        packed,
        scales,
        batch_rows,
        input_dimension,
        output_dimension
    );
    return 0;
}

COLI_KIMI_API int coli_kimi_mxfp4_matmul_input_slice(
    float *output,
    const float *activation_slice,
    const uint8_t *packed,
    const uint8_t *scales,
    int batch_rows,
    int original_input_dimension,
    int output_dimension,
    int input_start,
    int input_end
) {
    size_t original_row_bytes;
    size_t original_scale_columns;
    size_t slice_row_bytes;
    size_t slice_scale_columns;
    uint8_t *slice_packed;
    uint8_t *slice_scales;
    if (!output || !activation_slice || !packed || !scales || batch_rows <= 0 ||
        original_input_dimension <= 0 || output_dimension <= 0 ||
        original_input_dimension % 32 || input_start < 0 ||
        input_end <= input_start || input_end > original_input_dimension ||
        input_start % 32 || input_end % 32) {
        return -1;
    }
    original_row_bytes = (size_t)original_input_dimension / 2;
    original_scale_columns = (size_t)original_input_dimension / 32;
    slice_row_bytes = (size_t)(input_end - input_start) / 2;
    slice_scale_columns = (size_t)(input_end - input_start) / 32;
    slice_packed = (uint8_t *)malloc((size_t)output_dimension * slice_row_bytes);
    slice_scales = (uint8_t *)malloc(
        (size_t)output_dimension * slice_scale_columns
    );
    if (!slice_packed || !slice_scales) {
        free(slice_packed);
        free(slice_scales);
        return -2;
    }
#pragma omp parallel for schedule(static) if(output_dimension > 512)
    for (int row = 0; row < output_dimension; row++) {
        memcpy(
            slice_packed + (size_t)row * slice_row_bytes,
            packed + (size_t)row * original_row_bytes + (size_t)input_start / 2,
            slice_row_bytes
        );
        memcpy(
            slice_scales + (size_t)row * slice_scale_columns,
            scales + (size_t)row * original_scale_columns + (size_t)input_start / 32,
            slice_scale_columns
        );
    }
    matmul_mxfp4(
        output,
        activation_slice,
        slice_packed,
        slice_scales,
        batch_rows,
        input_end - input_start,
        output_dimension
    );
    free(slice_packed);
    free(slice_scales);
    return 0;
}

COLI_KIMI_API int coli_kimi_situ_glu(
    float *gate,
    const float *up,
    size_t count,
    float beta_one,
    float beta_two
) {
    if (!gate || !up || !count || beta_one <= 0.0f || beta_two <= 0.0f) {
        return -1;
    }
#pragma omp parallel for schedule(static) if(count > 32768)
    for (int64_t index = 0; index < (int64_t)count; index++) {
        float g = gate[index];
        float sigmoid;
        if (g >= 0.0f) {
            sigmoid = 1.0f / (1.0f + expf(-g));
        } else {
            float exponential = expf(g);
            sigmoid = exponential / (1.0f + exponential);
        }
        gate[index] = beta_one * tanhf(g / beta_one) * sigmoid *
                      beta_two * tanhf(up[index] / beta_two);
    }
    return 0;
}

COLI_KIMI_API int coli_kimi_scale_add(
    float *destination,
    const float *source,
    size_t count,
    float scale
) {
    if (!destination || !source || !count) {
        return -1;
    }
#pragma omp parallel for schedule(static) if(count > 32768)
    for (int64_t index = 0; index < (int64_t)count; index++) {
        destination[index] += scale * source[index];
    }
    return 0;
}
