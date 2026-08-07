#ifndef SWARM_MOE_RUNTIME_H
#define SWARM_MOE_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define SWARM_MOE_API __declspec(dllexport)
#else
#define SWARM_MOE_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SWARM_MOE_ABI_VERSION "swarm-colibri-moe-v2"

/*
 * Adapter-neutral reference ABI.  Projection storage is row-major:
 * gate/up=[intermediate, hidden], down=[hidden, intermediate].  Callers retain
 * responsibility for adapter-validated routing and quantization metadata.
 */
SWARM_MOE_API const char *swarm_moe_abi_version(void);

SWARM_MOE_API int swarm_moe_swiglu_f32(
    const float *activation,
    size_t rows,
    size_t hidden,
    const float *gate,
    const float *up,
    size_t intermediate,
    const float *down,
    float *output
);

SWARM_MOE_API int swarm_moe_swiglu_bf16(
    const float *activation,
    size_t rows,
    size_t hidden,
    const uint16_t *gate,
    const uint16_t *up,
    size_t intermediate,
    const uint16_t *down,
    float *output
);

/*
 * compressed-tensors symmetric W4A16 group quantization. Eight signed INT4
 * values (stored with a +8 offset) occupy each int32 along the input axis;
 * BF16 scales are [output, ceil(input / group_size)].
 */
SWARM_MOE_API int swarm_moe_swiglu_int4_g32(
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
);

#ifdef __cplusplus
}
#endif

#endif
