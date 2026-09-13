#include "backend_cuda.h"

#include "backend_gpu_compat.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <chrono>
#include <atomic>
#include <mutex>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

#ifdef COLI_ANS
#include <dietgpu/ans/GpuANSCodec.h>
#include <dietgpu/utils/StackDeviceMemory.h>
#endif

#ifndef COLI_CUDA_MIN_CC
#define COLI_CUDA_MIN_CC 0
#endif
#ifndef COLI_CUDA_HAS_FORWARD_PTX
#define COLI_CUDA_HAS_FORWARD_PTX 0
#endif

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
#include <sys/stat.h>
#endif

struct RaggedKVEntry {
    const void *key;
    const float *host_l,*host_r;
    float *latent,*rope;
    int length,capacity,K,R;
};

struct ColiCudaTensor {
    void *weights;
    float *scales;
    size_t weight_bytes;
    size_t scale_bytes;        /* fmt=7 uses byte UE8M0 scales; other formats f32 */
    int fmt, I, O, device;
    int gs;                    /* quant group size; 0 = per-row scales (#334) */
    int ng;                    /* number of scale groups per row = ceil(I/gs) for fmt=4 */
    size_t scale_count;        /* floats in `scales`: O per-row, O*ng grouped */
    int tracked;
    int weights_owned;
#ifdef COLI_ANS
    size_t archive_bytes;
    int compressed;
#endif
    RaggedKVEntry ragged[512];
    int ragged_count;
};

#ifdef COLI_ANS
struct AnsArenaChunk { uint8_t *p; size_t used,cap; };
#endif
typedef struct {
    int device;
    int compute_major,compute_minor;
    size_t shared_memory_default_bytes,shared_memory_optin_bytes;
    float *x, *y, *gate, *up;
    size_t x_cap, y_cap, gate_cap, up_cap;
    uint8_t *qx; float *qscale;
    size_t qx_cap, qscale_cap;
    float *host_x,*host_y,*host_kv; size_t host_x_cap,host_y_cap,host_kv_cap;
    float *aq,*al,*ar,*ac; size_t aq_cap,al_cap,ar_cap,ac_cap;
    float *pipe_buf[27]; size_t pipe_cap[27];   /* scratch persistenti del resident pipeline */
    cudaStream_t stream;
    cudaEvent_t ev_done; int ev_done_ok;        /* resident-group issue completion (#431 PR-C0) */
    cudaEvent_t profile_start,profile_stop; int profile_started;
    void *group_desc; size_t group_desc_cap;
    size_t tensor_count, tensor_bytes;
    int group_pending; size_t group_pending_bytes;   /* async expert-group in flight (Inc.4) */
#ifdef COLI_ANS
    void *ans_raw; size_t ans_raw_cap;
    void *ans_host; size_t ans_host_cap;
    int ans_copy_pending;
    dietgpu::StackDeviceMemory *ans_scratch;
    std::vector<AnsArenaChunk> *ans_chunks;
#endif
} DeviceContext;

typedef struct {
    const void *g,*u,*d; const float *gs,*us,*ds;
    int gf,uf,df,rows,offset;
    int ggs,ugs,dgs;      /* per-tensor quant group size; 0 = per-row scales (#334 fmt=4) */
} GroupDesc;

static DeviceContext g_ctx[COLI_CUDA_MAX_DEVICES];
static int g_nctx;
static uint64_t g_group_calls,g_group_experts,g_group_rows;
static double g_group_h2d_ms,g_group_kernel_ms,g_group_d2h_ms;
static std::mutex g_group_stats_mu;
static uint64_t g_kimi_calls,g_kimi_rows,g_kimi_h2d_bytes,g_kimi_d2h_bytes;
static double g_kimi_h2d_ms,g_kimi_kernel_ms,g_kimi_d2h_ms;
static double g_kimi_gate_ms,g_kimi_up_ms,g_kimi_situ_ms,g_kimi_down_ms;
static double g_kimi_gate_up_pair_ms;
static std::atomic<int> g_kimi_telemetry_mode{1};
static std::atomic<int> g_kimi_up_first{0};
static std::atomic<int> g_kimi_fused_gate_up{1};
static std::mutex g_kimi_stats_mu;
static uint64_t g_kimi_router_calls;
static double g_kimi_router_logits_ms,g_kimi_router_selection_ms,g_kimi_router_d2h_ms;
static uint64_t g_kimi_dense_calls;
static double g_kimi_dense_kernel_ms;
#ifdef COLI_ANS
static FILE *g_ans_sidecar;
static int g_ans_sidecar_pack;
#if defined(__linux__)
static int g_ans_direct_fd=-1;
static off_t g_ans_direct_off;
#endif
static uint64_t g_ans_load_records;
static double g_ans_header_s,g_ans_read_s,g_ans_stage_s,g_ans_enqueue_s;
static int g_ans_profile_printed;
static double ans_now_s(){
    using clock=std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}
#endif

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "[CUDA] %s: %s\n", what, cudaGetErrorString(err));
    (void)cudaGetLastError();   /* consume the sticky error: a failed call must
                                   not poison the next launch's error check */
    return 0;
}

static DeviceContext *find_ctx(int device) {
    for (int i = 0; i < g_nctx; i++) if (g_ctx[i].device == device) return &g_ctx[i];
    return nullptr;
}

/* cudaSetDevice on every call doubles expert-matmul time on 2 GPUs when the
 * serial expert loop alternates devices (measured on RTX 5090 + 4090: 14.3s
 * -> 25.4s per 32 tokens). The current device is per-thread in the CUDA
 * runtime, so a thread-local cache skips the redundant switches. */
static thread_local int g_current_device = -1;

static int select_ctx(DeviceContext *ctx) {
    if (!ctx) return 0;
    if (g_current_device == ctx->device) return 1;
    if (!cuda_ok(cudaSetDevice(ctx->device), "select device")) return 0;
    g_current_device = ctx->device;
    return 1;
}

/* fmt=6 (E8/IQ3) geometry, mirroring quant.h. A super-block packs 256 weights
 * into 98 bytes: 64 codebook indices, 8 words of (4x7 signs + 4-bit sub-scale),
 * and one fp16 super-scale. Scales live INSIDE the block, so fmt=6 tensors carry
 * no separate scale array (#452). */
#define COLI_E8_QK      256
#define COLI_E8_SUB      32
#define COLI_E8_BBYTES   98

__host__ __device__ static size_t row_bytes(int fmt, int I) {
    if (fmt == 0) return (size_t)I * sizeof(float);
    if (fmt == 1) return (size_t)I;
    if (fmt == 2 || fmt == 4) return (size_t)(I + 1) / 2;   /* fmt=4: same packed int4 */
    if (fmt == 3) return (size_t)(I + 3) / 4;
    if (fmt == 4) return (size_t)(I + 1) / 2;   /* grouped int4: nibbles like fmt 2 */
    if (fmt == 6) return (size_t)(((int64_t)I + COLI_E8_QK - 1) / COLI_E8_QK) * COLI_E8_BBYTES;
    if (fmt == 7) return (size_t)(I + 1) / 2;   /* native E2M1 nibbles */
    if (fmt == 8) return (size_t)I * sizeof(uint16_t); /* resident BF16 embedding */
    return 0;
}

/* The E8 codebook, uploaded once per device from quant.h's e8_grid so the table
 * has a single source of truth and cannot drift from the CPU decoder. */
__constant__ uint8_t c_e8_grid[256][4];
__device__ __constant__ float c_mx4_lut[16] = {
    0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f,-0.f,-.5f,-1.f,-1.5f,-2.f,-3.f,-4.f,-6.f
};

__device__ __forceinline__ float mx4_scale_dev(uint8_t exponent) {
    return __uint_as_float((uint32_t)exponent << 23);
}

/* A super-block is 98 bytes, so nothing inside it is guaranteed 4- or 2-byte
 * aligned: assemble the words byte-wise instead of dereferencing. */
__device__ __forceinline__ uint32_t e8_ld_u32(const uint8_t *p){
    return (uint32_t)p[0] | ((uint32_t)p[1]<<8) | ((uint32_t)p[2]<<16) | ((uint32_t)p[3]<<24);
}
/* Mirrors e8_fp16_to_f32 rather than calling __half2float, so the two decoders
 * cannot disagree on subnormals. */
__device__ __forceinline__ float e8_fp16(const uint8_t *p){
    uint16_t h = (uint16_t)p[0] | ((uint16_t)p[1]<<8);
    uint32_t sign=(uint32_t)(h&0x8000)<<16, exp=(h>>10)&0x1F, man=h&0x3FF, bits;
    if (!exp)         bits = man ? (sign|((127u-15u+1u-1u)<<23)|(man<<13)) : sign;
    else if (exp==31) bits = sign|0x7F800000u|(man<<13);
    else              bits = sign|((exp+112u)<<23)|(man<<13);
    float f; memcpy(&f,&bits,4); return f;
}
/* Expand one 32-weight sub-block; mirrors e8_expand_sub in quant.h. */
__device__ __forceinline__ void e8_expand_sub_dev(const uint8_t *blk, int ib, float d, float *out){
    uint32_t word = e8_ld_u32(blk + COLI_E8_QK/4 + ib*4);
    float db = d * (0.5f + (float)((word>>28)&0xF)) * 0.5f;
    const uint8_t *idx = blk + ib*8;
    for (int l=0;l<4;l++){
        uint32_t seven=(word>>(7*l))&0x7F;
        const uint8_t *g0=c_e8_grid[idx[l*2+0]], *g1=c_e8_grid[idx[l*2+1]];
        int par=0;
        for (int j=0;j<8;j++){
            int neg = j<7 ? (int)((seven>>j)&1) : 0;
            if (j<7) par^=neg; else neg=par;        /* odd parity closes the block */
            float mag = (j<4 ? (float)g0[j] : (float)g1[j-4]) * 0.5f;
            out[l*8+j] = neg ? -mag*db : mag*db;
        }
    }
}

__device__ static float weight_at(const void *weights, int fmt, size_t row, int i) {
    const uint8_t *base = static_cast<const uint8_t *>(weights) + row;
    if (fmt == 0) return reinterpret_cast<const float *>(base)[i];
    if (fmt == 1) return static_cast<float>(reinterpret_cast<const int8_t *>(base)[i]);
    const uint8_t *q = base;
    if (fmt == 2 || fmt == 4) {                               /* fmt=4: same nibble layout */
        uint8_t v = q[i >> 1];
        int n=(i&1)?(v>>4):(v&15); return static_cast<float>(n&8?n-16:n);
    }
    uint8_t v = q[i >> 2];
    return static_cast<float>(((v >> ((i & 3) * 2)) & 3) - 2);
}

/* Scale for output `row`, input element `k`. fmt=4 (grouped int4) stores ng
 * scales per row at scales[row*ng + k/gs]; every other quantized format has
 * one scale per row at scales[row]. Mirrors quant_matmul's fmt==4 branch so the
 * attention absorb kernels apply per-group scales instead of the per-row
 * (fmt=2) semantic that crashed #298's g64 kv_b. */
__device__ static float absorb_scale(const float *wscale, int fmt, int gs, int ng, int row, int k) {
    if (!fmt) return 1.f;
    if (fmt != 4) return wscale[row];
    int g = k / gs; if (g >= ng) g = ng - 1;   /* tail of the last (partial) group */
    return wscale[(size_t)row * ng + g];
}

__global__ static void offset_to_signed_s4(uint8_t *q,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n)q[i]^=0x88;
}

__global__ static void quant_matmul(float *y, const float *x, const void *weights,
                                    const float *scales, int fmt, int S, int I, int O,
                                    size_t rb, int gs, int ng) {
    int o = blockIdx.x;
    int s = blockIdx.y;
    float sum = 0.0f;
    size_t row = (size_t)o * rb;
    const float *xs = x + (size_t)s * I;
    if (fmt == 6) {
        /* E8/IQ3: decode is per 32-weight sub-block, so threads stride over
         * sub-blocks rather than elements -- expanding once per 32 weights
         * instead of redoing the word/parity work for every element. */
        const uint8_t *wrow = static_cast<const uint8_t *>(weights) + row;
        int nsub = (I + COLI_E8_SUB - 1) / COLI_E8_SUB;
        for (int sb = threadIdx.x; sb < nsub; sb += blockDim.x) {
            const uint8_t *blk = wrow + (size_t)(sb / (COLI_E8_QK/COLI_E8_SUB)) * COLI_E8_BBYTES;
            float w[COLI_E8_SUB];
            e8_expand_sub_dev(blk, sb % (COLI_E8_QK/COLI_E8_SUB), e8_fp16(blk+96), w);
            int off = sb*COLI_E8_SUB, n = I-off < COLI_E8_SUB ? I-off : COLI_E8_SUB;
            for (int k=0;k<n;k++) sum += xs[off+k]*w[k];
        }
    } else if (fmt == 7) {
        /* Native Kimi K3 routed-expert layout.  Keep the E2M1 nibbles and
         * UE8M0 group-32 exponents byte-exact on device: no dequantized weight
         * tensor or layout-conversion launch exists on this path. */
        const uint8_t *wrow = static_cast<const uint8_t *>(weights) + row;
        const uint8_t *scl = reinterpret_cast<const uint8_t *>(scales) + (size_t)o * ng;
        for (int i = threadIdx.x; i < I; i += blockDim.x) {
            uint8_t packed = wrow[i >> 1];
            int nibble = (i & 1) ? (packed >> 4) : (packed & 15);
            int group = i / 32;
            if (group >= ng) group = ng - 1;
            sum += xs[i] * c_mx4_lut[nibble] * mx4_scale_dev(scl[group]);
        }
    } else if (fmt == 4) {
        /* Grouped int4: one f32 scale per gs elements along I (ng groups per row).
         * Scale layout: scales[o*ng + g]. Each thread strides through I, applying
         * the appropriate group scale as it crosses group boundaries. This matches
         * the CPU matmul_i4_grouped accumulation exactly. */
        const float *scl = scales + (size_t)o * ng;
        for (int i = threadIdx.x; i < I; i += blockDim.x) {
            int g = i / gs;
            if (g >= ng) g = ng - 1;  /* tail elements in the last (partial) group */
            sum += xs[i] * weight_at(weights, fmt, row, i) * scl[g];
        }
    } else {
        for (int i = threadIdx.x; i < I; i += blockDim.x)
            sum += xs[i] * weight_at(weights, fmt, row, i);
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int n = blockDim.x >> 1; n; n >>= 1) {
        if (threadIdx.x < n) partial[threadIdx.x] += partial[threadIdx.x + n];
        __syncthreads();
    }
    if (!threadIdx.x)
        y[(size_t)s * O + o] = (fmt && fmt != 4 && fmt != 6 && fmt != 7)
            ? partial[0] * scales[o] : partial[0];
}

/* fmt=6 activation rotation, y = Q^T x for Q = D*H/sqrt(n) (#452). One block per
 * row; the power-of-two block is staged in shared memory, capping n at 4096
 * floats -- which covers every block GLM produces (6144 -> 2048+4096, 1536 ->
 * 512+1024). The sign stream is regenerated in-kernel from the same xorshift64*
 * that quant.h's e8_signs uses, so no rotation data is stored or uploaded.
 *
 * Placement note: all routed experts of a layer share one gate/up input, so that
 * rotation belongs to the CALLER (once per layer). This kernel exists for the
 * down projection, whose input is the per-expert silu(gate)*up product and so
 * cannot be shared -- mirroring colibri.c's split at moe(). */
__global__ static void e8_rot_rows_kernel(float *rows, int dim, int off, int n){
    extern __shared__ float sh[];
    __shared__ uint8_t sbits[4096/8];
    if (!threadIdx.x) {
        uint64_t s = 417u + (uint64_t)n;
        for (int i=0;i<(n+7)/8;i++){
            s^=s>>12; s^=s<<25; s^=s>>27;
            sbits[i] = (uint8_t)((s*2685821657736338717ULL)>>56);
        }
    }
    __syncthreads();
    float *row = rows + (size_t)blockIdx.x*dim + off;
    for (int i=threadIdx.x;i<n;i+=blockDim.x){
        float v=row[i];
        sh[i] = (sbits[i>>3]>>(i&7)&1) ? -v : v;
    }
    __syncthreads();
    for (int len=1;len<n;len<<=1){
        for (int j=threadIdx.x;j<n/2;j+=blockDim.x){
            int i = (j/len)*(len<<1) + (j%len);
            float u=sh[i], v=sh[i+len];
            sh[i]=u+v; sh[i+len]=u-v;
        }
        __syncthreads();
    }
    float sc=rsqrtf((float)n);
    for (int i=threadIdx.x;i<n;i+=blockDim.x) row[i]=sh[i]*sc;
}

/* Rotate nr rows in place, tiling non-power-of-two dims block-diagonally exactly
 * as e8_rot_rows does. Returns 0 if a block exceeds the shared-memory cap. */
static int e8_rot_rows_dev(float *rows, int nr, int dim, cudaStream_t stream){
    int off = 0;
    while (off < dim) {
        int rem = dim-off, b = rem & (-rem);
        while (b > 4096) b >>= 1;
        e8_rot_rows_kernel<<<(unsigned)nr, 256, (size_t)b*sizeof(float), stream>>>(rows, dim, off, b);
        if (cudaGetLastError() != cudaSuccess) return 0;
        off += b;
    }
    return 1;
}

__global__ static void silu_mul(float *gate, const float *up, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = gate[i];
        gate[i] = (v / (1.0f + expf(-v))) * up[i];
    }
}

/* H014-030d: reuse one quantized weight load across a small, exact row set.
 * Each row keeps the same per-thread input traversal and the same 256-way
 * reduction tree as quant_matmul, so this changes weight traffic rather than
 * numerical order.  The production batch contract deliberately exposes only
 * the incrementally tested 2/4/8 specializations. */
template<int ROWS>
__global__ static void quant_matmul_rows_reuse(
        float *y, const float *x, const void *weights, const float *scales,
        int fmt, int I, int O, size_t rb, int gs, int ng) {
    int o = blockIdx.x;
    size_t row = (size_t)o * rb;
    float sums[ROWS];
#pragma unroll
    for (int s=0;s<ROWS;s++) sums[s]=0.0f;

    if (fmt == 4) {
        const float *scl = scales + (size_t)o * ng;
        for (int i=threadIdx.x;i<I;i+=blockDim.x) {
            int g=i/gs;if(g>=ng)g=ng-1;
            float raw=weight_at(weights,fmt,row,i);
#pragma unroll
            for (int s=0;s<ROWS;s++)
                sums[s] += x[(size_t)s*I+i] * raw * scl[g];
        }
    } else {
        for (int i=threadIdx.x;i<I;i+=blockDim.x) {
            float raw=weight_at(weights,fmt,row,i);
#pragma unroll
            for (int s=0;s<ROWS;s++)
                sums[s] += x[(size_t)s*I+i] * raw;
        }
    }

    __shared__ float partial[ROWS][256];
#pragma unroll
    for (int s=0;s<ROWS;s++) partial[s][threadIdx.x]=sums[s];
    __syncthreads();
    for (int n=blockDim.x>>1;n;n>>=1) {
        if (threadIdx.x<n) {
#pragma unroll
            for (int s=0;s<ROWS;s++)
                partial[s][threadIdx.x] += partial[s][threadIdx.x+n];
        }
        __syncthreads();
    }
    if (!threadIdx.x) {
#pragma unroll
        for (int s=0;s<ROWS;s++)
            y[(size_t)s*O+o] = (fmt && fmt != 4)
                ? partial[s][0] * scales[o] : partial[s][0];
    }
}

/* Kimi's gate and up projections have identical geometry and input.  Decode
 * otherwise pays the first-launch/stream penalty twice across two independent
 * launches.  This native-fmt pair kernel preserves each dot-product reduction
 * order while sharing the activation load and launch. */
__global__ static void mxfp4_matmul_pair(float *gate_y, float *up_y,
                                         const float *x,
                                         const uint8_t *gate_w, const uint8_t *up_w,
                                         const uint8_t *gate_s, const uint8_t *up_s,
                                         int S, int I, int O, size_t rb, int ng) {
    int o = blockIdx.x, s = blockIdx.y;
    const float *xs = x + (size_t)s * I;
    const uint8_t *gw = gate_w + (size_t)o * rb;
    const uint8_t *uw = up_w + (size_t)o * rb;
    const uint8_t *gs = gate_s + (size_t)o * ng;
    const uint8_t *us = up_s + (size_t)o * ng;
    float gate_sum = 0.f, up_sum = 0.f;
    for (int i=threadIdx.x; i<I; i+=blockDim.x) {
        uint8_t gb=gw[i>>1], ub=uw[i>>1];
        int shift=(i&1)*4, group=i/32;
        float activation=xs[i];
        gate_sum += activation*c_mx4_lut[(gb>>shift)&15]*mx4_scale_dev(gs[group]);
        up_sum += activation*c_mx4_lut[(ub>>shift)&15]*mx4_scale_dev(us[group]);
    }
    __shared__ float gate_partial[256], up_partial[256];
    gate_partial[threadIdx.x]=gate_sum; up_partial[threadIdx.x]=up_sum;
    __syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){
        if(threadIdx.x<n){
            gate_partial[threadIdx.x]+=gate_partial[threadIdx.x+n];
            up_partial[threadIdx.x]+=up_partial[threadIdx.x+n];
        }
        __syncthreads();
    }
    if(!threadIdx.x){
        gate_y[(size_t)s*O+o]=gate_partial[0];
        up_y[(size_t)s*O+o]=up_partial[0];
    }
}

/* H014-028: batch rows share the same immutable expert weights.  The previous
 * grid placed the row on blockIdx.y, so every row independently reloaded every
 * MXFP4 nibble and UE8M0 scale.  These exact-row kernels keep one output in one
 * block and accumulate all rows while each weight value is resident.  The
 * per-row input traversal and tree-reduction order are unchanged.  Template
 * arity prevents a large worst-case accumulator array from penalising the
 * currently certified two-row path. */
template<int ROWS>
__global__ static void mxfp4_matmul_pair_rows(float *gate_y, float *up_y,
                                              const float *x,
                                              const uint8_t *gate_w,
                                              const uint8_t *up_w,
                                              const uint8_t *gate_s,
                                              const uint8_t *up_s,
                                              int I, int O, size_t rb, int ng) {
    int o = blockIdx.x;
    const uint8_t *gw = gate_w + (size_t)o * rb;
    const uint8_t *uw = up_w + (size_t)o * rb;
    const uint8_t *gs = gate_s + (size_t)o * ng;
    const uint8_t *us = up_s + (size_t)o * ng;
    float gate_sum[ROWS] = {};
    float up_sum[ROWS] = {};
    for (int i=threadIdx.x; i<I; i+=blockDim.x) {
        uint8_t gb=gw[i>>1], ub=uw[i>>1];
        int shift=(i&1)*4, group=i/32;
        float gate_value=c_mx4_lut[(gb>>shift)&15];
        float up_value=c_mx4_lut[(ub>>shift)&15];
        float gate_scale=mx4_scale_dev(gs[group]);
        float up_scale=mx4_scale_dev(us[group]);
#pragma unroll
        for (int s=0; s<ROWS; ++s) {
            float activation=x[(size_t)s*I+i];
            gate_sum[s] += activation*gate_value*gate_scale;
            up_sum[s] += activation*up_value*up_scale;
        }
    }
    __shared__ float gate_partial[ROWS][256], up_partial[ROWS][256];
#pragma unroll
    for (int s=0; s<ROWS; ++s) {
        gate_partial[s][threadIdx.x]=gate_sum[s];
        up_partial[s][threadIdx.x]=up_sum[s];
    }
    __syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){
        if(threadIdx.x<n){
#pragma unroll
            for (int s=0; s<ROWS; ++s) {
                gate_partial[s][threadIdx.x]+=gate_partial[s][threadIdx.x+n];
                up_partial[s][threadIdx.x]+=up_partial[s][threadIdx.x+n];
            }
        }
        __syncthreads();
    }
    if(!threadIdx.x){
#pragma unroll
        for (int s=0; s<ROWS; ++s) {
            gate_y[(size_t)s*O+o]=gate_partial[s][0];
            up_y[(size_t)s*O+o]=up_partial[s][0];
        }
    }
}

template<int ROWS>
__global__ static void mxfp4_matmul_rows(float *y, const float *x,
                                         const uint8_t *weights,
                                         const uint8_t *scales,
                                         int I, int O, size_t rb, int ng) {
    int o = blockIdx.x;
    const uint8_t *wrow = weights + (size_t)o * rb;
    const uint8_t *scl = scales + (size_t)o * ng;
    float sum[ROWS] = {};
    for (int i=threadIdx.x; i<I; i+=blockDim.x) {
        uint8_t packed=wrow[i>>1];
        int shift=(i&1)*4, group=i/32;
        float value=c_mx4_lut[(packed>>shift)&15];
        float scale=mx4_scale_dev(scl[group]);
#pragma unroll
        for (int s=0; s<ROWS; ++s)
            sum[s] += x[(size_t)s*I+i]*value*scale;
    }
    __shared__ float partial[ROWS][256];
#pragma unroll
    for (int s=0; s<ROWS; ++s) partial[s][threadIdx.x]=sum[s];
    __syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){
        if(threadIdx.x<n){
#pragma unroll
            for (int s=0; s<ROWS; ++s)
                partial[s][threadIdx.x]+=partial[s][threadIdx.x+n];
        }
        __syncthreads();
    }
    if(!threadIdx.x){
#pragma unroll
        for (int s=0; s<ROWS; ++s) y[(size_t)s*O+o]=partial[s][0];
    }
}

__global__ static void kimi_situ_mul(float *gate, const float *up, size_t n,
                                      float beta, float linear_beta) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i], u = up[i];
        float sigmoid_g = 1.0f / (1.0f + expf(-g));
        gate[i] = beta * tanhf(g / beta) * sigmoid_g
                * linear_beta * tanhf(u / linear_beta);
    }
}

/* Four warps share one A tile and compute 16x64 outputs.  This matters for
 * prefill: the first prototype reloaded/converter A once per 16 output cols. */
__global__ static void w4a16_matmul(float *y,const float *x,const uint8_t *w,
                                    const float *scale,int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+warp*16;
    __shared__ __half ah[256],bh[4][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    size_t rb=(size_t)(K+1)/2;
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){
            int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);
        }
        for(int z=lane;z<256;z+=32){
            int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gn*rb+(gk>>1)];int a=(gk&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*scale[gn];}
            bh[warp][z]=__float2half(v);           /* [Ntile,Ktile] == B col-major */
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[4][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

/* Gate and up use the same input.  Eight warps compute both 16x64 projections
 * while sharing the FP32->FP16 conversion of A. */
__global__ static void w4a16_gate_up(float *gate,float *up,const float *x,
        const uint8_t *gw,const uint8_t *uw,const float *gs,const float *us,
        int M,int K,int N){
#if __CUDA_ARCH__ >= 700
    using namespace nvcuda;int warp=threadIdx.x>>5,lane=threadIdx.x&31,which=warp&1,tile=warp>>1;
    int m0=blockIdx.y*16,n0=blockIdx.x*64+tile*16;const uint8_t *w=which?uw:gw;
    const float *scale=which?us:gs;float *y=which?up:gate;size_t rb=(size_t)(K+1)/2;
    __shared__ __half ah[256],bh[8][256];
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
    for(int k0=0;k0<K;k0+=16){
        for(int z=threadIdx.x;z<256;z+=blockDim.x){int m=z/16,k=z%16,gm=m0+m,gk=k0+k;
            ah[z]=(gm<M&&gk<K)?__float2half(x[(size_t)gm*K+gk]):__float2half(0.f);}
        for(int z=lane;z<256;z+=32){int n=z/16,gk=k0+(z%16),gn=n0+n;float v=0.f;
            if(gn<N&&gk<K){uint8_t q=w[(size_t)gn*rb+(gk>>1)];int a=(gk&1)?q>>4:q&15;
                v=(float)(a&8?a-16:a)*scale[gn];}bh[warp][z]=__float2half(v);}
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
        wmma::load_matrix_sync(af,ah,16);wmma::load_matrix_sync(bf,bh[warp],16);
        wmma::mma_sync(acc,af,bf,acc);__syncthreads();
    }
    __shared__ float out[8][256];wmma::store_matrix_sync(out[warp],acc,16,wmma::mem_row_major);__syncwarp();
    for(int z=lane;z<256;z+=32){int m=z/16,n=z%16;
        if(m0+m<M&&n0+n<N)y[(size_t)(m0+m)*N+n0+n]=out[warp][z];}
#endif
}

__global__ static void quantize_s4_rows(uint8_t *q,float *scale,const float *x,int S,int K){
    int s=blockIdx.x; if(s>=S)return; const float *xs=x+(size_t)s*K;
    float v=0; for(int i=threadIdx.x;i<K;i+=blockDim.x)v=fmaxf(v,fabsf(xs[i]));
    __shared__ float m[256]; m[threadIdx.x]=v; __syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)m[threadIdx.x]=fmaxf(m[threadIdx.x],m[threadIdx.x+n]);__syncthreads();}
    float sc=m[0]>0?m[0]/7.f:1.f; if(!threadIdx.x)scale[s]=sc;
    uint8_t *dst=q+(size_t)s*((K+1)/2);
    for(int b=threadIdx.x;b<(K+1)/2;b+=blockDim.x){
        int i=b*2,a=__float2int_rn(xs[i]/sc),c=i+1<K?__float2int_rn(xs[i+1]/sc):0;
        a=max(-8,min(7,a)); c=max(-8,min(7,c)); dst[b]=(uint8_t)((a&15)|((c&15)<<4));
    }
}

__global__ static void grouped_s4_wmma(float *y,const uint8_t *x,const float *xscale,
                                        const GroupDesc *desc,int K,int O,int which){
#if __CUDA_ARCH__ >= 750
    using namespace nvcuda;
    int warp=threadIdx.x/32,lane=threadIdx.x%32,tile=blockIdx.x*8+warp,c=blockIdx.y;
    if(tile*8>=O)return; GroupDesc d=desc[c];
    const void *w=which==0?d.g:(which==1?d.u:d.d);
    const float *ws=which==0?d.gs:(which==1?d.us:d.ds);
    int fmt=which==0?d.gf:(which==1?d.uf:d.df);
    if(fmt!=2)return;
    wmma::fragment<wmma::accumulator,8,8,32,int> acc; wmma::fill_fragment(acc,0);
    const uint8_t *a=x+(size_t)d.offset*((K+1)/2);
    const uint8_t *b=(const uint8_t*)w+(size_t)(tile*8)*((K+1)/2);
    for(int k=0;k<K;k+=32){
        wmma::fragment<wmma::matrix_a,8,8,32,wmma::experimental::precision::s4,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,8,8,32,wmma::experimental::precision::s4,wmma::col_major> bf;
        wmma::load_matrix_sync(af,a+k/2,K);
        wmma::load_matrix_sync(bf,b+k/2,K);
        wmma::mma_sync(acc,af,bf,acc);
    }
    __shared__ int out[8][64]; wmma::store_matrix_sync(out[warp],acc,8,wmma::mem_row_major);
    for(int i=lane;i<64;i+=32){int s=i/8,o=tile*8+i%8;
        if(s<d.rows&&o<O)y[(size_t)(d.offset+s)*O+o]=(float)out[warp][i]*xscale[d.offset+s]*ws[o];}
#endif
}

__global__ static void grouped_hidden(float *y,const float *x,const GroupDesc *desc,
                                      int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    const void *w=which?d.u:d.g; const float *sc=which?d.us:d.gs; int fmt=which?d.uf:d.gf;
    size_t rb=row_bytes(fmt,D),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*D;
    float sum=0; for(int i=threadIdx.x;i<D;i+=blockDim.x) sum+=xs[i]*weight_at(w,fmt,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*I+o]=p[0]*(fmt?sc[o]:1.f);
}

__global__ static void grouped_down(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z; GroupDesc d=desc[c];
    if(s>=d.rows) return;
    size_t rb=row_bytes(d.df,I),row=(size_t)o*rb; const float *xs=x+(size_t)(d.offset+s)*I;
    float sum=0; for(int i=threadIdx.x;i<I;i+=blockDim.x) sum+=xs[i]*weight_at(d.d,d.df,row,i);
    __shared__ float p[256]; p[threadIdx.x]=sum; __syncthreads();
    for(int n=128;n;n>>=1){ if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n]; __syncthreads(); }
    if(!threadIdx.x) y[(size_t)(d.offset+s)*D+o]=p[0]*(d.df?d.ds[o]:1.f);
}

/* Native fmt=6 expert groups.  One block owns one (expert,row,output) and
 * expands each 32-weight E8 sub-block once for both gate/up projections. */
__global__ static void grouped_hidden_e8_dual(float *gate,const float *x,
                                               const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    size_t rb=row_bytes(6,D);
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*rb;
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*rb;
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    int nsub=(D+COLI_E8_SUB-1)/COLI_E8_SUB;
    for(int sb=threadIdx.x;sb<nsub;sb+=blockDim.x){
        int ib=sb%(COLI_E8_QK/COLI_E8_SUB);
        const uint8_t *gb=gr+(size_t)(sb/(COLI_E8_QK/COLI_E8_SUB))*COLI_E8_BBYTES;
        const uint8_t *ub=ur+(size_t)(sb/(COLI_E8_QK/COLI_E8_SUB))*COLI_E8_BBYTES;
        float gw[COLI_E8_SUB],uw[COLI_E8_SUB];
        e8_expand_sub_dev(gb,ib,e8_fp16(gb+96),gw);
        e8_expand_sub_dev(ub,ib,e8_fp16(ub+96),uw);
        int off=sb*COLI_E8_SUB,n=D-off<COLI_E8_SUB?D-off:COLI_E8_SUB;
        for(int k=0;k<n;k++){float v=xs[off+k];ga+=v*gw[k];ua+=v*uw[k];}
    }
    __shared__ float gp[256],up[256];gp[threadIdx.x]=ga;up[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];up[threadIdx.x]+=up[threadIdx.x+n];}__syncthreads();}
    if(!threadIdx.x){float g=gp[0];gate[(size_t)(d.offset+s)*I+o]=(g/(1.f+expf(-g)))*up[0];}
}

__global__ static void grouped_down_e8(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    size_t rb=row_bytes(6,I);const uint8_t *wr=(const uint8_t*)d.d+(size_t)o*rb;
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    int nsub=(I+COLI_E8_SUB-1)/COLI_E8_SUB;
    for(int sb=threadIdx.x;sb<nsub;sb+=blockDim.x){
        int ib=sb%(COLI_E8_QK/COLI_E8_SUB);
        const uint8_t *blk=wr+(size_t)(sb/(COLI_E8_QK/COLI_E8_SUB))*COLI_E8_BBYTES;
        float w[COLI_E8_SUB];e8_expand_sub_dev(blk,ib,e8_fp16(blk+96),w);
        int off=sb*COLI_E8_SUB,n=I-off<COLI_E8_SUB?I-off:COLI_E8_SUB;
        for(int k=0;k<n;k++)sum+=xs[off+k]*w[k];
    }
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0];
}

__device__ static void unpack_s4(uint8_t v,float *lo,float *hi){
    int a=v&15,b=v>>4; *lo=(float)(a&8?a-16:a); *hi=(float)(b&8?b-16:b);
}

/* Exact low-row W4A32 path. It consumes each packed weight byte once instead
 * of routing both nibbles through weight_at(), preserving FP32 activations. */
__global__ static void grouped_hidden_w4(float *y,const float *x,const GroupDesc *desc,
                                         int I,int D,int which){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *w=(const uint8_t*)(which?d.u:d.g);const float *sc=which?d.us:d.gs;
    const uint8_t *row=w+(size_t)o*((D+1)/2);const float *xs=x+(size_t)(d.offset+s)*D;
    float sum=0;for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<D)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*I+o]=p[0]*sc[o];
}

__global__ static void grouped_hidden_w4_dual(float *gate,float *up,const float *x,
                                               const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float g0,g1,u0,u1;unpack_s4(gr[b],&g0,&g1);unpack_s4(ur[b],&u0,&u1);
        int i=b*2;ga+=xs[i]*g0;ua+=xs[i]*u0;if(i+1<D){ga+=xs[i+1]*g1;ua+=xs[i+1]*u1;}}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    /* Fused epilogue: silu(gate)*up lands here instead of a third kernel —
     * the exact silu_mul expression on the exact same inputs, so bit-identical,
     * and the up[] round-trip through global memory disappears. up stays a
     * param so the launch sites keep their signature. */
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;
        float g=gp[0]*d.gs[o],u=upv[0]*d.us[o];
        gate[z]=(g/(1.0f+expf(-g)))*u;(void)up;}
}

__global__ static void grouped_down_w4(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int b=threadIdx.x;b<(I+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;sum+=xs[i]*a;if(i+1<I)sum+=xs[i+1]*z;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0]*d.ds[o];
}

/* fmt=4 grouped-int4 variants (#334): identical structure to the w4 kernels,
 * but the scale varies along the input dimension — sc[o*ng + i/gs], applied
 * per element inside the accumulation (gs is even, so a packed byte never
 * straddles a group). gs<=0 degrades to per-row (ng=1), so mixed fmt2/fmt4
 * groups run correctly through this one kernel family. */
__global__ static void grouped_hidden_g4_dual(float *gate,float *up,const float *x,
                                              const GroupDesc *desc,int I,int D){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *gr=(const uint8_t*)d.g+(size_t)o*((D+1)/2);
    const uint8_t *ur=(const uint8_t*)d.u+(size_t)o*((D+1)/2);
    int ggs=d.ggs>0?d.ggs:D, ugs=d.ugs>0?d.ugs:D;
    const float *gsc=d.gs+(size_t)o*(size_t)((D+ggs-1)/ggs);
    const float *usc=d.us+(size_t)o*(size_t)((D+ugs-1)/ugs);
    const float *xs=x+(size_t)(d.offset+s)*D;float ga=0,ua=0;
    for(int b=threadIdx.x;b<(D+1)/2;b+=blockDim.x){float g0,g1,u0,u1;unpack_s4(gr[b],&g0,&g1);unpack_s4(ur[b],&u0,&u1);
        int i=b*2;float gv=gsc[i/ggs],uv=usc[i/ugs];
        ga+=xs[i]*g0*gv;ua+=xs[i]*u0*uv;
        if(i+1<D){ga+=xs[i+1]*g1*gv;ua+=xs[i+1]*u1*uv;}}
    __shared__ float gp[256],upv[256];gp[threadIdx.x]=ga;upv[threadIdx.x]=ua;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n){gp[threadIdx.x]+=gp[threadIdx.x+n];upv[threadIdx.x]+=upv[threadIdx.x+n];}__syncthreads();}
    /* same epilogue fusion as the w4 dual above (per-group scales already
     * applied inside the accumulation, so silu runs on the raw sums) */
    if(!threadIdx.x){size_t z=(size_t)(d.offset+s)*I+o;
        float g=gp[0],u=upv[0];
        gate[z]=(g/(1.0f+expf(-g)))*u;(void)up;}
}
__global__ static void grouped_down_g4(float *y,const float *x,const GroupDesc *desc,int D,int I){
    int o=blockIdx.x,s=blockIdx.y,c=blockIdx.z;GroupDesc d=desc[c];if(s>=d.rows)return;
    const uint8_t *row=(const uint8_t*)d.d+(size_t)o*((I+1)/2);
    int dgs=d.dgs>0?d.dgs:I;
    const float *dsc=d.ds+(size_t)o*(size_t)((I+dgs-1)/dgs);
    const float *xs=x+(size_t)(d.offset+s)*I;float sum=0;
    for(int b=threadIdx.x;b<(I+1)/2;b+=blockDim.x){float a,z;unpack_s4(row[b],&a,&z);
        int i=b*2;float sv=dsc[i/dgs];
        sum+=xs[i]*a*sv;if(i+1<I)sum+=xs[i+1]*z*sv;}
    __shared__ float p[256];p[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n;n>>=1){if(threadIdx.x<n)p[threadIdx.x]+=p[threadIdx.x+n];__syncthreads();}
    if(!threadIdx.x)y[(size_t)(d.offset+s)*D+o]=p[0];
}

__global__ static void attention_absorb_kernel(float *ctx,const float *q,const float *latent,
                                                const float *rope,const void *weights,const float *wscale,
                                                int fmt,int H,int Q,int R,int V,int K,int T,float scale,
                                                int gs,int ng){
    int h=blockIdx.x,tid=threadIdx.x,rbase=h*(Q+V);extern __shared__ float sm[];
    float *qa=sm,*cl=qa+K,*scores=cl+K;
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=q[(size_t)h*(Q+R)+d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<T;t+=blockDim.x){float a=0;const float *lt=latent+(size_t)t*K,*rt=rope+(size_t)t*R;
        for(int k=0;k<K;k++)a+=qa[k]*lt[k];for(int d=0;d<R;d++)a+=q[(size_t)h*(Q+R)+Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    if(!tid){float mx=scores[0];for(int t=1;t<T;t++)mx=fmaxf(mx,scores[t]);float z=0;
        for(int t=0;t<T;t++){scores[t]=expf(scores[t]-mx);z+=scores[t];}for(int t=0;t<T;t++)scores[t]/=z;}
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<T;t++)a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*absorb_scale(wscale,fmt,gs,ng,row,k);ctx[(size_t)h*V+v]=a;}
}

__global__ static void attention_absorb_batch_kernel(float *ctx,const float *q,
        const float *latent,const float *rope,const void *weights,const float *wscale,
        int fmt,int S,int H,int Q,int R,int V,int K,int T,float scale,
        int gs,int ng){
    int s=blockIdx.y,h=blockIdx.x,tid=threadIdx.x,nt=T-S+s+1,rbase=h*(Q+V);
    if(s>=S||nt<1)return;
    extern __shared__ float sm[];float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+T;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<nt;t+=blockDim.x){float a=0;const float *lt=latent+(size_t)t*K;
        const float *rt=rope+(size_t)t*R;for(int k=0;k<K;k++)a+=qa[k]*lt[k];
        for(int d=0;d<R;d++)a+=qs[Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int t=tid;t<nt;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];local=0;for(int t=tid;t<nt;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];for(int t=tid;t<nt;t+=blockDim.x)scores[t]*=inv;
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<nt;t++)
        a+=scores[t]*latent[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*absorb_scale(wscale,fmt,gs,ng,row,k);
        ctx[((size_t)s*H+h)*V+v]=a;}
}

/* Long-context batch-1 MLA needs one score float per cached token.  CUDA keeps
 * the historical per-block default unless a kernel explicitly opts into the
 * device's larger limit.  Reject before launch when the physical limit cannot
 * satisfy the request; a failed opt-in must never become a launch experiment. */
static int configure_attention_absorb_shared(DeviceContext *ctx,size_t bytes){
    if(!ctx||bytes>ctx->shared_memory_optin_bytes)return 0;
    if(bytes<=ctx->shared_memory_default_bytes)return 1;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    /* HIP is intentionally unchanged in H014-033c.  Its retained opt-in value
     * equals the default, so the branch above is the only accepted case. */
    return 0;
#else
    return cuda_ok(cudaFuncSetAttribute(attention_absorb_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,(int)bytes),
        "Kimi MLA opt-in shared memory");
#endif
}

/* Kimi gated-MLA uses a sigmoid output gate, not the SiLU gate used by its
 * expert MLPs.  Keep this as a separate primitive so an MLA stage can remain
 * device-resident without changing the semantics of pipe_silu_mul. */
__global__ static void kimi_mla_sigmoid_gate(float *ctx,const float *gate,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)ctx[i]*=1.f/(1.f+expf(-gate[i]));
}

/* Independent device-resident KV sequence per row. lengths selects the valid
 * prefix; latent/rope point at paged caches updated by the host wrapper. */
__global__ static void attention_absorb_ragged_kernel(float *ctx,const float *q,
        const float *const *latent,const float *const *rope,const int *lengths,
        const void *weights,const float *wscale,int fmt,int S,int H,int Q,int R,
        int V,int K,int T,float scale,int gs,int ng){
    int s=blockIdx.y,h=blockIdx.x,tid=threadIdx.x,nt=lengths[s],rbase=h*(Q+V);
    if(s>=S||nt<1||nt>T)return;
    extern __shared__ float sm[];float *qa=sm,*cl=qa+K,*scores=cl+K,*red=scores+T;
    const float *qs=q+((size_t)s*H+h)*(Q+R);
    const float *ls=latent[s],*rs=rope[s];
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int d=0;d<Q;d++)
        a+=qs[d]*weight_at(weights,fmt,(size_t)(rbase+d)*row_bytes(fmt,K),k)*
          absorb_scale(wscale,fmt,gs,ng,rbase+d,k);qa[k]=a;}
    __syncthreads();
    for(int t=tid;t<nt;t+=blockDim.x){float a=0;const float *lt=ls+(size_t)t*K;
        const float *rt=rs+(size_t)t*R;for(int k=0;k<K;k++)a+=qa[k]*lt[k];
        for(int d=0;d<R;d++)a+=qs[Q+d]*rt[d];scores[t]=a*scale;}
    __syncthreads();
    float local=-3.402823466e+38F;for(int t=tid;t<nt;t+=blockDim.x)local=fmaxf(local,scores[t]);
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]=fmaxf(red[tid],red[tid+n]);__syncthreads();}
    float mx=red[0];local=0;for(int t=tid;t<nt;t+=blockDim.x){float e=expf(scores[t]-mx);scores[t]=e;local+=e;}
    red[tid]=local;__syncthreads();
    for(int n=blockDim.x>>1;n;n>>=1){if(tid<n)red[tid]+=red[tid+n];__syncthreads();}
    float inv=1.f/red[0];for(int t=tid;t<nt;t+=blockDim.x)scores[t]*=inv;
    __syncthreads();
    for(int k=tid;k<K;k+=blockDim.x){float a=0;for(int t=0;t<nt;t++)a+=scores[t]*ls[(size_t)t*K+k];cl[k]=a;}
    __syncthreads();
    for(int v=tid;v<V;v+=blockDim.x){int row=rbase+Q+v;float a=0;size_t rb=row_bytes(fmt,K);
        for(int k=0;k<K;k++)a+=cl[k]*weight_at(weights,fmt,(size_t)row*rb,k)*
            absorb_scale(wscale,fmt,gs,ng,row,k);
        ctx[((size_t)s*H+h)*V+v]=a;}
}

__global__ static void ragged_kv_append(float *const *latent,float *const *rope,
        const float *packed,const int *old_len,const int *add,const int *offset,int K,int R){
    int s=blockIdx.x,n=add[s],base=offset[s];
    for(int i=threadIdx.x;i<n*(K+R);i+=blockDim.x){
        if(i<n*K)latent[s][(size_t)old_len[s]*K+i]=packed[base+i];
        else rope[s][(size_t)old_len[s]*R+i-n*K]=packed[base+i];
    }
}

static int reserve(float **ptr, size_t *cap, size_t bytes) {
    if (*cap >= bytes) return 1;
    if (*ptr) cudaFree(*ptr);
    *ptr = nullptr;
    *cap = 0;
    if (!cuda_ok(cudaMalloc(ptr, bytes), "scratch allocation")) return 0;
    *cap = bytes;
    return 1;
}

static int reserve_bytes(void **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes) return 1; if(*ptr) cudaFree(*ptr); *ptr=nullptr; *cap=0;
    if(!cuda_ok(cudaMalloc(ptr,bytes),"descriptor allocation")) return 0; *cap=bytes; return 1;
}

static int reserve_pinned(float **ptr,size_t *cap,size_t bytes){
    if(*cap>=bytes)return 1;if(*ptr)cudaFreeHost(*ptr);*ptr=nullptr;*cap=0;
    if(!cuda_ok(cudaMallocHost(ptr,bytes),"pinned staging allocation"))return 0;*cap=bytes;return 1;
}

#ifdef COLI_ANS
static void *ans_arena_alloc(DeviceContext *ctx,size_t bytes){
    bytes=(bytes+255)&~size_t(255);
    if(!ctx->ans_chunks)ctx->ans_chunks=new std::vector<AnsArenaChunk>;
    if(ctx->ans_chunks->empty()||ctx->ans_chunks->back().cap-ctx->ans_chunks->back().used<bytes){
        size_t cap=256ull<<20;if(cap<bytes)cap=bytes;
        uint8_t *p=nullptr;if(!cuda_ok(cudaMalloc(&p,cap),"ANS arena chunk"))return nullptr;
        ctx->ans_chunks->push_back({p,0,cap});
    }
    AnsArenaChunk &c=ctx->ans_chunks->back();void *p=c.p+c.used;c.used+=bytes;return p;
}
static int ans_host_reserve(DeviceContext *ctx,size_t bytes){
    if(ctx->ans_copy_pending){
        if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"ANS sidecar upload synchronize"))return 0;
        ctx->ans_copy_pending=0;
    }
    if(ctx->ans_host_cap>=bytes)return 1;
    if(ctx->ans_host)cudaFreeHost(ctx->ans_host);
    ctx->ans_host=nullptr;ctx->ans_host_cap=0;
    if(!cuda_ok(cudaMallocHost(&ctx->ans_host,bytes),"ANS pinned staging allocation"))return 0;
    ctx->ans_host_cap=bytes;return 1;
}
static int prepare_group_weights(DeviceContext *ctx,
        ColiCudaTensor *const *gates,ColiCudaTensor *const *ups,
        ColiCudaTensor *const *downs,int count,GroupDesc *host){
    int n=0; size_t total=0;
    for(int c=0;c<count;c++){
        ColiCudaTensor *q[3]={gates[c],ups[c],downs[c]};
        for(int k=0;k<3;k++) if(q[k]->compressed){n++;total+=q[k]->weight_bytes;}
    }
    if(!n) return 1;
    if(!g_ans_profile_printed&&std::getenv("COLI_ANS_PROFILE")){
        g_ans_profile_printed=1;
        std::fprintf(stderr,
            "[ANS] load profile: %llu records | header %.2fs | read %.2fs | "
            "staging/alloc %.2fs | enqueue %.2fs\n",
            (unsigned long long)g_ans_load_records,g_ans_header_s,g_ans_read_s,
            g_ans_stage_s,g_ans_enqueue_s);
    }
    if(!ctx->ans_scratch||!reserve_bytes(&ctx->ans_raw,&ctx->ans_raw_cap,total)) return 0;
    std::vector<const void*> in; in.reserve(n);
    std::vector<void*> out; out.reserve(n);
    std::vector<uint32_t> cap; cap.reserve(n);
    size_t off=0;
    for(int c=0;c<count;c++){
        ColiCudaTensor *q[3]={gates[c],ups[c],downs[c]};
        const void **dst[3]={&host[c].g,&host[c].u,&host[c].d};
        for(int k=0;k<3;k++) if(q[k]->compressed){
            void *raw=(uint8_t*)ctx->ans_raw+off;
            in.push_back(q[k]->weights);out.push_back(raw);cap.push_back((uint32_t)q[k]->weight_bytes);
            *dst[k]=raw;off+=q[k]->weight_bytes;
        }
    }
    dietgpu::ANSCodecConfig config(11,false);
    dietgpu::ansDecodeBatchPointer(*ctx->ans_scratch,config,(uint32_t)n,in.data(),out.data(),
                                   cap.data(),nullptr,nullptr,ctx->stream);
    return cuda_ok(cudaGetLastError(),"ANS expert decode launch");
}
#else
static int prepare_group_weights(DeviceContext *,ColiCudaTensor *const *,
        ColiCudaTensor *const *,ColiCudaTensor *const *,int,GroupDesc *){return 1;}
#endif

/* Publish quant.h's E8 codebook to every configured device. __constant__ memory
 * is per-device, so this walks the contexts; the engine calls it once after init
 * rather than the backend carrying a second copy of the table that could drift
 * from the CPU decoder's (#452). Safe to call before any fmt=6 upload only. */
extern "C" int coli_cuda_e8_set_grid(const void *grid) {
    if (!grid || g_nctx < 1) return 0;
    for (int i = 0; i < g_nctx; i++) {
        if (!select_ctx(&g_ctx[i])) return 0;
        if (!cuda_ok(cudaMemcpyToSymbol(c_e8_grid, grid, sizeof(c_e8_grid)), "E8 codebook upload"))
            return 0;
    }
    return 1;
}

extern "C" int coli_cuda_init(const int *devices, int count) {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    /* #509: the ROCm runtime (comgr, MIOpen, roctracer) reads $TEMP as a temp-dir
     * path. A stray numeric TEMP (the engine's legacy sampling alias) makes comgr's
     * lazy init fail inside the first stream create -- SIGSEGV in the error-unwind
     * on gfx1100, clean hipErrorOutOfMemory on gfx1030. The engine has already
     * parsed g_temp by the time we get here, so a TEMP that is not a real directory
     * is safe to drop before the first ROCm call; a genuine temp-dir is preserved. */
    {
        const char *t = std::getenv("TEMP");
        struct stat st;
        if (t && *t && (stat(t, &st) != 0 || !S_ISDIR(st.st_mode))) unsetenv("TEMP");
    }
#endif
    int available = 0;
    if (!devices || count < 1 || count > COLI_CUDA_MAX_DEVICES) return 0;
    if (!cuda_ok(cudaGetDeviceCount(&available), "device discovery")) return 0;
    g_nctx = 0;
    for (int i = 0; i < count; i++) {
        int device = devices[i];
        if (device < 0 || device >= available) {
            std::fprintf(stderr, "[CUDA] invalid device %d (available: 0..%d)\n", device, available - 1);
            g_nctx = 0;
            return 0;
        }
        if (find_ctx(device)) {
            std::fprintf(stderr, "[CUDA] duplicate device %d\n", device);
            g_nctx = 0;
            return 0;
        }
        DeviceContext *ctx = &g_ctx[g_nctx];
        *ctx = {};
        ctx->device = device;
        if (!select_ctx(ctx)) { g_nctx = 0; return 0; }
        cudaDeviceProp prop{};
        if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) { g_nctx = 0; return 0; }
        ctx->compute_major=prop.major;ctx->compute_minor=prop.minor;
        ctx->shared_memory_default_bytes=(size_t)prop.sharedMemPerBlock;
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
        ctx->shared_memory_optin_bytes=ctx->shared_memory_default_bytes;
#else
        ctx->shared_memory_optin_bytes=(size_t)prop.sharedMemPerBlockOptin;
        if(ctx->shared_memory_optin_bytes<ctx->shared_memory_default_bytes)
            ctx->shared_memory_optin_bytes=ctx->shared_memory_default_bytes;
#endif
        int device_cc=prop.major*10+prop.minor;
        if(!coli_cuda_binary_accepts_compute_capability(prop.major,prop.minor)){
            std::fprintf(stderr,"[CUDA] device %d sm_%d rejected: binary minimum is sm_%d\n",
                         device,device_cc,COLI_CUDA_MIN_CC);
            g_nctx=0;return 0;
        }
        if(!cuda_ok(cudaStreamCreateWithFlags(&ctx->stream,cudaStreamNonBlocking),"stream creation")){
            g_nctx=0;return 0;
        }
#ifdef COLI_ANS
        if(std::getenv("CUDA_RAW_EXPERTS")){
            ctx->ans_scratch = new dietgpu::StackDeviceMemory(device, 1ull << 30);
            ctx->ans_chunks = new std::vector<AnsArenaChunk>;
        }
#endif
        g_nctx++;
        std::fprintf(stderr, "[CUDA] device %d: %s, %.1f GB VRAM, sm_%d%d\n",
                     device, prop.name, prop.totalGlobalMem / 1e9, prop.major, prop.minor);
    }
    return 1;
}

extern "C" void coli_cuda_shutdown(void) {
    for (int i = 0; i < g_nctx; i++) {
        DeviceContext *ctx = &g_ctx[i];
        if (!select_ctx(ctx)) continue;
        if (ctx->x) cudaFree(ctx->x);
        if (ctx->y) cudaFree(ctx->y);
        if (ctx->gate) cudaFree(ctx->gate);
        if (ctx->up) cudaFree(ctx->up);
        if (ctx->qx) cudaFree(ctx->qx);
        if (ctx->qscale) cudaFree(ctx->qscale);
        if(ctx->aq)cudaFree(ctx->aq);if(ctx->al)cudaFree(ctx->al);if(ctx->ar)cudaFree(ctx->ar);if(ctx->ac)cudaFree(ctx->ac);
        for(int b=0;b<27;b++) if(ctx->pipe_buf[b]) cudaFree(ctx->pipe_buf[b]);
        if (ctx->host_x) cudaFreeHost(ctx->host_x);
        if (ctx->host_y) cudaFreeHost(ctx->host_y);
        if (ctx->host_kv) cudaFreeHost(ctx->host_kv);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
        if(ctx->profile_start)cudaEventDestroy(ctx->profile_start);
        if(ctx->profile_stop)cudaEventDestroy(ctx->profile_stop);
        if (ctx->group_desc) cudaFree(ctx->group_desc);
#ifdef COLI_ANS
        if(ctx->ans_copy_pending)cudaStreamSynchronize(ctx->stream);
        if(ctx->ans_host)cudaFreeHost(ctx->ans_host);
        if (ctx->ans_raw) cudaFree(ctx->ans_raw);
        if(ctx->ans_chunks){for(auto &c:*ctx->ans_chunks)cudaFree(c.p);delete ctx->ans_chunks;}
        delete ctx->ans_scratch;
        ctx->ans_scratch=nullptr;ctx->ans_chunks=nullptr;ctx->ans_raw=nullptr;ctx->ans_raw_cap=0;
        ctx->ans_host=nullptr;ctx->ans_host_cap=0;ctx->ans_copy_pending=0;
#endif
        ctx->x = ctx->y = ctx->gate = ctx->up = nullptr;
        ctx->qx=nullptr; ctx->qscale=nullptr;
        ctx->aq=ctx->al=ctx->ar=ctx->ac=nullptr;
        ctx->host_x=ctx->host_y=ctx->host_kv=nullptr;ctx->stream=nullptr;
        ctx->profile_start=ctx->profile_stop=nullptr;ctx->profile_started=0;
        ctx->x_cap = ctx->y_cap = ctx->gate_cap = ctx->up_cap = 0;
        ctx->qx_cap=ctx->qscale_cap=0;
        ctx->aq_cap=ctx->al_cap=ctx->ar_cap=ctx->ac_cap=0;
        ctx->host_x_cap=ctx->host_y_cap=ctx->host_kv_cap=0;
        ctx->group_desc=nullptr; ctx->group_desc_cap=0;
    }
    g_nctx = 0;
#ifdef COLI_ANS
    if(g_ans_sidecar){std::fclose(g_ans_sidecar);g_ans_sidecar=nullptr;}
#if defined(__linux__)
    if(g_ans_direct_fd>=0){close(g_ans_direct_fd);g_ans_direct_fd=-1;g_ans_direct_off=0;}
#endif
#endif
}

extern "C" int coli_cuda_device_count(void) { return g_nctx; }

extern "C" int coli_cuda_device_at(int index) {
    return index >= 0 && index < g_nctx ? g_ctx[index].device : -1;
}

extern "C" int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes) {
    DeviceContext *ctx = find_ctx(device);
    if (!free_bytes || !total_bytes || !select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemGetInfo(free_bytes, total_bytes), "memory info");
}

/* #653: 1 when the device shares physical memory with the host (Grace-Blackwell /
 * GB10, Jetson, integrated GPUs). On these the expert tier and the RAM cache draw
 * from the same pool, so the RAM budget must account for the tier; on a discrete GPU
 * VRAM is a separate pool and this returns 0. */
extern "C" int coli_cuda_device_integrated(int device) {
    cudaDeviceProp prop{};
    if (!cuda_ok(cudaGetDeviceProperties(&prop, device), "device properties")) return 0;
    return prop.integrated ? 1 : 0;
}

extern "C" void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes) {
    size_t count = 0, bytes = 0;
    for (int i = 0; i < g_nctx; i++) if (device < 0 || g_ctx[i].device == device) {
        count += g_ctx[i].tensor_count;
        bytes += g_ctx[i].tensor_bytes;
    }
    if (tensor_count) *tensor_count = count;
    if (tensor_bytes) *tensor_bytes = bytes;
}

extern "C" void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                                        double *h2d_ms, double *kernel_ms, double *d2h_ms) {
    if(calls) *calls=g_group_calls; if(experts) *experts=g_group_experts; if(rows) *rows=g_group_rows;
    if(h2d_ms) *h2d_ms=g_group_h2d_ms; if(kernel_ms) *kernel_ms=g_group_kernel_ms;
    if(d2h_ms) *d2h_ms=g_group_d2h_ms;
}

extern "C" int coli_cuda_device_shared_memory_limits(int device,size_t *default_bytes,
        size_t *optin_bytes){
    DeviceContext *ctx=find_ctx(device);
    if(!ctx||!default_bytes||!optin_bytes)return 0;
    *default_bytes=ctx->shared_memory_default_bytes;
    *optin_bytes=ctx->shared_memory_optin_bytes;
    return 1;
}

extern "C" int coli_cuda_profile_begin(int device) {
    DeviceContext *ctx=find_ctx(device);
    if(!select_ctx(ctx)||ctx->profile_started)return 0;
    if(!ctx->profile_start&&
       !cuda_ok(cudaEventCreate(&ctx->profile_start),"profile start event"))return 0;
    if(!ctx->profile_stop&&
       !cuda_ok(cudaEventCreate(&ctx->profile_stop),"profile stop event"))return 0;
    if(!cuda_ok(cudaEventRecord(ctx->profile_start,0),"profile begin"))return 0;
    ctx->profile_started=1;return 1;
}

extern "C" int coli_cuda_profile_end(int device, double *elapsed_ms) {
    DeviceContext *ctx=find_ctx(device);float milliseconds=0.f;
    if(!elapsed_ms||!select_ctx(ctx)||!ctx->profile_started)return 0;
    int ok=cuda_ok(cudaEventRecord(ctx->profile_stop,0),"profile end")&&
           cuda_ok(cudaEventSynchronize(ctx->profile_stop),"profile synchronize")&&
           cuda_ok(cudaEventElapsedTime(&milliseconds,ctx->profile_start,ctx->profile_stop),
                   "profile elapsed time");
    ctx->profile_started=0;if(!ok)return 0;*elapsed_ms=milliseconds;return 1;
}

extern "C" int coli_cuda_binary_min_compute_capability(void) {
    return COLI_CUDA_MIN_CC;
}

extern "C" int coli_cuda_binary_has_forward_ptx(void) {
    return COLI_CUDA_HAS_FORWARD_PTX ? 1 : 0;
}

extern "C" int coli_cuda_binary_accepts_compute_capability(int major, int minor) {
    if(major<0||minor<0||minor>9) return 0;
    int cc=major*10+minor;
    if(COLI_CUDA_MIN_CC<=0) return 1;
    if(cc<COLI_CUDA_MIN_CC) return 0;
    if(cc==COLI_CUDA_MIN_CC) return 1;
    return COLI_CUDA_HAS_FORWARD_PTX ? 1 : 0;
}

extern "C" void coli_cuda_kimi_expert_stats(uint64_t *calls, uint64_t *rows,
                                               uint64_t *h2d_bytes, uint64_t *d2h_bytes,
                                               double *h2d_ms, double *kernel_ms,
                                               double *d2h_ms) {
    std::lock_guard<std::mutex> lock(g_kimi_stats_mu);
    if (calls) *calls = g_kimi_calls;
    if (rows) *rows = g_kimi_rows;
    if (h2d_bytes) *h2d_bytes = g_kimi_h2d_bytes;
    if (d2h_bytes) *d2h_bytes = g_kimi_d2h_bytes;
    if (h2d_ms) *h2d_ms = g_kimi_h2d_ms;
    if (kernel_ms) *kernel_ms = g_kimi_kernel_ms;
    if (d2h_ms) *d2h_ms = g_kimi_d2h_ms;
}

extern "C" void coli_cuda_kimi_expert_phase_stats(double *gate_ms, double *up_ms,
                                                     double *situ_ms, double *down_ms) {
    std::lock_guard<std::mutex> lock(g_kimi_stats_mu);
    if (gate_ms) *gate_ms = g_kimi_gate_ms;
    if (up_ms) *up_ms = g_kimi_up_ms;
    if (situ_ms) *situ_ms = g_kimi_situ_ms;
    if (down_ms) *down_ms = g_kimi_down_ms;
}

extern "C" void coli_cuda_kimi_expert_pair_stats(double *gate_up_pair_ms) {
    std::lock_guard<std::mutex> lock(g_kimi_stats_mu);
    if (gate_up_pair_ms) *gate_up_pair_ms = g_kimi_gate_up_pair_ms;
}

extern "C" void coli_cuda_kimi_expert_stats_reset(void) {
    std::lock_guard<std::mutex> lock(g_kimi_stats_mu);
    g_kimi_calls = g_kimi_rows = g_kimi_h2d_bytes = g_kimi_d2h_bytes = 0;
    g_kimi_h2d_ms = g_kimi_kernel_ms = g_kimi_d2h_ms = 0.0;
    g_kimi_gate_ms = g_kimi_up_ms = g_kimi_situ_ms = g_kimi_down_ms = 0.0;
    g_kimi_gate_up_pair_ms = 0.0;
}

extern "C" int coli_cuda_kimi_set_telemetry(int mode) {
    if (mode < 0 || mode > 2) return 0;
    g_kimi_telemetry_mode.store(mode, std::memory_order_relaxed);
    return 1;
}

extern "C" int coli_cuda_kimi_set_up_first(int up_first) {
    if (up_first != 0 && up_first != 1) return 0;
    g_kimi_up_first.store(up_first, std::memory_order_relaxed);
    return 1;
}

extern "C" int coli_cuda_kimi_set_fused_gate_up(int fused) {
    if (fused != 0 && fused != 1) return 0;
    g_kimi_fused_gate_up.store(fused, std::memory_order_relaxed);
    return 1;
}

extern "C" void coli_cuda_kimi_router_stats(uint64_t *calls, double *logits_ms,
                                              double *selection_ms, double *d2h_ms) {
    std::lock_guard<std::mutex> lock(g_kimi_stats_mu);
    if(calls)*calls=g_kimi_router_calls;
    if(logits_ms)*logits_ms=g_kimi_router_logits_ms;
    if(selection_ms)*selection_ms=g_kimi_router_selection_ms;
    if(d2h_ms)*d2h_ms=g_kimi_router_d2h_ms;
}

extern "C" void coli_cuda_kimi_router_stats_reset(void) {
    std::lock_guard<std::mutex> lock(g_kimi_stats_mu);
    g_kimi_router_calls=0;
    g_kimi_router_logits_ms=g_kimi_router_selection_ms=g_kimi_router_d2h_ms=0.0;
}

extern "C" void coli_cuda_kimi_dense_stats(uint64_t *calls, double *kernel_ms) {
    std::lock_guard<std::mutex> lock(g_kimi_stats_mu);
    if(calls)*calls=g_kimi_dense_calls;
    if(kernel_ms)*kernel_ms=g_kimi_dense_kernel_ms;
}

extern "C" void coli_cuda_kimi_dense_stats_reset(void) {
    std::lock_guard<std::mutex> lock(g_kimi_stats_mu);
    g_kimi_dense_calls=0;g_kimi_dense_kernel_ms=0.0;
}

/* group size for the NEXT upload on this thread (fmt=4): routed through a
 * thread_local so the widely-wired upload signature (and the Windows DLL ABI)
 * stays untouched. pin_load uploads in parallel, hence thread_local. */
static thread_local int g_upload_gs = 0;
extern "C" int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor,
                                         const void *weights, const float *scales,
                                         int fmt, int I, int O, int device, int gs);
extern "C" int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                                        const void *weights, const float *scales,
                                        int fmt, int I, int O, int device) {
    if (!tensor) return 0;
    if (*tensor) {
        /* Cached device copy: usable even when the caller's host pointers are
         * gone. CUDA_RELEASE_HOST slots null their host pointers after upload,
         * and with the old order (!weights checked first) every later matmul
         * on such a slot failed here — the GPU tier silently never computed
         * for host-released slab experts. */
        ColiCudaTensor *t = *tensor;
        int want_gs = ((fmt==4 || fmt==7) && g_upload_gs>0) ? g_upload_gs : 0;
        return t->fmt == fmt && t->I == I && t->O == O && t->device == device && t->gs == want_gs;
    }
    DeviceContext *ctx = find_ctx(device);
    if (!weights || I < 1 || O < 1 || !select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    /* fmt=6 keeps its scales inside each 98-byte block, so it is the one
     * quantized format that legitimately arrives with scales == NULL. */
    if (!rb || (fmt && fmt != 6 && fmt != 8 && !scales) ||
        (fmt == 7 && g_upload_gs != 32)) return 0;
    ColiCudaTensor *t = static_cast<ColiCudaTensor *>(std::calloc(1, sizeof(*t)));
    if (!t) return 0;
    t->fmt = fmt; t->I = I; t->O = O; t->device = device; t->weight_bytes = rb * (size_t)O;
    t->gs = ((fmt==4 || fmt==7) && g_upload_gs>0) ? g_upload_gs : 0;
    t->ng = t->gs ? (I + t->gs - 1) / t->gs : 1;
    t->scale_count = t->gs ? (size_t)O * (size_t)t->ng : (size_t)O;
    t->scale_bytes = (fmt == 6 || fmt == 8)
        ? 0 : t->scale_count * (fmt == 7 ? 1 : sizeof(float));
    if (!cuda_ok(cudaMalloc(&t->weights, t->weight_bytes), "tensor allocation") ||
        !cuda_ok(cudaMemcpy(t->weights, weights, t->weight_bytes, cudaMemcpyHostToDevice), "tensor upload")) {
        coli_cuda_tensor_free(t);
        return 0;
    }
    t->weights_owned=1;
    if(fmt==2||fmt==4){ /* same nibble layout: offset-binary -> signed in place */
        offset_to_signed_s4<<<(unsigned)((t->weight_bytes+255)/256),256>>>((uint8_t*)t->weights,t->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"int4 weight conversion")){coli_cuda_tensor_free(t);return 0;}}
    if (fmt && fmt != 6 && fmt != 8) {
        if (!cuda_ok(cudaMalloc(&t->scales, t->scale_bytes), "scale allocation") ||
            !cuda_ok(cudaMemcpy(t->scales, scales, t->scale_bytes, cudaMemcpyHostToDevice), "scale upload")) {
            coli_cuda_tensor_free(t);
            return 0;
        }
    }
    if (fmt == 6 || fmt == 8) t->scale_count = 0; /* no separate scale allocation */
    t->tracked = 1;
    ctx->tensor_count++;
    ctx->tensor_bytes += t->weight_bytes + t->scale_bytes;
    *tensor = t;
    return 1;
}
extern "C" int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor,
                                         const void *weights, const float *scales,
                                         int fmt, int I, int O, int device, int gs){
    g_upload_gs = gs>0 ? gs : 0;
    int r = coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device);
    g_upload_gs = 0;
    return r;
}

#ifdef COLI_ANS
struct AnsSidecarHeader {
    uint32_t magic,raw_bytes,archive_bytes,fmt,I,O;
};
static FILE *ans_sidecar(void){
    if(g_ans_sidecar) return g_ans_sidecar;
    const char *path=std::getenv("COLI_ANS_SIDECAR");
    if(!path||!*path) return nullptr;
    g_ans_sidecar_pack=std::getenv("COLI_ANS_PACK")&&std::atoi(std::getenv("COLI_ANS_PACK"));
    g_ans_sidecar=std::fopen(path,g_ans_sidecar_pack?"wb":"rb");
#if defined(__linux__)
    if(g_ans_sidecar&&!g_ans_sidecar_pack&&std::getenv("COLI_ANS_DIRECT")&&
       std::atoi(std::getenv("COLI_ANS_DIRECT"))){
        g_ans_direct_fd=open(path,O_RDONLY|O_DIRECT);
        if(g_ans_direct_fd<0)std::fprintf(stderr,"[ANS] O_DIRECT unavailable; using buffered sidecar\n");
    }
    if(g_ans_sidecar&&!g_ans_sidecar_pack&&g_ans_direct_fd<0)
        posix_fadvise(fileno(g_ans_sidecar),0,0,POSIX_FADV_SEQUENTIAL);
#endif
    return g_ans_sidecar;
}
extern "C" int coli_cuda_tensor_upload_compressed(ColiCudaTensor **tensor,
        const void *weights,const float *scales,int fmt,int I,int O,int device){
    if(fmt!=2 || !tensor || *tensor) return 0;  /* prototype: per-row int4 experts only */
    FILE *sidecar=ans_sidecar();
    if(sidecar&&!g_ans_sidecar_pack){
        AnsSidecarHeader h{};
        size_t expected=((size_t)I+1)/2*(size_t)O;
        double t0=ans_now_s();
#if defined(__linux__)
        size_t direct_delta=0,direct_bytes=0,direct_data=0;
        if(g_ans_direct_fd>=0){
            alignas(4096) uint8_t first[8192];
            off_t start=g_ans_direct_off&~off_t(4095);
            direct_delta=(size_t)(g_ans_direct_off-start);
            ssize_t got=pread(g_ans_direct_fd,first,sizeof(first),start);
            if(got<(ssize_t)(direct_delta+sizeof(h))){
                std::fprintf(stderr,"[ANS] direct header read failed at %lld: got %lld errno %d\n",
                    (long long)g_ans_direct_off,(long long)got,errno);
                return 0;
            }
            std::memcpy(&h,first+direct_delta,sizeof(h));
        }else
#endif
        if(std::fread(&h,sizeof(h),1,sidecar)!=1)return 0;
        if(h.magic!=0x31534e41u||
           h.fmt!=(uint32_t)fmt||h.I!=(uint32_t)I||h.O!=(uint32_t)O||
           expected>UINT32_MAX||h.raw_bytes!=(uint32_t)expected||
           !h.archive_bytes||h.archive_bytes>dietgpu::getMaxCompressedSize(h.raw_bytes)){
            std::fprintf(stderr,"[ANS] invalid or mismatched sidecar record\n");
            return 0;
        }
        g_ans_header_s+=ans_now_s()-t0;
        DeviceContext *ctx=find_ctx(device); if(!ctx||!select_ctx(ctx)) return 0;
        ColiCudaTensor *t=(ColiCudaTensor*)std::calloc(1,sizeof(*t)); if(!t)return 0;
        t->fmt=fmt;t->I=I;t->O=O;t->device=device;t->weight_bytes=h.raw_bytes;
        t->scale_count=(size_t)O;t->scale_bytes=(size_t)O*sizeof(float);
        t->archive_bytes=h.archive_bytes;t->compressed=1;
        t0=ans_now_s();
        t->weights=ans_arena_alloc(ctx,h.archive_bytes);
        size_t scale_bytes=(size_t)O*sizeof(float),scale_off;
#if defined(__linux__)
        if(g_ans_direct_fd>=0){
            size_t record_bytes=sizeof(h)+(size_t)h.archive_bytes;
            direct_data=direct_delta+sizeof(h);
            direct_bytes=(direct_delta+record_bytes+4095)&~size_t(4095);
            scale_off=(direct_bytes+255)&~size_t(255);
        }else
#endif
            scale_off=(h.archive_bytes+255)&~size_t(255);
        if(!t->weights||!ans_host_reserve(ctx,scale_off+scale_bytes)||
           !cuda_ok(cudaMalloc(&t->scales,scale_bytes),"ANS sidecar scales")){
            coli_cuda_tensor_free(t);return 0;
        }
        g_ans_stage_s+=ans_now_s()-t0;
        t0=ans_now_s();
#if defined(__linux__)
        if(g_ans_direct_fd>=0){
            off_t start=g_ans_direct_off&~off_t(4095);
            ssize_t got=pread(g_ans_direct_fd,ctx->ans_host,direct_bytes,start);
            if(got<(ssize_t)(direct_data+h.archive_bytes)){
                std::fprintf(stderr,
                    "[ANS] direct record read failed at %lld: need %zu got %lld errno %d\n",
                    (long long)g_ans_direct_off,direct_data+h.archive_bytes,
                    (long long)got,errno);
                coli_cuda_tensor_free(t);return 0;
            }
            g_ans_direct_off+=(off_t)sizeof(h)+(off_t)h.archive_bytes;
        }else
#endif
        if(std::fread(ctx->ans_host,h.archive_bytes,1,sidecar)!=1){
            std::fprintf(stderr,"[ANS] truncated sidecar record\n");
            coli_cuda_tensor_free(t);return 0;
        }
        g_ans_read_s+=ans_now_s()-t0;
        std::memcpy((uint8_t*)ctx->ans_host+scale_off,scales,scale_bytes);
        t0=ans_now_s();
        void *archive_src=
#if defined(__linux__)
            g_ans_direct_fd>=0?(uint8_t*)ctx->ans_host+direct_data:
#endif
            ctx->ans_host;
        if(!cuda_ok(cudaMemcpyAsync(t->weights,archive_src,h.archive_bytes,
                                   cudaMemcpyHostToDevice,ctx->stream),"ANS sidecar upload")||
           !cuda_ok(cudaMemcpyAsync(t->scales,(uint8_t*)ctx->ans_host+scale_off,scale_bytes,
                                   cudaMemcpyHostToDevice,ctx->stream),"ANS sidecar scale upload")){
            coli_cuda_tensor_free(t);return 0;
        }
        g_ans_enqueue_s+=ans_now_s()-t0;g_ans_load_records++;
        ctx->ans_copy_pending=1;
        t->tracked=1;ctx->tensor_count++;ctx->tensor_bytes+=h.archive_bytes+(size_t)O*sizeof(float);
        *tensor=t;return 1;
    }
    if(!sidecar||!g_ans_sidecar_pack) return 0;
    if(!coli_cuda_tensor_upload(tensor,weights,scales,fmt,I,O,device)) return 0;
    ColiCudaTensor *t=*tensor;
    DeviceContext *ctx=find_ctx(device);
    if(!ctx||!ctx->ans_scratch||!select_ctx(ctx)){ coli_cuda_tensor_free(t);*tensor=nullptr;return 0; }
    uint32_t raw=(uint32_t)t->weight_bytes;
    uint32_t bound=dietgpu::getMaxCompressedSize(raw), *dsize=nullptr;
    void *tmp=nullptr;
    if(!cuda_ok(cudaMalloc(&tmp,bound),"ANS archive allocation")||
       !cuda_ok(cudaMalloc(&dsize,sizeof(*dsize)),"ANS size allocation")){
        if(tmp)cudaFree(tmp);if(dsize)cudaFree(dsize);coli_cuda_tensor_free(t);*tensor=nullptr;return 0;
    }
    dietgpu::ANSCodecConfig config(11,false);
    /* tensor_upload converted offset-binary nibbles on the legacy stream.
     * ctx->stream is explicitly non-blocking, so it does not inherit the
     * legacy-stream dependency. Finish that one-time conversion before the
     * encoder reads the bytes. */
    if(!cuda_ok(cudaStreamSynchronize(0),"ANS source conversion synchronize")){
        cudaFree(tmp);cudaFree(dsize);coli_cuda_tensor_free(t);*tensor=nullptr;return 0;
    }
    dietgpu::ansEncodeBatchStride(*ctx->ans_scratch,config,1,t->weights,raw,raw,nullptr,
                                  tmp,bound,dsize,ctx->stream);
    uint32_t used=0;
    int ok=cuda_ok(cudaMemcpyAsync(&used,dsize,sizeof(used),cudaMemcpyDeviceToHost,ctx->stream),
                   "ANS size download")&&
           cuda_ok(cudaStreamSynchronize(ctx->stream),"ANS encode synchronize")&&used>0&&used<raw;
    cudaFree(dsize);
    if(!ok){cudaFree(tmp);coli_cuda_tensor_free(t);*tensor=nullptr;return 0;}
    if(g_ans_sidecar_pack){
        std::vector<uint8_t> archive(used);
        AnsSidecarHeader h{0x31534e41u,raw,used,(uint32_t)fmt,(uint32_t)I,(uint32_t)O};
        ok=cuda_ok(cudaMemcpy(archive.data(),tmp,used,cudaMemcpyDeviceToHost),"ANS sidecar download")&&
           std::fwrite(&h,sizeof(h),1,sidecar)==1&&
           std::fwrite(archive.data(),archive.size(),1,sidecar)==1;
        cudaFree(tmp);coli_cuda_tensor_free(t);*tensor=nullptr;
        if(!ok)return 0;
        t=(ColiCudaTensor*)std::calloc(1,sizeof(*t));if(!t)return 0;
        t->fmt=fmt;t->I=I;t->O=O;t->device=device;t->weight_bytes=raw;
        t->archive_bytes=used;t->compressed=1;*tensor=t;
        return 1;
    }
    return 0;
}
#endif

extern "C" int coli_cuda_tensor_update(ColiCudaTensor *tensor,
                                          const void *weights,
                                          const float *scales) {
    if (!tensor || !weights || (tensor->fmt && tensor->fmt != 6 && !scales)) return 0;
#ifdef COLI_ANS
    if(tensor->compressed) return 0;
#endif
    DeviceContext *ctx=find_ctx(tensor->device);
    if (!select_ctx(ctx)) return 0;
    if (!cuda_ok(cudaMemcpy(tensor->weights,weights,tensor->weight_bytes,
                            cudaMemcpyHostToDevice),"tensor refresh")) return 0;
    if(tensor->fmt==2||tensor->fmt==4){
        offset_to_signed_s4<<<(unsigned)((tensor->weight_bytes+255)/256),256>>>(
            (uint8_t*)tensor->weights,tensor->weight_bytes);
        if(!cuda_ok(cudaGetLastError(),"int4 weight refresh")) return 0;
    }
    /* fmt=6 has no scale buffer at all (scales live in-block, scale_count 0), and
     * the fallback below would otherwise copy O floats out of a NULL host pointer. */
    return !tensor->fmt || tensor->fmt==6 || cuda_ok(cudaMemcpy(tensor->scales,scales,
        tensor->scale_bytes,
        cudaMemcpyHostToDevice),"scale refresh");
}

/* Test hook: COLI_GPU_FAIL_AFTER=N makes every GPU COMPUTE entry point report
 * failure after N successful calls (N=0: every call fails), exercising the
 * engine's CPU fallbacks and host-rematerialization end-to-end without real
 * hardware faults. Uploads/queries are not gated. Unset: no effect. */
static long g_gpu_calls;
static int fault_injected(void) {
    const char *fa = std::getenv("COLI_GPU_FAIL_AFTER");
    return fa && g_gpu_calls++ >= std::atol(fa);
}

extern "C" int coli_cuda_matmul(ColiCudaTensor **tensor,
                                 float *y, const float *x,
                                 const void *weights, const float *scales,
                                 int fmt, int S, int I, int O, int device, int gs) {
    if (fault_injected()) return 0;
    /* fmt=4 carries [O, ceil(I/gs)] scales: without the group size the plain
     * upload truncates the buffer to O floats and quant_matmul divides by
     * gs==0. Callers must come through the gs>0 path (upload_g) or stay on
     * the CPU (#298, #334). */
    if (fmt == 4 && gs <= 0) return 0;
    if (S < 1) return 0;
    if (gs > 0) { if (!coli_cuda_tensor_upload_g(tensor, weights, scales, fmt, I, O, device, gs)) return 0; }
    else        { if (!coli_cuda_tensor_upload(tensor, weights, scales, fmt, I, O, device)) return 0; }
    ColiCudaTensor *t = *tensor;
    DeviceContext *ctx = find_ctx(t->device);
    if (!select_ctx(ctx)) return 0;
    size_t rb = row_bytes(fmt, I);
    size_t xb = (size_t)S * I * sizeof(float), yb = (size_t)S * O * sizeof(float);
    if (!reserve(&ctx->x, &ctx->x_cap, xb) || !reserve(&ctx->y, &ctx->y_cap, yb)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x, x, xb, cudaMemcpyHostToDevice), "input upload")) return 0;
    dim3 grid((unsigned)O, (unsigned)S);
    quant_matmul<<<grid, 256>>>(ctx->y, ctx->x, t->weights, t->scales, fmt, S, I, O, rb, t->gs, t->ng);
    if (!cuda_ok(cudaGetLastError(), "matmul launch") ||
        !cuda_ok(cudaMemcpy(y, ctx->y, yb, cudaMemcpyDeviceToHost), "output download")) return 0;
    return 1;
}

extern "C" int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                                      ColiCudaTensor *down, float *y,
                                      const float *x, int S) {
    if (fault_injected()) return 0;
    /* same reason as coli_cuda_matmul: fmt=4 without recorded group info would
     * misread the scales (and divide by gs==0 in the kernel). */
    if (gate && ((gate->fmt == 4 && gate->gs <= 0) ||
                 (up && up->fmt == 4 && up->gs <= 0) ||
                 (down && down->fmt == 4 && down->gs <= 0))) return 0;
    if (!gate || !up || !down || !x || !y || S < 1 ||
        gate->device != up->device || gate->device != down->device ||
        gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;
    DeviceContext *ctx = find_ctx(gate->device);
    if (!select_ctx(ctx)) return 0;
    int D = gate->I, I = gate->O;
    size_t xb=(size_t)S*D*sizeof(float), ib=(size_t)S*I*sizeof(float);
    size_t yb=(size_t)S*D*sizeof(float);
    if (!reserve(&ctx->x,&ctx->x_cap,xb) || !reserve(&ctx->y,&ctx->y_cap,yb) ||
        !reserve(&ctx->gate,&ctx->gate_cap,ib) || !reserve(&ctx->up,&ctx->up_cap,ib)) return 0;
    if (!cuda_ok(cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice),"expert input upload")) return 0;
    dim3 hidden_grid((unsigned)I,(unsigned)S), output_grid((unsigned)D,(unsigned)S);
    quant_matmul<<<hidden_grid,256>>>(ctx->gate,ctx->x,gate->weights,gate->scales,
        gate->fmt,S,D,I,row_bytes(gate->fmt,D),gate->gs,gate->ng);
    quant_matmul<<<hidden_grid,256>>>(ctx->up,ctx->x,up->weights,up->scales,
        up->fmt,S,D,I,row_bytes(up->fmt,D),up->gs,up->ng);
    size_t n=(size_t)S*I;
    silu_mul<<<(unsigned)((n+255)/256),256>>>(ctx->gate,ctx->up,n);
    /* fmt=6: the down projection stores W@Q, so its input needs Q^T applied. This
     * one is per-expert (the silu product is not shared), unlike the gate/up input
     * rotation, which the caller does once per layer -- same split as moe(). */
    if (down->fmt == 6 && !e8_rot_rows_dev(ctx->gate, S, I, 0)) return 0;
    quant_matmul<<<output_grid,256>>>(ctx->y,ctx->gate,down->weights,down->scales,
        down->fmt,S,I,D,row_bytes(down->fmt,I),down->gs,down->ng);
    if (!cuda_ok(cudaGetLastError(),"expert MLP launch") ||
        !cuda_ok(cudaMemcpy(y,ctx->y,yb,cudaMemcpyDeviceToHost),"expert output download")) return 0;
    return 1;
}

static int kimi_telemetry_mode(void) {
    return g_kimi_telemetry_mode.load(std::memory_order_relaxed);
}

extern "C" int coli_cuda_kimi_expert_max_certified_batch(void) {
    return COLI_CUDA_KIMI_EXPERT_MAX_CERTIFIED_BATCH;
}

static int kimi_expert_batch_is_certified(int batch) {
    if (batch == 1 || batch == 2) return 1;
#if COLI_CUDA_KIMI_EXPERT_MAX_CERTIFIED_BATCH >= 4
    if (batch == 4) return 1;
#endif
#if COLI_CUDA_KIMI_EXPERT_MAX_CERTIFIED_BATCH >= 8
    if (batch == 8) return 1;
#endif
#if COLI_CUDA_KIMI_EXPERT_MAX_CERTIFIED_BATCH >= 16
    if (batch == 16) return 1;
#endif
    return 0;
}

extern "C" int coli_cuda_kimi_expert_supports_batch(int batch) {
    return kimi_expert_batch_is_certified(batch);
}

extern "C" int coli_cuda_kimi_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                                           ColiCudaTensor *down, float *y,
                                           const float *x, int S,
                                           float beta, float linear_beta) {
    if (fault_injected()) return 0;
    if (!gate || !up || !down || !x || !y ||
        !kimi_expert_batch_is_certified(S) ||
        beta <= 0.f || linear_beta <= 0.f ||
        gate->fmt != 7 || up->fmt != 7 || down->fmt != 7 ||
        gate->gs != 32 || up->gs != 32 || down->gs != 32 ||
        gate->device != up->device || gate->device != down->device ||
        gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;
    DeviceContext *ctx = find_ctx(gate->device);
    if (!select_ctx(ctx)) return 0;
    int D = gate->I, I = gate->O;
    size_t xb=(size_t)S*D*sizeof(float), ib=(size_t)S*I*sizeof(float);
    if (!reserve(&ctx->x,&ctx->x_cap,xb) || !reserve(&ctx->y,&ctx->y_cap,xb) ||
        !reserve(&ctx->gate,&ctx->gate_cap,ib) || !reserve(&ctx->up,&ctx->up_cap,ib) ||
        !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb) ||
        !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb)) return 0;

    int telemetry = kimi_telemetry_mode();
    cudaEvent_t event[7] = {};
    if (telemetry == 2) {
        for (int i=0; i<7; ++i) {
            if (!cuda_ok(cudaEventCreate(&event[i]), "Kimi profile event")) {
                for (int j=0; j<i; ++j) cudaEventDestroy(event[j]);
                return 0;
            }
        }
        cudaEventRecord(event[0], ctx->stream);
    }

    std::memcpy(ctx->host_x, x, xb);
    int ok = cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                     "Kimi expert input upload");
    if (telemetry == 2 && ok) cudaEventRecord(event[1], ctx->stream);
    dim3 hidden_grid((unsigned)I,(unsigned)S), output_grid((unsigned)D,(unsigned)S);
    int fused_gate_up = g_kimi_fused_gate_up.load(std::memory_order_relaxed);
    if (ok) {
        int up_first = g_kimi_up_first.load(std::memory_order_relaxed);
        if (fused_gate_up) {
            if (S == 16) {
                mxfp4_matmul_pair_rows<16><<<(unsigned)I,256,0,ctx->stream>>>(
                    ctx->gate,ctx->up,ctx->x,
                    (const uint8_t*)gate->weights,(const uint8_t*)up->weights,
                    (const uint8_t*)gate->scales,(const uint8_t*)up->scales,
                    D,I,row_bytes(7,D),gate->ng);
            } else if (S == 8) {
                mxfp4_matmul_pair_rows<8><<<(unsigned)I,256,0,ctx->stream>>>(
                    ctx->gate,ctx->up,ctx->x,
                    (const uint8_t*)gate->weights,(const uint8_t*)up->weights,
                    (const uint8_t*)gate->scales,(const uint8_t*)up->scales,
                    D,I,row_bytes(7,D),gate->ng);
            } else if (S == 4) {
                mxfp4_matmul_pair_rows<4><<<(unsigned)I,256,0,ctx->stream>>>(
                    ctx->gate,ctx->up,ctx->x,
                    (const uint8_t*)gate->weights,(const uint8_t*)up->weights,
                    (const uint8_t*)gate->scales,(const uint8_t*)up->scales,
                    D,I,row_bytes(7,D),gate->ng);
            } else if (S == 2) {
                mxfp4_matmul_pair_rows<2><<<(unsigned)I,256,0,ctx->stream>>>(
                    ctx->gate,ctx->up,ctx->x,
                    (const uint8_t*)gate->weights,(const uint8_t*)up->weights,
                    (const uint8_t*)gate->scales,(const uint8_t*)up->scales,
                    D,I,row_bytes(7,D),gate->ng);
            } else {
                mxfp4_matmul_pair<<<hidden_grid,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,
                    (const uint8_t*)gate->weights,(const uint8_t*)up->weights,
                    (const uint8_t*)gate->scales,(const uint8_t*)up->scales,
                    S,D,I,row_bytes(7,D),gate->ng);
            }
            if (telemetry == 2) {
                cudaEventRecord(event[2], ctx->stream);
                cudaEventRecord(event[3], ctx->stream);
            }
        } else if (up_first) {
            if (S == 16)
                mxfp4_matmul_rows<16><<<(unsigned)I,256,0,ctx->stream>>>(ctx->up,ctx->x,
                    (const uint8_t*)up->weights,(const uint8_t*)up->scales,
                    D,I,row_bytes(7,D),up->ng);
            else if (S == 8)
                mxfp4_matmul_rows<8><<<(unsigned)I,256,0,ctx->stream>>>(ctx->up,ctx->x,
                    (const uint8_t*)up->weights,(const uint8_t*)up->scales,
                    D,I,row_bytes(7,D),up->ng);
            else if (S == 4)
                mxfp4_matmul_rows<4><<<(unsigned)I,256,0,ctx->stream>>>(ctx->up,ctx->x,
                    (const uint8_t*)up->weights,(const uint8_t*)up->scales,
                    D,I,row_bytes(7,D),up->ng);
            else if (S == 2)
                mxfp4_matmul_rows<2><<<(unsigned)I,256,0,ctx->stream>>>(ctx->up,ctx->x,
                    (const uint8_t*)up->weights,(const uint8_t*)up->scales,
                    D,I,row_bytes(7,D),up->ng);
            else
                quant_matmul<<<hidden_grid,256,0,ctx->stream>>>(ctx->up,ctx->x,up->weights,
                    up->scales,7,S,D,I,row_bytes(7,D),32,up->ng);
            if (telemetry == 2) cudaEventRecord(event[2], ctx->stream);
            if (S == 16)
                mxfp4_matmul_rows<16><<<(unsigned)I,256,0,ctx->stream>>>(ctx->gate,ctx->x,
                    (const uint8_t*)gate->weights,(const uint8_t*)gate->scales,
                    D,I,row_bytes(7,D),gate->ng);
            else if (S == 8)
                mxfp4_matmul_rows<8><<<(unsigned)I,256,0,ctx->stream>>>(ctx->gate,ctx->x,
                    (const uint8_t*)gate->weights,(const uint8_t*)gate->scales,
                    D,I,row_bytes(7,D),gate->ng);
            else if (S == 4)
                mxfp4_matmul_rows<4><<<(unsigned)I,256,0,ctx->stream>>>(ctx->gate,ctx->x,
                    (const uint8_t*)gate->weights,(const uint8_t*)gate->scales,
                    D,I,row_bytes(7,D),gate->ng);
            else if (S == 2)
                mxfp4_matmul_rows<2><<<(unsigned)I,256,0,ctx->stream>>>(ctx->gate,ctx->x,
                    (const uint8_t*)gate->weights,(const uint8_t*)gate->scales,
                    D,I,row_bytes(7,D),gate->ng);
            else
                quant_matmul<<<hidden_grid,256,0,ctx->stream>>>(ctx->gate,ctx->x,gate->weights,
                    gate->scales,7,S,D,I,row_bytes(7,D),32,gate->ng);
            if (telemetry == 2) cudaEventRecord(event[3], ctx->stream);
        } else {
            if (S == 16)
                mxfp4_matmul_rows<16><<<(unsigned)I,256,0,ctx->stream>>>(ctx->gate,ctx->x,
                    (const uint8_t*)gate->weights,(const uint8_t*)gate->scales,
                    D,I,row_bytes(7,D),gate->ng);
            else if (S == 8)
                mxfp4_matmul_rows<8><<<(unsigned)I,256,0,ctx->stream>>>(ctx->gate,ctx->x,
                    (const uint8_t*)gate->weights,(const uint8_t*)gate->scales,
                    D,I,row_bytes(7,D),gate->ng);
            else if (S == 4)
                mxfp4_matmul_rows<4><<<(unsigned)I,256,0,ctx->stream>>>(ctx->gate,ctx->x,
                    (const uint8_t*)gate->weights,(const uint8_t*)gate->scales,
                    D,I,row_bytes(7,D),gate->ng);
            else if (S == 2)
                mxfp4_matmul_rows<2><<<(unsigned)I,256,0,ctx->stream>>>(ctx->gate,ctx->x,
                    (const uint8_t*)gate->weights,(const uint8_t*)gate->scales,
                    D,I,row_bytes(7,D),gate->ng);
            else
                quant_matmul<<<hidden_grid,256,0,ctx->stream>>>(ctx->gate,ctx->x,gate->weights,
                    gate->scales,7,S,D,I,row_bytes(7,D),32,gate->ng);
            if (telemetry == 2) cudaEventRecord(event[2], ctx->stream);
            if (S == 16)
                mxfp4_matmul_rows<16><<<(unsigned)I,256,0,ctx->stream>>>(ctx->up,ctx->x,
                    (const uint8_t*)up->weights,(const uint8_t*)up->scales,
                    D,I,row_bytes(7,D),up->ng);
            else if (S == 8)
                mxfp4_matmul_rows<8><<<(unsigned)I,256,0,ctx->stream>>>(ctx->up,ctx->x,
                    (const uint8_t*)up->weights,(const uint8_t*)up->scales,
                    D,I,row_bytes(7,D),up->ng);
            else if (S == 4)
                mxfp4_matmul_rows<4><<<(unsigned)I,256,0,ctx->stream>>>(ctx->up,ctx->x,
                    (const uint8_t*)up->weights,(const uint8_t*)up->scales,
                    D,I,row_bytes(7,D),up->ng);
            else if (S == 2)
                mxfp4_matmul_rows<2><<<(unsigned)I,256,0,ctx->stream>>>(ctx->up,ctx->x,
                    (const uint8_t*)up->weights,(const uint8_t*)up->scales,
                    D,I,row_bytes(7,D),up->ng);
            else
                quant_matmul<<<hidden_grid,256,0,ctx->stream>>>(ctx->up,ctx->x,up->weights,
                    up->scales,7,S,D,I,row_bytes(7,D),32,up->ng);
            if (telemetry == 2) cudaEventRecord(event[3], ctx->stream);
        }
        size_t n=(size_t)S*I;
        kimi_situ_mul<<<(unsigned)((n+255)/256),256,0,ctx->stream>>>(
            ctx->gate,ctx->up,n,beta,linear_beta);
        if (telemetry == 2) cudaEventRecord(event[4], ctx->stream);
        if (S == 16)
            mxfp4_matmul_rows<16><<<(unsigned)D,256,0,ctx->stream>>>(ctx->y,ctx->gate,
                (const uint8_t*)down->weights,(const uint8_t*)down->scales,
                I,D,row_bytes(7,I),down->ng);
        else if (S == 8)
            mxfp4_matmul_rows<8><<<(unsigned)D,256,0,ctx->stream>>>(ctx->y,ctx->gate,
                (const uint8_t*)down->weights,(const uint8_t*)down->scales,
                I,D,row_bytes(7,I),down->ng);
        else if (S == 4)
            mxfp4_matmul_rows<4><<<(unsigned)D,256,0,ctx->stream>>>(ctx->y,ctx->gate,
                (const uint8_t*)down->weights,(const uint8_t*)down->scales,
                I,D,row_bytes(7,I),down->ng);
        else if (S == 2)
            mxfp4_matmul_rows<2><<<(unsigned)D,256,0,ctx->stream>>>(ctx->y,ctx->gate,
                (const uint8_t*)down->weights,(const uint8_t*)down->scales,
                I,D,row_bytes(7,I),down->ng);
        else
            quant_matmul<<<output_grid,256,0,ctx->stream>>>(ctx->y,ctx->gate,down->weights,
                down->scales,7,S,I,D,row_bytes(7,I),32,down->ng);
        if (telemetry == 2) cudaEventRecord(event[5], ctx->stream);
        ok = cuda_ok(cudaGetLastError(), "Kimi expert MLP launch");
    }
    if (ok) ok = cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                         "Kimi expert output download");
    if (telemetry == 2 && ok) cudaEventRecord(event[6], ctx->stream);
    if (ok) ok = cuda_ok(cudaStreamSynchronize(ctx->stream), "Kimi expert synchronize");
    if (ok) std::memcpy(y,ctx->host_y,xb);

    double h2d_ms=0.0,kernel_ms=0.0,d2h_ms=0.0;
    double gate_ms=0.0,up_ms=0.0,situ_ms=0.0,down_ms=0.0,gate_up_pair_ms=0.0;
    if (telemetry == 2 && ok) {
        float a=0.f,b=0.f,c=0.f,first=0.f,second=0.f,s=0.f,d=0.f;
        ok = cuda_ok(cudaEventElapsedTime(&a,event[0],event[1]),"Kimi H2D timing") &&
             cuda_ok(cudaEventElapsedTime(&first,event[1],event[2]),"Kimi first projection timing") &&
             cuda_ok(cudaEventElapsedTime(&second,event[2],event[3]),"Kimi second projection timing") &&
             cuda_ok(cudaEventElapsedTime(&s,event[3],event[4]),"Kimi SiTU timing") &&
             cuda_ok(cudaEventElapsedTime(&d,event[4],event[5]),"Kimi down timing") &&
             cuda_ok(cudaEventElapsedTime(&b,event[1],event[5]),"Kimi kernel timing") &&
             cuda_ok(cudaEventElapsedTime(&c,event[5],event[6]),"Kimi D2H timing");
        h2d_ms=a; kernel_ms=b; d2h_ms=c;
        if (fused_gate_up) {
            gate_up_pair_ms=first;
        } else if (g_kimi_up_first.load(std::memory_order_relaxed)) {
            up_ms=first; gate_ms=second;
        } else {
            gate_ms=first; up_ms=second;
        }
        situ_ms=s; down_ms=d;
    }
    if (telemetry == 2) for (int i=0; i<7; ++i) cudaEventDestroy(event[i]);
    if (ok && telemetry) {
        std::lock_guard<std::mutex> lock(g_kimi_stats_mu);
        g_kimi_calls++; g_kimi_rows += (uint64_t)S;
        g_kimi_h2d_bytes += xb; g_kimi_d2h_bytes += xb;
        g_kimi_h2d_ms += h2d_ms; g_kimi_kernel_ms += kernel_ms; g_kimi_d2h_ms += d2h_ms;
        g_kimi_gate_ms += gate_ms; g_kimi_up_ms += up_ms;
        g_kimi_situ_ms += situ_ms; g_kimi_down_ms += down_ms;
        g_kimi_gate_up_pair_ms += gate_up_pair_ms;
    }
    return ok;
}

extern "C" int coli_cuda_kimi_expert_mlp_dev(ColiCudaTensor *gate, ColiCudaTensor *up,
                                               ColiCudaTensor *down, float *y_dev,
                                               const float *x_dev, int S,
                                               float beta, float linear_beta) {
    if (fault_injected()) return 0;
    if (!gate || !up || !down || !x_dev || !y_dev ||
        !kimi_expert_batch_is_certified(S) || beta <= 0.f ||
        linear_beta <= 0.f || gate->fmt != up->fmt || gate->fmt != down->fmt ||
        !((gate->fmt == 7 && gate->gs == 32 && up->gs == 32 && down->gs == 32) ||
          (gate->fmt == 4 && gate->gs == 64 && up->gs == 64 && down->gs == 64)) ||
        gate->device != up->device || gate->device != down->device ||
        gate->I != up->I || gate->O != up->O ||
        down->I != gate->O || down->O != gate->I) return 0;
    DeviceContext *ctx=find_ctx(gate->device);
    if(!select_ctx(ctx)) return 0;
    int D=gate->I,I=gate->O;
    size_t ib=(size_t)S*I*sizeof(float);
    if(!reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)) return 0;
    dim3 hidden_grid((unsigned)I,(unsigned)S),output_grid((unsigned)D,(unsigned)S);
    int fmt=gate->fmt;
    if(fmt==7&&g_kimi_fused_gate_up.load(std::memory_order_relaxed)){
        if(S==16)
            mxfp4_matmul_pair_rows<16><<<(unsigned)I,256>>>(ctx->gate,ctx->up,x_dev,
                (const uint8_t*)gate->weights,(const uint8_t*)up->weights,
                (const uint8_t*)gate->scales,(const uint8_t*)up->scales,
                D,I,row_bytes(7,D),gate->ng);
        else if(S==8)
            mxfp4_matmul_pair_rows<8><<<(unsigned)I,256>>>(ctx->gate,ctx->up,x_dev,
                (const uint8_t*)gate->weights,(const uint8_t*)up->weights,
                (const uint8_t*)gate->scales,(const uint8_t*)up->scales,
                D,I,row_bytes(7,D),gate->ng);
        else if(S==4)
            mxfp4_matmul_pair_rows<4><<<(unsigned)I,256>>>(ctx->gate,ctx->up,x_dev,
                (const uint8_t*)gate->weights,(const uint8_t*)up->weights,
                (const uint8_t*)gate->scales,(const uint8_t*)up->scales,
                D,I,row_bytes(7,D),gate->ng);
        else if(S==2)
            mxfp4_matmul_pair_rows<2><<<(unsigned)I,256>>>(ctx->gate,ctx->up,x_dev,
                (const uint8_t*)gate->weights,(const uint8_t*)up->weights,
                (const uint8_t*)gate->scales,(const uint8_t*)up->scales,
                D,I,row_bytes(7,D),gate->ng);
        else
            mxfp4_matmul_pair<<<hidden_grid,256>>>(ctx->gate,ctx->up,x_dev,
                (const uint8_t*)gate->weights,(const uint8_t*)up->weights,
                (const uint8_t*)gate->scales,(const uint8_t*)up->scales,
                S,D,I,row_bytes(7,D),gate->ng);
    }else{
        if(fmt==7&&(S==2||S==4||S==8||S==16)){
            if(S==16){
                mxfp4_matmul_rows<16><<<(unsigned)I,256>>>(ctx->gate,x_dev,
                    (const uint8_t*)gate->weights,(const uint8_t*)gate->scales,
                    D,I,row_bytes(7,D),gate->ng);
                mxfp4_matmul_rows<16><<<(unsigned)I,256>>>(ctx->up,x_dev,
                    (const uint8_t*)up->weights,(const uint8_t*)up->scales,
                    D,I,row_bytes(7,D),up->ng);
            }else if(S==8){
                mxfp4_matmul_rows<8><<<(unsigned)I,256>>>(ctx->gate,x_dev,
                    (const uint8_t*)gate->weights,(const uint8_t*)gate->scales,
                    D,I,row_bytes(7,D),gate->ng);
                mxfp4_matmul_rows<8><<<(unsigned)I,256>>>(ctx->up,x_dev,
                    (const uint8_t*)up->weights,(const uint8_t*)up->scales,
                    D,I,row_bytes(7,D),up->ng);
            }else if(S==4){
                mxfp4_matmul_rows<4><<<(unsigned)I,256>>>(ctx->gate,x_dev,
                    (const uint8_t*)gate->weights,(const uint8_t*)gate->scales,
                    D,I,row_bytes(7,D),gate->ng);
                mxfp4_matmul_rows<4><<<(unsigned)I,256>>>(ctx->up,x_dev,
                    (const uint8_t*)up->weights,(const uint8_t*)up->scales,
                    D,I,row_bytes(7,D),up->ng);
            }else{
                mxfp4_matmul_rows<2><<<(unsigned)I,256>>>(ctx->gate,x_dev,
                    (const uint8_t*)gate->weights,(const uint8_t*)gate->scales,
                    D,I,row_bytes(7,D),gate->ng);
                mxfp4_matmul_rows<2><<<(unsigned)I,256>>>(ctx->up,x_dev,
                    (const uint8_t*)up->weights,(const uint8_t*)up->scales,
                    D,I,row_bytes(7,D),up->ng);
            }
        }else{
            quant_matmul<<<hidden_grid,256>>>(ctx->gate,x_dev,gate->weights,gate->scales,
                fmt,S,D,I,row_bytes(fmt,D),gate->gs,gate->ng);
            quant_matmul<<<hidden_grid,256>>>(ctx->up,x_dev,up->weights,up->scales,
                fmt,S,D,I,row_bytes(fmt,D),up->gs,up->ng);
        }
    }
    size_t n=(size_t)S*I;
    kimi_situ_mul<<<(unsigned)((n+255)/256),256>>>(ctx->gate,ctx->up,n,beta,linear_beta);
    if(fmt==7&&S==16)
        mxfp4_matmul_rows<16><<<(unsigned)D,256>>>(y_dev,ctx->gate,
            (const uint8_t*)down->weights,(const uint8_t*)down->scales,
            I,D,row_bytes(7,I),down->ng);
    else if(fmt==7&&S==8)
        mxfp4_matmul_rows<8><<<(unsigned)D,256>>>(y_dev,ctx->gate,
            (const uint8_t*)down->weights,(const uint8_t*)down->scales,
            I,D,row_bytes(7,I),down->ng);
    else if(fmt==7&&S==4)
        mxfp4_matmul_rows<4><<<(unsigned)D,256>>>(y_dev,ctx->gate,
            (const uint8_t*)down->weights,(const uint8_t*)down->scales,
            I,D,row_bytes(7,I),down->ng);
    else if(fmt==7&&S==2)
        mxfp4_matmul_rows<2><<<(unsigned)D,256>>>(y_dev,ctx->gate,
            (const uint8_t*)down->weights,(const uint8_t*)down->scales,
            I,D,row_bytes(7,I),down->ng);
    else
        quant_matmul<<<output_grid,256>>>(y_dev,ctx->gate,down->weights,down->scales,
            fmt,S,I,D,row_bytes(fmt,I),down->gs,down->ng);
    return cuda_ok(cudaGetLastError(),"Kimi resident expert MLP launch");
}

extern "C" int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate,ColiCudaTensor *up,
        ColiCudaTensor *down,float *y,const float *x,int S){
    if (fault_injected()) return 0;
    if(!gate||!up||!down||!x||!y||S<1||gate->fmt!=2||up->fmt!=2||down->fmt!=2||
       gate->device!=up->device||gate->device!=down->device||gate->I!=up->I||
       gate->O!=up->O||down->I!=gate->O||down->O!=gate->I)return 0;
    DeviceContext *ctx=find_ctx(gate->device);if(!select_ctx(ctx)||!COLI_GPU_HAS_WMMA||ctx->compute_major<7)return 0;
    int D=gate->I,I=gate->O;size_t xb=(size_t)S*D*sizeof(float),ib=(size_t)S*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->gate,&ctx->gate_cap,ib)||
       !reserve(&ctx->up,&ctx->up_cap,ib)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb))return 0;
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                               "shared w4a16 input upload"))return 0;
    dim3 hidden((unsigned)((I+63)/64),(unsigned)((S+15)/16));
    dim3 output((unsigned)((D+63)/64),(unsigned)((S+15)/16));
    w4a16_gate_up<<<hidden,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,
        (const uint8_t*)gate->weights,(const uint8_t*)up->weights,gate->scales,up->scales,S,D,I);
    silu_mul<<<(unsigned)(((size_t)S*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)S*I);
    w4a16_matmul<<<output,128,0,ctx->stream>>>(ctx->y,ctx->gate,(const uint8_t*)down->weights,down->scales,S,I,D);
    if(!cuda_ok(cudaGetLastError(),"shared w4a16 launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                               "shared w4a16 output download")||
       !cuda_ok(cudaStreamSynchronize(ctx->stream),"shared w4a16 synchronize"))return 0;
    std::memcpy(y,ctx->host_y,xb);
    return 1;
}

extern "C" int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                                        ColiCudaTensor *const *ups,
                                        ColiCudaTensor *const *downs,
                                        const int *rows, int count,
                                        float *y, const float *x) {
    if (fault_injected()) return 0;
    if (!gates || !ups || !downs || !rows || !x || !y || count < 1) return 0;
    ColiCudaTensor *first=gates[0];
    if (!first) return 0;
    int device=first->device,D=first->I,I=first->O,total=0,max_rows=0;
    GroupDesc host[64]; if(count>64) return 0;
    int all_s4=1,all_q4=1,any_g4=0,any_e8=0,all_e8=1;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total,
                 g->gs,u->gs,d->gs};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        all_q4&=(g->fmt==2||g->fmt==4)&&(u->fmt==2||u->fmt==4)&&(d->fmt==2||d->fmt==4)&&
                !(g->gs&1)&&!(u->gs&1)&&!(d->gs&1);   /* even gs: a packed byte never straddles groups */
        any_g4|=g->fmt==4||u->fmt==4||d->fmt==4;
        any_e8|=g->fmt==6||u->fmt==6||d->fmt==6;
        all_e8&=g->fmt==6&&u->fmt==6&&d->fmt==6;
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    /* Mixed E8 groups cannot use either homogeneous grouped kernel. */
    if(any_e8&&!all_e8){
        int off=0;
        for(int c=0;c<count;c++){
            if(!coli_cuda_expert_mlp(gates[c],ups[c],downs[c],
                    y+(size_t)off*D,x+(size_t)off*D,rows[c])) return 0;
            off+=rows[c];
        }
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total; }
        return 1;
    }
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    if(!prepare_group_weights(ctx,gates,ups,downs,count,host)) return 0;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))) return 0;
    int async=!getenv("COLI_CUDA_ASYNC")||atoi(getenv("COLI_CUDA_ASYNC"));
    if(async&&(!reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
               !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb)))return 0;
    cudaError_t copy_desc=async?cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                                cudaMemcpyHostToDevice,ctx->stream)
                               :cudaMemcpy(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_desc,"expert group descriptors"))return 0;
    int profile=getenv("COLI_CUDA_PROFILE")&&atoi(getenv("COLI_CUDA_PROFILE"));
    cudaEvent_t ev[4]={};
    if(profile) for(int i=0;i<4;i++) if(!cuda_ok(cudaEventCreate(&ev[i]),"profile event")){
        for(int j=0;j<i;j++) cudaEventDestroy(ev[j]); profile=0; break; }   /* (#B8) don't leak the events already created */
    if(profile) cudaEventRecord(ev[0],ctx->stream);
    if(async)std::memcpy(ctx->host_x,x,xb);
    cudaError_t copy_x=async?cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream)
                            :cudaMemcpy(ctx->x,x,xb,cudaMemcpyHostToDevice);
    if(!cuda_ok(copy_x,"expert group input upload")) return 0;
    if(profile) cudaEventRecord(ev[1],ctx->stream);
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    int tc=getenv("COLI_CUDA_TC_INT4")&&atoi(getenv("COLI_CUDA_TC_INT4"));
    /* grouped_s4_wmma's body needs __CUDA_ARCH__>=750: on builds where the
     * WMMA kernels are compiled out (COLI_HIP_NO_WMMA) the launch would
     * succeed with an EMPTY kernel and the output buffer would silently keep
     * stale data. Gate the branch like TC_W4A16 below does. */
    tc=tc&&COLI_GPU_HAS_WMMA&&all_s4&&D%32==0&&I%32==0&&D%8==0&&I%8==0;
    int tc_min=getenv("COLI_CUDA_TC_MIN_ROWS")?atoi(getenv("COLI_CUDA_TC_MIN_ROWS")):8;
    for(int c=0;c<count&&tc;c++)tc=rows[c]>=tc_min;
    if(all_e8){
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden_e8_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D);
        if(!e8_rot_rows_dev(ctx->gate,total,I,ctx->stream))return 0;
        grouped_down_e8<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else if(tc){
        size_t qb=(size_t)(total+7)*(size_t)(D>I?D:I)/2;
        if(!reserve_bytes((void**)&ctx->qx,&ctx->qx_cap,qb)||
           !reserve(&ctx->qscale,&ctx->qscale_cap,(size_t)(total+7)*sizeof(float)))return 0;
        cudaMemsetAsync(ctx->qx,0,qb,ctx->stream);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->x,total,D);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->gate,ctx->qx,ctx->qscale,dev,D,I,0);
        grouped_s4_wmma<<<dim3((unsigned)((I+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->up,ctx->qx,ctx->qscale,dev,D,I,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        quantize_s4_rows<<<total,256,0,ctx->stream>>>(ctx->qx,ctx->qscale,ctx->gate,total,I);
        grouped_s4_wmma<<<dim3((unsigned)((D+63)/64),(unsigned)count),256,0,ctx->stream>>>(ctx->y,ctx->qx,ctx->qscale,dev,I,D,2);
    }else if(all_s4&&COLI_GPU_HAS_WMMA&&ctx->compute_major>=7&&getenv("COLI_CUDA_TC_W4A16")&&
             atoi(getenv("COLI_CUDA_TC_W4A16"))&&
             [&]{ int tc16_min=getenv("COLI_CUDA_TC_W4A16_MIN")?atoi(getenv("COLI_CUDA_TC_W4A16_MIN")):16;
                  for(int c=0;c<count;c++) if(rows[c]>=tc16_min) return 1;
                  return 0; }()){
        /* At least one expert has enough rows for a Tensor Core tile. Groups
         * where EVERY expert is below the threshold (decode: r=1) fall through
         * to the grouped-W4 path below — 3 launches for the whole group instead
         * of 4 per expert (#431: the launch flood measured at ~981 micro-kernels
         * per token came from decode riding this branch's per-expert fallback). */
        /* W4A16 Tensor Core per gruppo: attivazioni fp16 per tile (lossless al
         * contrario del path W4A4), un lancio per expert dentro lo stream —
         * l'overhead di lancio e' trascurabile rispetto ai GEMM. */
        int tc16_min=getenv("COLI_CUDA_TC_W4A16_MIN")?atoi(getenv("COLI_CUDA_TC_W4A16_MIN")):16;
        int off16=0;
        for(int c=0;c<count;c++){
            int r=rows[c];
            float *g16=ctx->gate+(size_t)off16*I,*u16=ctx->up+(size_t)off16*I;
            float *x16=ctx->x+(size_t)off16*D,*y16=ctx->y+(size_t)off16*D;
            if(r>=tc16_min){
                dim3 hg16((unsigned)((I+63)/64),(unsigned)((r+15)/16));
                dim3 og16((unsigned)((D+63)/64),(unsigned)((r+15)/16));
                w4a16_gate_up<<<hg16,256,0,ctx->stream>>>(g16,u16,x16,
                    (const uint8_t*)host[c].g,(const uint8_t*)host[c].u,host[c].gs,host[c].us,r,D,I);
                silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
                w4a16_matmul<<<og16,128,0,ctx->stream>>>(y16,g16,
                    (const uint8_t*)host[c].d,host[c].ds,r,I,D);
            }else{
                /* piccoli batch: tile TC quasi vuoti + overhead di lancio — il
                 * kernel naive per-elemento resta piu' veloce (misurato in decode) */
                quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(g16,x16,
                    host[c].g,host[c].gs,host[c].gf,r,D,I,row_bytes(host[c].gf,D),0,1);
                quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(u16,x16,
                    host[c].u,host[c].us,host[c].uf,r,D,I,row_bytes(host[c].uf,D),0,1);
                silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
                quant_matmul<<<dim3((unsigned)D,(unsigned)r),256,0,ctx->stream>>>(y16,g16,
                    host[c].d,host[c].ds,host[c].df,r,I,D,row_bytes(host[c].df,I),0,1);
            }
            off16+=r;
        }
    }else if(all_s4&&(!getenv("COLI_CUDA_W4_PACKED")||atoi(getenv("COLI_CUDA_W4_PACKED")))){
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        int dual=!getenv("COLI_CUDA_DUAL_PROJ")||atoi(getenv("COLI_CUDA_DUAL_PROJ"));
        if(dual)grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        else{   /* non-dual path has no fused epilogue: silu stays a kernel here */
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
            silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        }
        grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else if(all_q4&&any_g4){
        /* grouped-int4 (fmt=4) present: per-group scales (#334). fmt=2 members
         * ride along as the ng=1 special case. silu fused in the dual epilogue. */
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden_g4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        grouped_down_g4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else{
        /* generic path decodes fmt 0/1/2/3 only — a fmt=4 group that slipped the
         * gates above (odd gs) must NOT be silently decoded as int2 (#334). */
        for(int c=0;c<count;c++)
            if(host[c].gf==4||host[c].uf==4||host[c].df==4) return 0;
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count),og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
        grouped_hidden<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
        silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(ctx->gate,ctx->up,(size_t)total*I);
        grouped_down<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }
    if(profile) cudaEventRecord(ev[2],ctx->stream);
    if(!async&&!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
    cudaError_t copy_y=async?cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream)
                            :cudaMemcpy(y,ctx->y,xb,cudaMemcpyDeviceToHost);
    if(!cuda_ok(cudaGetLastError(),"expert group launch")||!cuda_ok(copy_y,"expert group output download"))return 0;
    if(async){if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group synchronize"))return 0;
        std::memcpy(y,ctx->host_y,xb);}
    if(profile){
        cudaEventRecord(ev[3],ctx->stream); cudaEventSynchronize(ev[3]); float a=0,b=0,c=0;
        cudaEventElapsedTime(&a,ev[0],ev[1]); cudaEventElapsedTime(&b,ev[1],ev[2]);
        cudaEventElapsedTime(&c,ev[2],ev[3]);
        { std::lock_guard<std::mutex> lock(g_group_stats_mu);
          g_group_h2d_ms+=a; g_group_kernel_ms+=b; g_group_d2h_ms+=c; }
        for(int i=0;i<4;i++) cudaEventDestroy(ev[i]);
    }
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total; }
    return 1;
}

/* ---- Async expert group (Inc.4): issue/take split of coli_cuda_expert_group ----
 * The measured cost of the sync call at decode is ~0.45 ms/call of HOST-side wait
 * (stream sync + staging), vs ~0.18 ms of actual GPU work — 70% tax, paid ~5x per
 * layer because a token's 8 experts scatter across devices. issue() stages and
 * launches on the device stream and returns immediately; take() syncs and hands
 * back the pinned result rows. One issue may be outstanding per device; moe()
 * takes at each layer end, which also orders the next layer's reuse of the ctx
 * scratch buffers. Small batches only (decode/spec): bigger totals keep the sync
 * path with its TC variants. Numerics are the sync path's small-batch kernels,
 * so greedy output is byte-identical by construction. */
extern "C" int coli_cuda_expert_group_issue(ColiCudaTensor *const *gates,
                                              ColiCudaTensor *const *ups,
                                              ColiCudaTensor *const *downs,
                                              const int *rows, int count,
                                              const float *x) {
    if (!gates || !ups || !downs || !rows || !x || count < 1 || count > 64) return 0;
    ColiCudaTensor *first=gates[0];
    if (!first) return 0;
    int device=first->device,D=first->I,I=first->O,total=0,max_rows=0,all_s4=1,any_e8=0,all_e8=1;
    GroupDesc host[64];
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||rows[c]<1||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,rows[c],total,
                 g->gs,u->gs,d->gs};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        any_e8|=g->fmt==6||u->fmt==6||d->fmt==6;
        all_e8&=g->fmt==6&&u->fmt==6&&d->fmt==6;
        total+=rows[c]; if(rows[c]>max_rows) max_rows=rows[c];
    }
    if(any_e8&&!all_e8) return 0;
    if(total>8) return 0;                       /* decode-scale only */
    DeviceContext *ctx=find_ctx(device); if(!ctx||ctx->group_pending||!select_ctx(ctx)) return 0;
    if(!prepare_group_weights(ctx,gates,ups,downs,count,host)) return 0;
    size_t xb=(size_t)total*D*sizeof(float), ib=(size_t)total*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)count*sizeof(GroupDesc))||
       !reserve_pinned(&ctx->host_x,&ctx->host_x_cap,xb)||
       !reserve_pinned(&ctx->host_y,&ctx->host_y_cap,xb)) return 0;
    std::memcpy(ctx->host_x,x,xb);
    if(!cuda_ok(cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                cudaMemcpyHostToDevice,ctx->stream),
                "expert group issue descriptors")||
       !cuda_ok(cudaMemcpyAsync(ctx->x,ctx->host_x,xb,cudaMemcpyHostToDevice,ctx->stream),
                "expert group issue upload")) return 0;
    if(all_e8){
        GroupDesc *dev=(GroupDesc*)ctx->group_desc;
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count);
        dim3 og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        grouped_hidden_e8_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D);
        if(!e8_rot_rows_dev(ctx->gate,total,I,ctx->stream))return 0;
        grouped_down_e8<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    }else if(all_s4&&(!getenv("COLI_CUDA_W4_PACKED")||atoi(getenv("COLI_CUDA_W4_PACKED")))){
        GroupDesc *dev=(GroupDesc*)ctx->group_desc;
        dim3 hg((unsigned)I,(unsigned)max_rows,(unsigned)count);
        dim3 og((unsigned)D,(unsigned)max_rows,(unsigned)count);
        int dual=!getenv("COLI_CUDA_DUAL_PROJ")||atoi(getenv("COLI_CUDA_DUAL_PROJ"));
        if(dual) grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);
        else {
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->x,dev,I,D,0);
            grouped_hidden_w4<<<hg,256,0,ctx->stream>>>(ctx->up,ctx->x,dev,I,D,1);
            silu_mul<<<(unsigned)(((size_t)total*I+255)/256),256,0,ctx->stream>>>(
                ctx->gate,ctx->up,(size_t)total*I);
        }
        grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    } else for(int c=0;c<count;c++){
        int r=rows[c];
        float *g16=ctx->gate+(size_t)host[c].offset*I,*u16=ctx->up+(size_t)host[c].offset*I;
        float *x16=ctx->x+(size_t)host[c].offset*D,*y16=ctx->y+(size_t)host[c].offset*D;
        quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(g16,x16,
            host[c].g,host[c].gs,host[c].gf,r,D,I,row_bytes(host[c].gf,D),0,1);
        quant_matmul<<<dim3((unsigned)I,(unsigned)r),256,0,ctx->stream>>>(u16,x16,
            host[c].u,host[c].us,host[c].uf,r,D,I,row_bytes(host[c].uf,D),0,1);
        silu_mul<<<(unsigned)(((size_t)r*I+255)/256),256,0,ctx->stream>>>(g16,u16,(size_t)r*I);
        quant_matmul<<<dim3((unsigned)D,(unsigned)r),256,0,ctx->stream>>>(y16,g16,
            host[c].d,host[c].ds,host[c].df,r,I,D,row_bytes(host[c].df,I),0,1);
    }
    if(!cuda_ok(cudaGetLastError(),"expert group issue launch")||
       !cuda_ok(cudaMemcpyAsync(ctx->host_y,ctx->y,xb,cudaMemcpyDeviceToHost,ctx->stream),
                "expert group issue download")) return 0;
    ctx->group_pending=1; ctx->group_pending_bytes=xb;
    { std::lock_guard<std::mutex> lock(g_group_stats_mu);
      g_group_calls++; g_group_experts+=(uint64_t)count; g_group_rows+=(uint64_t)total; }
    return 1;
}

extern "C" const float *coli_cuda_expert_group_take(int device) {
    DeviceContext *ctx=find_ctx(device);
    if(!ctx||!ctx->group_pending) return nullptr;
    ctx->group_pending=0;
    if(!select_ctx(ctx)) return nullptr;
    if(!cuda_ok(cudaStreamSynchronize(ctx->stream),"expert group take")) return nullptr;
    return ctx->host_y;
}


extern "C" int coli_cuda_attention_absorb(ColiCudaTensor *w,float *ctx,const float *q,
                                            const float *latent,const float *rope,int H,int Q,
                                            int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!w||!ctx||!q||!latent||!rope||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>4096||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention rope upload"))return 0;
    size_t shared=(size_t)(2*K+T)*sizeof(float);
    attention_absorb_kernel<<<H,256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,dc->ar,w->weights,w->scales,
        w->fmt,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"attention absorb launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"attention context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention synchronize"))return 0;
    return 1;
}

static int attention_absorb_batch_run(ColiCudaTensor *w,ColiCudaTensor *proj,float *out,
        const float *q,const float *latent,const float *rope,int S,int H,int Q,int R,int V,
        int K,int T,float scale){
    if(!w||!out||!q||!latent||!rope||S<1||H<1||Q<1||R<1||V<1||K<1||K>512||
       T<S||T>8192||w->I!=K||w->O!=H*(Q+V))return 0;
    if(proj&&(proj->device!=w->device||proj->I!=H*V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float),lb=(size_t)T*K*sizeof(float);
    size_t rb=(size_t)T*R*sizeof(float),cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->al,&dc->al_cap,lb)||
       !reserve(&dc->ar,&dc->ar_cap,rb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"attention batch q upload")||
       !cuda_ok(cudaMemcpyAsync(dc->al,latent,lb,cudaMemcpyHostToDevice,dc->stream),"attention batch latent upload")||
       !cuda_ok(cudaMemcpyAsync(dc->ar,rope,rb,cudaMemcpyHostToDevice,dc->stream),"attention batch rope upload"))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,dc->aq,dc->al,
        dc->ar,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"attention batch launch"))return 0;
    const float *src=dc->ac;size_t ob=cb;
    if(proj){
        ob=(size_t)S*proj->O*sizeof(float);if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
        quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
            proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
        if(!cuda_ok(cudaGetLastError(),"attention o_proj launch"))return 0;src=dc->y;
    }
    if(!cuda_ok(cudaMemcpyAsync(out,src,ob,cudaMemcpyDeviceToHost,dc->stream),
                               proj?"attention projected output download":"attention batch context download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"attention batch synchronize"))return 0;
    return 1;
}

extern "C" int coli_cuda_attention_absorb_batch(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent,const float *rope,int S,int H,int Q,int R,int V,int K,int T,
        float scale){
    if (fault_injected()) return 0;
    return attention_absorb_batch_run(w,nullptr,ctx,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

extern "C" int coli_cuda_attention_project_batch(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const float *latent,const float *rope,int S,int H,int Q,
        int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    return attention_absorb_batch_run(w,proj,out,q,latent,rope,S,H,Q,R,V,K,T,scale);
}

extern "C" int coli_cuda_attention_project_ragged(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q,const void *const *keys,
        const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int T,float scale){
    if(!w||!proj||!out||!q||!keys||!latent||!rope||!lengths||S<1||S>512||T<1||T>8192||
       H<1||Q<1||R<1||V<1||K<1||K>512||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);
    if(!select_ctx(dc))return 0;
    float **dl=(float**)std::malloc((size_t)S*sizeof(*dl));
    float **dr=(float**)std::malloc((size_t)S*sizeof(*dr));
    int *old=(int*)std::malloc((size_t)S*sizeof(*old));
    int *add=(int*)std::malloc((size_t)S*sizeof(*add));
    int *off=(int*)std::malloc((size_t)S*sizeof(*off));int packed_n=0;
    if(!dl||!dr||!old||!add||!off){std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);return 0;}
    for(int s=0;s<S;s++){
        if(!keys[s]||lengths[s]<1||lengths[s]>T){std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);return 0;}
        RaggedKVEntry *e=nullptr;
        for(int i=0;i<w->ragged_count;i++)if(w->ragged[i].key==keys[s]){e=&w->ragged[i];break;}
        if(!e){
            if(w->ragged_count>=512){std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);return 0;}
            e=&w->ragged[w->ragged_count++];std::memset(e,0,sizeof(*e));e->key=keys[s];
        }
        if(e->K!=K||e->R!=R||e->host_l!=latent[s]||e->host_r!=rope[s]||lengths[s]<e->length){
            if(e->latent)cudaFree(e->latent);if(e->rope)cudaFree(e->rope);
            e->latent=e->rope=nullptr;e->length=e->capacity=0;
            e->K=K;e->R=R;e->host_l=latent[s];e->host_r=rope[s];
        }
        if(lengths[s]>e->capacity){
            int cap=(lengths[s]+63)&~63;float *nl=nullptr,*nr=nullptr;
            if(!cuda_ok(cudaMalloc(&nl,(size_t)cap*K*sizeof(float)),"ragged KV latent page")||
               !cuda_ok(cudaMalloc(&nr,(size_t)cap*R*sizeof(float)),"ragged KV rope page")){
                if(nl)cudaFree(nl);if(nr)cudaFree(nr);std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);return 0;
            }
            if(e->length){
                cudaMemcpyAsync(nl,e->latent,(size_t)e->length*K*sizeof(float),cudaMemcpyDeviceToDevice,dc->stream);
                cudaMemcpyAsync(nr,e->rope,(size_t)e->length*R*sizeof(float),cudaMemcpyDeviceToDevice,dc->stream);
            }
            if(e->latent)cudaFree(e->latent);if(e->rope)cudaFree(e->rope);
            e->latent=nl;e->rope=nr;e->capacity=cap;
        }
        dl[s]=e->latent;dr[s]=e->rope;old[s]=e->length;add[s]=lengths[s]-e->length;
        off[s]=packed_n;packed_n+=add[s]*(K+R);
    }
    size_t qb=(size_t)S*H*(Q+R)*sizeof(float);
    size_t cb=(size_t)S*H*V*sizeof(float),ob=(size_t)S*proj->O*sizeof(float);
    size_t pb=(size_t)packed_n*sizeof(float);
    size_t desc=(size_t)S*(2*sizeof(float*)+4*sizeof(int));
    int ok=reserve(&dc->aq,&dc->aq_cap,qb)&&reserve(&dc->ac,&dc->ac_cap,cb)&&
           reserve(&dc->y,&dc->y_cap,ob)&&reserve_bytes(&dc->group_desc,&dc->group_desc_cap,desc)&&
           (!pb||(reserve(&dc->al,&dc->al_cap,pb)&&reserve_pinned(&dc->host_kv,&dc->host_kv_cap,pb)));
    char *db=(char*)dc->group_desc;float **ddl=(float**)db,**ddr=ddl+S;
    int *dn=(int*)(ddr+S),*dold=dn+S,*dadd=dold+S,*doff=dadd+S;
    if(ok&&pb){
        for(int s=0;s<S;s++)if(add[s]){
            float *p=dc->host_kv+off[s];
            std::memcpy(p,latent[s]+(size_t)old[s]*K,(size_t)add[s]*K*sizeof(float));
            std::memcpy(p+(size_t)add[s]*K,rope[s]+(size_t)old[s]*R,(size_t)add[s]*R*sizeof(float));
        }
        ok=cuda_ok(cudaMemcpyAsync(dc->al,dc->host_kv,pb,cudaMemcpyHostToDevice,dc->stream),"ragged KV append upload");
    }
    if(ok)ok=cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"ragged q upload")&&
             cuda_ok(cudaMemcpyAsync(ddl,dl,(size_t)S*sizeof(float*),cudaMemcpyHostToDevice,dc->stream),"ragged latent pointers")&&
             cuda_ok(cudaMemcpyAsync(ddr,dr,(size_t)S*sizeof(float*),cudaMemcpyHostToDevice,dc->stream),"ragged rope pointers")&&
             cuda_ok(cudaMemcpyAsync(dn,lengths,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged lengths upload")&&
             cuda_ok(cudaMemcpyAsync(dold,old,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged old lengths")&&
             cuda_ok(cudaMemcpyAsync(dadd,add,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged append lengths")&&
             cuda_ok(cudaMemcpyAsync(doff,off,(size_t)S*sizeof(int),cudaMemcpyHostToDevice,dc->stream),"ragged append offsets");
    if(ok&&pb)ragged_kv_append<<<S,256,0,dc->stream>>>(ddl,ddr,dc->al,dold,dadd,doff,K,R);
    if(ok)for(int s=0;s<S;s++){
        for(int i=0;i<w->ragged_count;i++)if(w->ragged[i].key==keys[s]){w->ragged[i].length=lengths[s];break;}
    }
    std::free(dl);std::free(dr);std::free(old);std::free(add);std::free(off);if(!ok)return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_ragged_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,dc->aq,ddl,ddr,
        dn,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
    return cuda_ok(cudaGetLastError(),"ragged attention launch")&&
           cuda_ok(cudaMemcpyAsync(out,dc->y,ob,cudaMemcpyDeviceToHost,dc->stream),"ragged output download")&&
           cuda_ok(cudaStreamSynchronize(dc->stream),"ragged attention synchronize");
}

extern "C" void coli_cuda_tensor_free(ColiCudaTensor *tensor) {
    if (!tensor) return;
    DeviceContext *ctx = find_ctx(tensor->device);
    if (ctx) select_ctx(ctx);
    if (tensor->tracked && ctx) {
        /* Must mirror the upload's accounting exactly: fmt=6 never charged for a
         * scale buffer, and over-subtracting here trips the >= guard below, which
         * silently leaves the tensor's bytes on the device counter forever. */
        size_t storage_bytes =
#ifdef COLI_ANS
            tensor->compressed ? tensor->archive_bytes :
#endif
            tensor->weight_bytes;
        size_t bytes = storage_bytes + tensor->scale_bytes;
        if (ctx->tensor_count) ctx->tensor_count--;
        if (ctx->tensor_bytes >= bytes) ctx->tensor_bytes -= bytes;
    }
    if (tensor->weights&&tensor->weights_owned) cudaFree(tensor->weights);
    if (tensor->scales) cudaFree(tensor->scales);
    for(int i=0;i<tensor->ragged_count;i++){
        if(tensor->ragged[i].latent)cudaFree(tensor->ragged[i].latent);
        if(tensor->ragged[i].rope)cudaFree(tensor->ragged[i].rope);
    }
    std::free(tensor);
}

extern "C" size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor) {
    if (!tensor) return 0;
    size_t storage_bytes =
#ifdef COLI_ANS
        tensor->compressed ? tensor->archive_bytes :
#endif
        tensor->weight_bytes;
    return storage_bytes + tensor->scale_bytes;
}

extern "C" int coli_cuda_tensor_device(const ColiCudaTensor *tensor) {
    return tensor ? tensor->device : -1;
}

/* ==== resident-pipeline primitives (Inc.0, 2026-07-13) ====
 * Device-side building blocks so the residual stream can stay on the layer's
 * home device across a whole layer. Control flow stays on CPU; only the data
 * plane lives here. All entry points take DEVICE pointers (no transfers) —
 * the caller owns staging via the pipe buffer API below. */

__global__ static void kimi_embedding_bf16(float *output,const uint16_t *table,
                                            const int *token_ids,int count,int D,int vocab){
    size_t linear=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    size_t total=(size_t)count*D;if(linear>=total)return;
    int row=(int)(linear/D),column=(int)(linear%D),token=token_ids[row];
    if(token<0||token>=vocab){output[linear]=__int_as_float(0x7fffffff);return;}
    uint32_t bits=(uint32_t)table[(size_t)token*D+column]<<16;
    output[linear]=__uint_as_float(bits);
}

__global__ static void pipe_rmsnorm_rows(float *y,const float *x,const float *w,
                                         int D,float eps,int xstride,int ystride){
    const float *xr=x+(size_t)blockIdx.x*xstride; float *yr=y+(size_t)blockIdx.x*ystride;
    __shared__ double sh[256];
    double a=0; for(int i=threadIdx.x;i<D;i+=blockDim.x){ double v=xr[i]; a+=v*v; }
    sh[threadIdx.x]=a; __syncthreads();
    for(int s=blockDim.x/2;s>0;s>>=1){ if(threadIdx.x<s) sh[threadIdx.x]+=sh[threadIdx.x+s]; __syncthreads(); }
    float r=rsqrtf((float)(sh[0]/D)+eps);
    for(int i=threadIdx.x;i<D;i+=blockDim.x) yr[i]=xr[i]*r*w[i];
}

/* RoPE interleaved, identical math to glm.c rope_interleave. One block per row;
 * row layout: v + row*stride + offset holds R floats. pos index = row/heads
 * (heads=1 for k_rot rows, heads=H for [S,H,qh] query rows). */
__global__ static void pipe_rope_rows(float *v,const int *pos,int pos_base,int stride,
                                      int offset,int R,int heads,float theta){
    float *p=v+(size_t)blockIdx.x*stride+offset;
    int half=R/2, ps=pos?pos[blockIdx.x/heads]:pos_base+(int)(blockIdx.x/heads);
    __shared__ float in[256];
    for(int j=threadIdx.x;j<R;j+=blockDim.x) in[j]=p[j];
    __syncthreads();
    for(int j=threadIdx.x;j<half;j+=blockDim.x){
        float inv=__powf(theta,-2.0f*j/R);
        float ang=ps*inv, cs=__cosf(ang), sn=__sinf(ang);
        float a=in[2*j], b=in[2*j+1];
        p[j]=a*cs-b*sn; p[half+j]=b*cs+a*sn;
    }
}

__global__ static void pipe_add_n(float *x,const float *t,size_t n){
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]+=t[i];
}

/* Fixed-order partial merge: block b adds partial row b into x row rows[b].
 * Target rows are unique by construction (CPU pre-sums per token), so no
 * atomics — the 9.20.7 lesson. */
__global__ static void pipe_rows_add(float *x,const float *partial,const int *rows,
                                     int D){
    float *xr=x+(size_t)rows[blockIdx.x]*D;
    const float *pr=partial+(size_t)blockIdx.x*D;
    for(int i=threadIdx.x;i<D;i+=blockDim.x) xr[i]+=pr[i];
}

/* scratch persistente per (device,slot): cresce e resta — niente cudaMalloc/Free
 * per layer (78 x ~10 alloc/richiesta erano puro churn). */
extern "C" float *coli_cuda_pipe_scratch(int device,int slot,size_t bytes){
    DeviceContext *ctx=find_ctx(device);
    if(slot<0||slot>=27||!select_ctx(ctx)) return NULL;
    if(!reserve(&ctx->pipe_buf[slot],&ctx->pipe_cap[slot],bytes)) return NULL;
    return ctx->pipe_buf[slot];
}
extern "C" void *coli_cuda_pipe_alloc(int device,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return NULL;
    void *p=NULL;
    if(!cuda_ok(cudaMalloc(&p,bytes),"pipe alloc")) return NULL;
    return p;
}
extern "C" void coli_cuda_pipe_free(int device,void *p){
    DeviceContext *ctx=find_ctx(device); if(!p||!select_ctx(ctx)) return;
    cudaFree(p);
}
extern "C" int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice),"pipe upload");
}
extern "C" int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToHost),"pipe download");
}
extern "C" int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev,
                                      const float *w_dev,int S,int D,float eps){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||!select_ctx(ctx)) return 0;
    pipe_rmsnorm_rows<<<S,256>>>(y_dev,x_dev,w_dev,D,eps,D,D);
    return cuda_ok(cudaGetLastError(),"pipe rmsnorm");
}
extern "C" int coli_cuda_kimi_embedding_bf16_dev(ColiCudaTensor *embedding,
        float *output_dev,const int *token_ids_dev,int count){
    if(!embedding||!output_dev||!token_ids_dev||count<1||count>65536||
       embedding->fmt!=8)return 0;
    DeviceContext *ctx=find_ctx(embedding->device);if(!select_ctx(ctx))return 0;
    size_t total=(size_t)count*embedding->I;
    kimi_embedding_bf16<<<(unsigned)((total+255)/256),256>>>(output_dev,
        (const uint16_t*)embedding->weights,token_ids_dev,count,embedding->I,embedding->O);
    return cuda_ok(cudaGetLastError(),"Kimi BF16 embedding gather");
}
extern "C" int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev,
                                        const float *w_dev,int S,int D,float eps,
                                        int xstride,int ystride){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(S<1||D<1||xstride<D||ystride<D||!select_ctx(ctx)) return 0;
    pipe_rmsnorm_rows<<<S,256>>>(y_dev,x_dev,w_dev,D,eps,xstride,ystride);
    return cuda_ok(cudaGetLastError(),"pipe rmsnorm strided");
}
extern "C" int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,
                                   int rows,int stride,int offset,int R,int heads,
                                   float theta){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    pipe_rope_rows<<<rows,128>>>(v_dev,pos_dev,0,stride,offset,R,heads,theta);
    return cuda_ok(cudaGetLastError(),"pipe rope");
}
extern "C" int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows,
                                        int stride,int offset,int R,int heads,float theta){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device);
    if(rows<1||R<2||R>256||heads<1||!select_ctx(ctx)) return 0;
    pipe_rope_rows<<<rows,128>>>(v_dev,NULL,pos_base,stride,offset,R,heads,theta);
    return cuda_ok(cudaGetLastError(),"pipe rope base");
}
/* ---- device router (#431 PR-A) -------------------------------------------
 * Router for one decode row, entirely on the layer's home device: logits GEMV
 * (E x D, tiny) + sigmoid, bias-augmented top-K selection, route-level TOPP
 * truncation, norm_topk and routed_scale — a float-faithful clone of moe()'s
 * plain routing path (colibri.c FASE A). Selection uses one cooperative block
 * with an explicit (choice descending, expert id ascending) comparator.  That
 * preserves the serial reference's strict-'>', lowest-index tie break while
 * avoiding E x K work on a single CUDA thread; only the dot/expf rounding can
 * differ, which is the documented kernel-family divergence class (#100/#163).
 * Results are packed [idx[K] | w[K] | keff] in one scratch buffer and read
 * back with a single tiny D2H. */
__global__ void pipe_router_logits(const float *__restrict__ x,
                                   const float *__restrict__ W,
                                   const float *__restrict__ bias,
                                   int D, int E, float *logit, float *choice){
    int row = blockIdx.y;
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if(e>=E) return;
    x += (size_t)row*D;
    logit += (size_t)row*E;
    choice += (size_t)row*E;
    const float *w = W + (size_t)e*D;
    float acc = 0.f;
    /* Canonical routing is discontinuous at the top-K cutoff.  Accumulate in
     * input-index order, matching kimi_k3.c's scalar reference, so harmless
     * tree-reduction roundoff cannot substitute a near-tied expert. */
    for(int i=0; i<D; i++) acc += x[i]*w[i];
    float lg = 1.f/(1.f+expf(-acc));
    logit[e]=lg; choice[e]=lg+bias[e];
}
__global__ void pipe_router_select(const float *__restrict__ logit,
                                   const float *__restrict__ choice, int E,
                                   int Ksel, float topp, int norm_topk,
                                   float routed_scale, char *out){
    int row=blockIdx.x;
    size_t pack=(size_t)Ksel*(sizeof(int)+sizeof(float))+sizeof(int);
    logit += (size_t)row*E;
    choice += (size_t)row*E;
    out += (size_t)row*pack;
    int   *idx = (int*)out;
    float *w   = (float*)(out + Ksel*sizeof(int));
    int   *keff= (int*)(out + Ksel*(sizeof(int)+sizeof(float)));
    __shared__ float candidate_value[256];
    __shared__ int candidate_id[256];
    __shared__ int selected[64];
    __shared__ float selected_logit[64];
    for(int kk=0;kk<Ksel;kk++){
        int best=-1; float bv=-1e30f;
        for(int e=threadIdx.x;e<E;e+=blockDim.x){
            int taken=0;
            for(int j=0;j<kk;j++) if(selected[j]==e){taken=1;break;}
            float ev=choice[e];
            if(!taken && (ev>bv || (ev==bv && (best<0 || e<best)))){bv=ev;best=e;}
        }
        candidate_value[threadIdx.x]=bv;
        candidate_id[threadIdx.x]=best;
        __syncthreads();
        for(int stride=blockDim.x>>1;stride>0;stride>>=1){
            if(threadIdx.x<stride){
                float other_value=candidate_value[threadIdx.x+stride];
                int other_id=candidate_id[threadIdx.x+stride];
                float this_value=candidate_value[threadIdx.x];
                int this_id=candidate_id[threadIdx.x];
                if(other_id>=0 && (this_id<0 || other_value>this_value ||
                   (other_value==this_value && other_id<this_id))){
                    candidate_value[threadIdx.x]=other_value;
                    candidate_id[threadIdx.x]=other_id;
                }
            }
            __syncthreads();
        }
        if(!threadIdx.x){
            selected[kk]=candidate_id[0];
            selected_logit[kk]=logit[selected[kk]];
        }
        __syncthreads();
    }
    if(threadIdx.x) return;
    for(int kk=0;kk<Ksel;kk++){idx[kk]=selected[kk];w[kk]=selected_logit[kk];}
    int Ke=Ksel;
    if(topp>0.f && topp<1.f){
        for(int a=1;a<Ksel;a++){ int ii=idx[a]; float ww=w[a]; int b=a-1;
            while(b>=0 && w[b]<ww){ w[b+1]=w[b]; idx[b+1]=idx[b]; b--; } w[b+1]=ww; idx[b+1]=ii; }
        float tot=1e-20f; for(int kk=0;kk<Ksel;kk++) tot+=w[kk];
        float cum=0.f; for(int kk=0;kk<Ksel;kk++){ cum+=w[kk]; if(cum>=topp*tot){ Ke=kk+1; break; } }
    }
    if(norm_topk){ float sm=0.f; for(int kk=0;kk<Ke;kk++) sm+=w[kk]; sm+=1e-20f;
                   for(int kk=0;kk<Ke;kk++) w[kk]/=sm; }
    for(int kk=0;kk<Ke;kk++) w[kk]*=routed_scale;
    *keff=Ke;
}
extern "C" int coli_cuda_pipe_router(int device,const float *x_dev,
        const void *rw_dev,const void *rb_dev,int D,int E,int Ksel,
        float topp,int norm_topk,float routed_scale,
        int *idx_host,float *w_host,int *keff_host){
    DeviceContext *ctx=find_ctx(device);
    if(!x_dev||!rw_dev||!rb_dev||D<1||E<1||E>4096||Ksel<1||Ksel>64||!select_ctx(ctx)) return 0;
    size_t pack=(size_t)Ksel*(sizeof(int)+sizeof(float))+sizeof(int);
    float *logit=coli_cuda_pipe_scratch(device,22,(size_t)E*sizeof(float));
    float *chc  =coli_cuda_pipe_scratch(device,23,(size_t)E*sizeof(float));
    char  *out  =(char*)coli_cuda_pipe_scratch(device,24,pack);
    if(!logit||!chc||!out) return 0;
    int detailed=g_kimi_telemetry_mode.load(std::memory_order_relaxed)==2;
    cudaEvent_t event[4]={};
    if(detailed){
        for(int i=0;i<4;i++)if(!cuda_ok(cudaEventCreate(&event[i]),"Kimi router profile event")){
            for(int j=0;j<i;j++)cudaEventDestroy(event[j]);return 0;}
        cudaEventRecord(event[0],0);
    }
    pipe_router_logits<<<(E+127)/128,128>>>(
        x_dev,(const float*)rw_dev,(const float*)rb_dev,D,E,logit,chc);
    if(detailed)cudaEventRecord(event[1],0);
    pipe_router_select<<<1,256>>>(logit,chc,E,Ksel,topp,norm_topk,routed_scale,out);
    if(detailed)cudaEventRecord(event[2],0);
    if(!cuda_ok(cudaGetLastError(),"pipe router launch")) return 0;
    char buf[64*(sizeof(int)+sizeof(float))+sizeof(int)];
    int copy_ok;
    if(detailed){
        copy_ok=cuda_ok(cudaMemcpyAsync(buf,out,pack,cudaMemcpyDeviceToHost,0),
                        "pipe router readback")&&
                cuda_ok(cudaEventRecord(event[3],0),"pipe router profile record")&&
                cuda_ok(cudaEventSynchronize(event[3]),"pipe router profile sync");
    }else copy_ok=cuda_ok(cudaMemcpy(buf,out,pack,cudaMemcpyDeviceToHost),"pipe router readback");
    if(!copy_ok){if(detailed)for(int i=0;i<4;i++)cudaEventDestroy(event[i]);return 0;}
    if(detailed){
        float logits_ms=0.f,selection_ms=0.f,d2h_ms=0.f;
        int timing_ok=cuda_ok(cudaEventElapsedTime(&logits_ms,event[0],event[1]),
                             "Kimi router logits timing")&&
                      cuda_ok(cudaEventElapsedTime(&selection_ms,event[1],event[2]),
                             "Kimi router selection timing")&&
                      cuda_ok(cudaEventElapsedTime(&d2h_ms,event[2],event[3]),
                             "Kimi router D2H timing");
        for(int i=0;i<4;i++)cudaEventDestroy(event[i]);
        if(!timing_ok)return 0;
        std::lock_guard<std::mutex> lock(g_kimi_stats_mu);
        g_kimi_router_calls++;g_kimi_router_logits_ms+=logits_ms;
        g_kimi_router_selection_ms+=selection_ms;g_kimi_router_d2h_ms+=d2h_ms;
    }
    memcpy(idx_host,buf,(size_t)Ksel*sizeof(int));
    memcpy(w_host,buf+Ksel*sizeof(int),(size_t)Ksel*sizeof(float));
    memcpy(keff_host,buf+Ksel*(sizeof(int)+sizeof(float)),sizeof(int));
    return 1;
}
extern "C" int coli_cuda_pipe_router_batch(int device,const float *x_dev,
        const void *rw_dev,const void *rb_dev,int rows,int D,int E,int Ksel,
        float topp,int norm_topk,float routed_scale,
        int *idx_host,float *w_host,int *keff_host){
    DeviceContext *ctx=find_ctx(device);
    if(!x_dev||!rw_dev||!rb_dev||!idx_host||!w_host||!keff_host||
       !kimi_expert_batch_is_certified(rows)||D<1||E<1||E>4096||
       Ksel<1||Ksel>64||!select_ctx(ctx)) return 0;
    size_t pack=(size_t)Ksel*(sizeof(int)+sizeof(float))+sizeof(int);
    size_t total_pack=(size_t)rows*pack;
    float *logit=coli_cuda_pipe_scratch(device,22,(size_t)rows*E*sizeof(float));
    float *chc  =coli_cuda_pipe_scratch(device,23,(size_t)rows*E*sizeof(float));
    char  *out  =(char*)coli_cuda_pipe_scratch(device,24,total_pack);
    if(!logit||!chc||!out) return 0;
    int detailed=g_kimi_telemetry_mode.load(std::memory_order_relaxed)==2;
    cudaEvent_t event[4]={};
    if(detailed){
        for(int i=0;i<4;i++)if(!cuda_ok(cudaEventCreate(&event[i]),"Kimi router batch profile event")){
            for(int j=0;j<i;j++)cudaEventDestroy(event[j]);return 0;}
        cudaEventRecord(event[0],0);
    }
    dim3 logits_grid((unsigned)((E+127)/128),(unsigned)rows,1);
    pipe_router_logits<<<logits_grid,128>>>(
        x_dev,(const float*)rw_dev,(const float*)rb_dev,D,E,logit,chc);
    if(detailed)cudaEventRecord(event[1],0);
    pipe_router_select<<<rows,256>>>(
        logit,chc,E,Ksel,topp,norm_topk,routed_scale,out);
    if(detailed)cudaEventRecord(event[2],0);
    if(!cuda_ok(cudaGetLastError(),"pipe router batch launch")) return 0;
    char buf[16*(64*(sizeof(int)+sizeof(float))+sizeof(int))];
    int copy_ok;
    if(detailed){
        copy_ok=cuda_ok(cudaMemcpyAsync(buf,out,total_pack,cudaMemcpyDeviceToHost,0),
                        "pipe router batch readback")&&
                cuda_ok(cudaEventRecord(event[3],0),"pipe router batch profile record")&&
                cuda_ok(cudaEventSynchronize(event[3]),"pipe router batch profile sync");
    }else copy_ok=cuda_ok(cudaMemcpy(buf,out,total_pack,cudaMemcpyDeviceToHost),
                          "pipe router batch readback");
    if(!copy_ok){if(detailed)for(int i=0;i<4;i++)cudaEventDestroy(event[i]);return 0;}
    if(detailed){
        float logits_ms=0.f,selection_ms=0.f,d2h_ms=0.f;
        int timing_ok=cuda_ok(cudaEventElapsedTime(&logits_ms,event[0],event[1]),
                             "Kimi router batch logits timing")&&
                      cuda_ok(cudaEventElapsedTime(&selection_ms,event[1],event[2]),
                             "Kimi router batch selection timing")&&
                      cuda_ok(cudaEventElapsedTime(&d2h_ms,event[2],event[3]),
                             "Kimi router batch D2H timing");
        for(int i=0;i<4;i++)cudaEventDestroy(event[i]);
        if(!timing_ok)return 0;
        std::lock_guard<std::mutex> lock(g_kimi_stats_mu);
        g_kimi_router_calls++;g_kimi_router_logits_ms+=logits_ms;
        g_kimi_router_selection_ms+=selection_ms;g_kimi_router_d2h_ms+=d2h_ms;
    }
    for(int row=0;row<rows;row++){
        const char *source=buf+(size_t)row*pack;
        memcpy(idx_host+(size_t)row*Ksel,source,(size_t)Ksel*sizeof(int));
        memcpy(w_host+(size_t)row*Ksel,source+Ksel*sizeof(int),(size_t)Ksel*sizeof(float));
        memcpy(keff_host+row,source+Ksel*(sizeof(int)+sizeof(float)),sizeof(int));
    }
    return 1;
}
/* ---- resident expert-group accumulation (#431 PR-C0) ----------------------
 * Decode-time (S=1) expert groups without the host round-trip: the input row
 * is P2P'd from the layer's home device, the group runs through the grouped-W4
 * kernels on its own stream, the down-projection outputs are weighted and
 * reduced ON DEVICE (fixed expert order), and the device's partial sum is
 * peer-pushed into a per-issue slot on the home device. take() makes the home
 * legacy stream wait on every issue event and reduces the slots in issue order
 * — deterministic, no atomics, no host bytes. The CPU tier overlaps with all
 * of it exactly as before. */
__global__ static void bcast_row(float *dst,const float *src,int count,int D){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<D;i+=gridDim.x*blockDim.x){
        float v=src[i];
        for(int c=0;c<count;c++) dst[(size_t)c*D+i]=v;
    }
}
__global__ static void weighted_sum_rows(float *out,const float *y,const float *w,
                                         int count,int D){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<D;i+=gridDim.x*blockDim.x){
        float acc=0.f;
        for(int c=0;c<count;c++) acc+=w[c]*y[(size_t)c*D+i];   /* fixed order */
        out[i]=acc;
    }
}
__global__ static void sum_slots(float *dst,const float *slots,int n,int D){
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<D;i+=gridDim.x*blockDim.x){
        float acc=0.f;
        for(int s=0;s<n;s++) acc+=slots[(size_t)s*D+i];        /* issue order */
        dst[i]=acc;
    }
}
__global__ static void kimi_attnres_mix(float *out,const float *prefix,
        const float *bres,int nb,const float *sw,int D,float eps){
    __shared__ float score[16],prob[16];
    int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    if(warp<=nb){
        int e=warp;
        const float *v=e<nb?bres+(size_t)e*D:prefix;
        float ms=0.f,dot=0.f;
        for(int d=lane;d<D;d+=32){
            float value=v[d];
            ms+=value*value;dot+=value*sw[d];
        }
        for(int offset=16;offset>0;offset>>=1){
            ms+=__shfl_down_sync(0xffffffff,ms,offset);
            dot+=__shfl_down_sync(0xffffffff,dot,offset);
        }
        if(lane==0)score[e]=dot/sqrtf(ms/(float)D+eps);
    }
    __syncthreads();
    if(threadIdx.x==0){
        float maximum=score[0];
        for(int e=1;e<=nb;e++)if(score[e]>maximum)maximum=score[e];
        float total=0.f;
        for(int e=0;e<=nb;e++){prob[e]=expf(score[e]-maximum);total+=prob[e];}
        for(int e=0;e<=nb;e++)prob[e]/=total;
    }
    __syncthreads();
    for(int d=threadIdx.x;d<D;d+=blockDim.x){
        float value=0.f;
        for(int e=0;e<nb;e++)value+=prob[e]*bres[(size_t)e*D+d];
        value+=prob[nb]*prefix[d];
        out[d]=value;
    }
}
__global__ static void kimi_kda_core(float *output,float *q,float *k,float *v,
        const float *gate,const float *decay,const float *beta_raw,
        const float *conv_q,const float *conv_k,const float *conv_v,
        float *window_q,float *window_k,float *window_v,float *state,
        const float *dt,const float *a,const float *output_norm,int heads,
        int hd,int conv_width,float gate_lower_bound,float eps){
    int h=blockIdx.x,vv=threadIdx.x,d=h*hd+vv;
    if(h>=heads||vv>=hd)return;
    __shared__ float qconv[128],kconv[128],vconv[128];
    __shared__ float qn[128],kn[128],alpha[128],vt[128];
    __shared__ float qsum[128],ksum[128];
    __shared__ double norm_sum[128];
    float *windows[3]={window_q,window_k,window_v};
    const float *taps[3]={conv_q,conv_k,conv_v};
    const float *values[3]={q,k,v};
    for(int w=0;w<3;w++){
        float *window=windows[w]+(size_t)d*conv_width;
        for(int j=0;j<conv_width-1;j++)window[j]=window[j+1];
        window[conv_width-1]=values[w][d];
        float acc=0.f;const float *tap=taps[w]+(size_t)d*conv_width;
        for(int j=0;j<conv_width;j++)acc+=tap[j]*window[j];
        float convolved=acc/(1.f+expf(-acc));
        if(w==0)qconv[vv]=convolved;else if(w==1)kconv[vv]=convolved;
        else vconv[vv]=convolved;
    }
    qsum[vv]=qconv[vv]*qconv[vv];ksum[vv]=kconv[vv]*kconv[vv];__syncthreads();
    for(int stride=hd/2;stride>0;stride>>=1){
        if(vv<stride){qsum[vv]+=qsum[vv+stride];ksum[vv]+=ksum[vv+stride];}
        __syncthreads();
    }
    float qscale=rsqrtf((float)hd);
    qn[vv]=qconv[vv]*rsqrtf(qsum[0]+1e-6f)*qscale;
    kn[vv]=kconv[vv]*rsqrtf(ksum[0]+1e-6f);
    float z=decay[d]+dt[d];
    alpha[vv]=expf(gate_lower_bound/(1.f+expf(-a[h]*z)));
    __syncthreads();
    float *head_state=state+(size_t)h*hd*hd;
    float k_state=0.f;
    for(int kk=0;kk<hd;kk++){
        float *cell=head_state+(size_t)kk*hd+vv;
        float row=*cell*alpha[kk];*cell=row;k_state+=kn[kk]*row;
    }
    float beta=1.f/(1.f+expf(-beta_raw[h]));
    vt[vv]=(vconv[vv]-k_state)*beta;__syncthreads();
    float head_output=0.f;
    for(int kk=0;kk<hd;kk++){
        float *cell=head_state+(size_t)kk*hd+vv;
        float row=*cell+kn[kk]*vt[vv];*cell=row;head_output+=qn[kk]*row;
    }
    norm_sum[vv]=(double)head_output*(double)head_output;__syncthreads();
    for(int stride=hd/2;stride>0;stride>>=1){
        if(vv<stride)norm_sum[vv]+=norm_sum[vv+stride];
        __syncthreads();
    }
    float inverse=rsqrtf((float)(norm_sum[0]/(double)hd)+eps);
    float gate_value=1.f/(1.f+expf(-gate[d]));
    output[d]=head_output*inverse*output_norm[vv]*gate_value;
}
extern "C" int coli_cuda_kimi_moe_reduce_dev(int device,float *out_dev,
        const float *expert_rows_dev,const float *weights_dev,int count,int D){
    if(!out_dev||!expert_rows_dev||!weights_dev||count<1||count>64||D<1)return 0;
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx))return 0;
    int blocks=(D+255)/256; if(blocks>48)blocks=48;
    weighted_sum_rows<<<blocks,256>>>(out_dev,expert_rows_dev,weights_dev,count,D);
    return cuda_ok(cudaGetLastError(),"Kimi MoE reduction launch");
}
extern "C" int coli_cuda_kimi_attnres_mix_dev(int device,float *out_dev,
        const float *prefix_dev,const float *block_residuals_dev,int block_count,
        const float *score_weight_dev,int D,float eps){
    if(!out_dev||!prefix_dev||!block_residuals_dev||!score_weight_dev||
       block_count<1||block_count>15||D<1||eps<=0.f)return 0;
    DeviceContext *ctx=find_ctx(device);if(!select_ctx(ctx))return 0;
    kimi_attnres_mix<<<1,(block_count+1)*32>>>(out_dev,prefix_dev,block_residuals_dev,
        block_count,score_weight_dev,D,eps);
    return cuda_ok(cudaGetLastError(),"Kimi AttnRes mix launch");
}
extern "C" int coli_cuda_kimi_kda_core_dev(int device,float *output_dev,
        float *q_dev,float *k_dev,float *v_dev,const float *gate_dev,
        const float *decay_dev,const float *beta_dev,const float *conv_q_dev,
        const float *conv_k_dev,const float *conv_v_dev,float *window_q_dev,
        float *window_k_dev,float *window_v_dev,float *state_dev,const float *dt_dev,
        const float *a_dev,const float *output_norm_dev,int heads,int head_dim,
        int conv_width,float gate_lower_bound,float eps){
    if(!output_dev||!q_dev||!k_dev||!v_dev||!gate_dev||!decay_dev||!beta_dev||
       !conv_q_dev||!conv_k_dev||!conv_v_dev||!window_q_dev||!window_k_dev||
       !window_v_dev||!state_dev||!dt_dev||!a_dev||!output_norm_dev||
       heads!=96||head_dim!=128||conv_width!=4||gate_lower_bound>=0.f||eps<=0.f)
        return 0;
    DeviceContext *ctx=find_ctx(device);if(!select_ctx(ctx))return 0;
    kimi_kda_core<<<heads,head_dim>>>(output_dev,q_dev,k_dev,v_dev,gate_dev,
        decay_dev,beta_dev,conv_q_dev,conv_k_dev,conv_v_dev,window_q_dev,
        window_k_dev,window_v_dev,state_dev,dt_dev,a_dev,output_norm_dev,
        heads,head_dim,conv_width,gate_lower_bound,eps);
    return cuda_ok(cudaGetLastError(),"Kimi KDA core launch");
}
extern "C" int coli_cuda_kimi_mla_cache_append_dev(int device,float *latent_row_dev,
        float *rope_row_dev,const float *compressed_kv_dev,const float *norm_dev,
        int K,int R,float eps){
    if(!latent_row_dev||!rope_row_dev||!compressed_kv_dev||!norm_dev||
       K<1||K>512||R<1||R>256||eps<=0.f)return 0;
    DeviceContext *ctx=find_ctx(device);if(!select_ctx(ctx))return 0;
    /* Reuse the generic, FP64-accumulating RMSNorm implementation.  The only
     * MLA-specific work here is wiring its normalized latent and raw NoPE tail
     * into the two persistent cache planes without a host synchronization. */
    pipe_rmsnorm_rows<<<1,256>>>(latent_row_dev,compressed_kv_dev,norm_dev,K,eps,K,K);
    if(!cuda_ok(cudaGetLastError(),"Kimi MLA latent-cache RMSNorm launch"))return 0;
    return cuda_ok(cudaMemcpyAsync(rope_row_dev,compressed_kv_dev+K,
        (size_t)R*sizeof(float),cudaMemcpyDeviceToDevice,0),"Kimi MLA NoPE cache append");
}
extern "C" int coli_cuda_kimi_mla_absorb_dev(ColiCudaTensor *kv_b,float *ctx_dev,
        const float *q_dev,const float *latent_dev,const float *rope_dev,
        int H,int Q,int R,int V,int K,int T,float scale){
    if(!kv_b||!ctx_dev||!q_dev||!latent_dev||!rope_dev||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<1||T>COLI_CUDA_KIMI_MLA_MAX_CERTIFIED_CONTEXT||
       kv_b->I!=K||kv_b->O!=H*(Q+V))return 0;
    DeviceContext *ctx=find_ctx(kv_b->device);if(!select_ctx(ctx))return 0;
    size_t shared=(size_t)(2*K+T)*sizeof(float);
    if(!configure_attention_absorb_shared(ctx,shared))return 0;
    attention_absorb_kernel<<<H,256,shared>>>(ctx_dev,q_dev,latent_dev,rope_dev,
        kv_b->weights,kv_b->scales,kv_b->fmt,H,Q,R,V,K,T,scale,kv_b->gs,kv_b->ng);
    return cuda_ok(cudaGetLastError(),"Kimi MLA absorb attention launch");
}
extern "C" int coli_cuda_kimi_mla_sigmoid_gate_dev(int device,float *ctx_dev,
        const float *gate_dev,size_t n){
    if(!ctx_dev||!gate_dev||!n)return 0;
    DeviceContext *ctx=find_ctx(device);if(!select_ctx(ctx))return 0;
    kimi_mla_sigmoid_gate<<<(unsigned)((n+255)/256),256>>>(ctx_dev,gate_dev,n);
    return cuda_ok(cudaGetLastError(),"Kimi MLA sigmoid gate launch");
}
extern "C" int coli_cuda_expert_group_resident_issue(ColiCudaTensor *const *gates,
        ColiCudaTensor *const *ups, ColiCudaTensor *const *downs,
        const float *weights, int count,
        int home_device, const float *x_src_dev, float *partial_slot_dev){
    if(!gates||!ups||!downs||!weights||count<1||count>64||!x_src_dev||!partial_slot_dev) return 0;
    ColiCudaTensor *first=gates[0]; if(!first) return 0;
    int device=first->device,D=first->I,I=first->O;
    GroupDesc host[64];
    int total=0,all_s4=1;
    for(int c=0;c<count;c++){
        ColiCudaTensor *g=gates[c],*u=ups[c],*d=downs[c];
        if(!g||!u||!d||g->device!=device||u->device!=device||d->device!=device||
           g->I!=D||u->I!=D||g->O!=I||u->O!=I||d->I!=I||d->O!=D) return 0;
        host[c]={g->weights,u->weights,d->weights,g->scales,u->scales,d->scales,
                 g->fmt,u->fmt,d->fmt,1,total,
                 g->gs,u->gs,d->gs};
        all_s4&=g->fmt==2&&u->fmt==2&&d->fmt==2;
        total++;
    }
    if(!all_s4) return 0;                       /* resident path: per-row int4 only */
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    if(!prepare_group_weights(ctx,gates,ups,downs,count,host)) return 0;
    if(!ctx->ev_done_ok){
        if(!cuda_ok(cudaEventCreateWithFlags(&ctx->ev_done,cudaEventDisableTiming),
                    "resident group event")) return 0;
        ctx->ev_done_ok=1;
    }
    /* size for the 64-expert cap, not for `count`: reserve() reallocs on growth,
     * and a realloc here could free a buffer the PREVIOUS layer's still-queued
     * async work on this stream reads. Fixed caps make re-issue realloc-free. */
    size_t xb=(size_t)64*D*sizeof(float), ib=(size_t)64*I*sizeof(float);
    if(!reserve(&ctx->x,&ctx->x_cap,xb)||!reserve(&ctx->y,&ctx->y_cap,xb)||
       !reserve(&ctx->gate,&ctx->gate_cap,ib)||!reserve(&ctx->up,&ctx->up_cap,ib)||
       !reserve(&ctx->ac,&ctx->ac_cap,(size_t)(D+64)*sizeof(float))||
       !reserve_bytes(&ctx->group_desc,&ctx->group_desc_cap,(size_t)64*sizeof(GroupDesc)))
        return 0;
    float *w_dev=ctx->ac+D, *partial_local=ctx->ac;
    if(!cuda_ok(cudaMemcpyAsync(ctx->group_desc,host,(size_t)count*sizeof(GroupDesc),
                                cudaMemcpyHostToDevice,ctx->stream),"resident group desc")||
       !cuda_ok(cudaMemcpyAsync(w_dev,weights,(size_t)count*sizeof(float),
                                cudaMemcpyHostToDevice,ctx->stream),"resident group weights"))
        return 0;
    /* input row: P2P from the home device. The caller guarantees x_src_dev is
     * materialized (the pre-moe nrm download already synced the home stream). */
    if(!cuda_ok(cudaMemcpyPeerAsync(ctx->x,device,x_src_dev,home_device,
                                    (size_t)D*sizeof(float),ctx->stream),"resident group x p2p"))
        return 0;
    bcast_row<<<64,256,0,ctx->stream>>>(ctx->x,ctx->x,count,D);   /* row 0 -> rows 1..count-1 (in-place safe: row 0 rewritten with itself) */
    GroupDesc *dev=(GroupDesc*)ctx->group_desc;
    dim3 hg((unsigned)I,1,(unsigned)count),og((unsigned)D,1,(unsigned)count);
    grouped_hidden_w4_dual<<<hg,256,0,ctx->stream>>>(ctx->gate,ctx->up,ctx->x,dev,I,D);  /* silu fused in epilogue */
    grouped_down_w4<<<og,256,0,ctx->stream>>>(ctx->y,ctx->gate,dev,D,I);
    weighted_sum_rows<<<48,256,0,ctx->stream>>>(partial_local,ctx->y,w_dev,count,D);
    if(!cuda_ok(cudaMemcpyPeerAsync(partial_slot_dev,home_device,partial_local,device,
                                    (size_t)D*sizeof(float),ctx->stream),"resident partial p2p"))
        return 0;
    if(!cuda_ok(cudaEventRecord(ctx->ev_done,ctx->stream),"resident event record")) return 0;
    return cuda_ok(cudaGetLastError(),"resident group launch");
}
extern "C" int coli_cuda_expert_group_resident_take(int home_device,const int *devices,int n_issued,
                                           float *slots_dev,float *acc_dev,int D){
    if(n_issued<1||!slots_dev||!acc_dev||D<1) return 0;
    DeviceContext *home=find_ctx(home_device); if(!select_ctx(home)) return 0;
    for(int i=0;i<n_issued;i++){
        DeviceContext *src=find_ctx(devices[i]);
        if(!src||!src->ev_done_ok) return 0;
        if(!cuda_ok(cudaStreamWaitEvent(0,src->ev_done,0),"resident take wait")) return 0;
    }
    sum_slots<<<48,256>>>(acc_dev,slots_dev,n_issued,D);          /* legacy stream: ordered with pipe_* */
    return cuda_ok(cudaGetLastError(),"resident take reduce");
}
extern "C" int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src,
                                     int spitch,int width,int height){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaMemcpy2D(dst,(size_t)dpitch*4,src,(size_t)spitch*4,
        (size_t)width*4,height,cudaMemcpyDeviceToDevice),"pipe copy2d");
}
/* attention batch + fused o_proj with DEVICE-resident q/latent/rope: the whole
 * upstream projection chain stayed on this device, so nothing is uploaded here.
 * Only the final [S,O] projection is downloaded to host. */
extern "C" int coli_cuda_attention_project_batch_dev(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!w||!proj||!out||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe attention launch"))return 0;
    size_t ob=(size_t)S*proj->O*sizeof(float);
    if(!reserve(&dc->y,&dc->y_cap,ob))return 0;
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(dc->y,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe o_proj launch"))return 0;
    if(!cuda_ok(cudaMemcpyAsync(out,dc->y,ob,cudaMemcpyDeviceToHost,dc->stream),"pipe attention download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"pipe attention sync"))return 0;
    return 1;
}
extern "C" int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,
                                       size_t n){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    silu_mul<<<(unsigned)((n+255)/256),256>>>(gate_dev,up_dev,n);
    return cuda_ok(cudaGetLastError(),"pipe silu mul");
}
extern "C" int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(!n||!select_ctx(ctx)) return 0;
    pipe_add_n<<<(unsigned)((n+255)/256),256>>>(x_dev,t_dev,n);
    return cuda_ok(cudaGetLastError(),"pipe add");
}
extern "C" int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev,
                                       const int *rows_dev,int nrows,int D){
    if (fault_injected()) return 0;
    DeviceContext *ctx=find_ctx(device); if(nrows<1||D<1||!select_ctx(ctx)) return 0;
    pipe_rows_add<<<nrows,256>>>(x_dev,partial_dev,rows_dev,D);
    return cuda_ok(cudaGetLastError(),"pipe rows add");
}
/* GEMM with device-resident activations: same quant_matmul kernel as
 * coli_cuda_matmul, zero host transfers. */
extern "C" int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,
                                   int S){
    if (fault_injected()) return 0;
    if(!t||S<1) return 0;
    DeviceContext *ctx=find_ctx(t->device); if(!select_ctx(ctx)) return 0;
    int detailed=g_kimi_telemetry_mode.load(std::memory_order_relaxed)==2;
    cudaEvent_t start=nullptr,stop=nullptr;
    if(detailed){
        if(!cuda_ok(cudaEventCreate(&start),"Kimi dense profile start")||
           !cuda_ok(cudaEventCreate(&stop),"Kimi dense profile stop")||
           !cuda_ok(cudaEventRecord(start,0),"Kimi dense profile record")){
            if(start)cudaEventDestroy(start);if(stop)cudaEventDestroy(stop);return 0;
        }
    }
    dim3 grid((unsigned)t->O,(unsigned)S);
    quant_matmul<<<grid,256>>>(y_dev,x_dev,t->weights,t->scales,t->fmt,S,t->I,t->O,
        row_bytes(t->fmt,t->I),t->gs,t->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe gemm")){
        if(detailed){cudaEventDestroy(start);cudaEventDestroy(stop);}return 0;
    }
    if(detailed){
        float kernel_ms=0.f;
        int ok=cuda_ok(cudaEventRecord(stop,0),"Kimi dense profile stop record")&&
               cuda_ok(cudaEventSynchronize(stop),"Kimi dense profile sync")&&
               cuda_ok(cudaEventElapsedTime(&kernel_ms,start,stop),"Kimi dense timing");
        cudaEventDestroy(start);cudaEventDestroy(stop);if(!ok)return 0;
        std::lock_guard<std::mutex> lock(g_kimi_stats_mu);
        g_kimi_dense_calls++;g_kimi_dense_kernel_ms+=kernel_ms;
    }
    return 1;
}

extern "C" int coli_cuda_pipe_gemm_rows_reuse(ColiCudaTensor *t,float *y_dev,
                                                const float *x_dev,int S){
    if (fault_injected()) return 0;
    if(!t||!y_dev||!x_dev||!(S==1||S==2||S==4||S==8)) return 0;
    if(!(t->fmt==0||t->fmt==1||t->fmt==2||t->fmt==4)) return 0;
    DeviceContext *ctx=find_ctx(t->device);if(!select_ctx(ctx))return 0;
    if(S==1)return coli_cuda_pipe_gemm(t,y_dev,x_dev,S);
    size_t rb=row_bytes(t->fmt,t->I);
    if(S==2)quant_matmul_rows_reuse<2><<<t->O,256>>>(y_dev,x_dev,t->weights,t->scales,
        t->fmt,t->I,t->O,rb,t->gs,t->ng);
    else if(S==4)quant_matmul_rows_reuse<4><<<t->O,256>>>(y_dev,x_dev,t->weights,t->scales,
        t->fmt,t->I,t->O,rb,t->gs,t->ng);
    else quant_matmul_rows_reuse<8><<<t->O,256>>>(y_dev,x_dev,t->weights,t->scales,
        t->fmt,t->I,t->O,rb,t->gs,t->ng);
    return cuda_ok(cudaGetLastError(),"pipe row-reuse gemm");
}
/* copia diretta scheda->scheda (P2P se disponibile, altrimenti staging driver) */
extern "C" int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev,
                                        const float *src,size_t bytes){
    if(!dst||!src) return 0;
    if(dst_dev==src_dev){ DeviceContext *c=find_ctx(dst_dev); if(!select_ctx(c)) return 0;
        return cuda_ok(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToDevice),"pipe intra copy"); }
    return cuda_ok(cudaMemcpyPeer(dst,dst_dev,src,src_dev,bytes),"pipe peer copy");
}
/* come attention_project_batch_dev ma l'uscita di o_proj RESTA sul device (out_dev). */
extern "C" int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *w,ColiCudaTensor *proj,
        float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!w||!proj||!out_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V)||
       proj->device!=w->device||proj->I!=H*V)return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t cb=(size_t)S*H*V*sizeof(float);
    if(!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(dc->ac,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe attention launch (dev out)"))return 0;
    quant_matmul<<<dim3(proj->O,S),256,0,dc->stream>>>(out_dev,dc->ac,proj->weights,
        proj->scales,proj->fmt,S,proj->I,proj->O,row_bytes(proj->fmt,proj->I),proj->gs,proj->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe o_proj launch (dev out)"))return 0;
    return cuda_ok(cudaStreamSynchronize(dc->stream),"pipe attention sync (dev out)");
}
/* absorb batch con TUTTO su device (q/latent/rope gia' residenti sulla scheda
 * dello shard, ctx resta sul device): il cuore della attention head-shardata
 * dentro il pipeline. Nessun trasferimento host. */
extern "C" int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *w,float *ctx_dev,
        const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale){
    if (fault_injected()) return 0;
    if(!w||!ctx_dev||!q_dev||!latent_dev||!rope_dev||S<1||H<1||Q<1||R<1||V<1||
       K<1||K>512||T<S||T>8192||w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,S),256,shared,dc->stream>>>(ctx_dev,q_dev,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,S,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"pipe shard attention launch"))return 0;
    return cuda_ok(cudaStreamSynchronize(dc->stream),"pipe shard attention sync");
}
/* absorb per il DECODE con KV gia' residente: carica solo q (poche KB),
 * latent/rope arrivano dall'ombra device. ctx torna a host (S piccolo). */
extern "C" int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *w,float *ctx,const float *q,
        const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T,
        float scale){
    if (fault_injected()) return 0;
    if(!w||!ctx||!q||!latent_dev||!rope_dev||H<1||Q<1||R<1||V<1||K<1||K>512||T<1||T>8192||
       w->I!=K||w->O!=H*(Q+V))return 0;
    DeviceContext *dc=find_ctx(w->device);if(!select_ctx(dc))return 0;
    size_t qb=(size_t)H*(Q+R)*sizeof(float),cb=(size_t)H*V*sizeof(float);
    if(!reserve(&dc->aq,&dc->aq_cap,qb)||!reserve(&dc->ac,&dc->ac_cap,cb))return 0;
    if(!cuda_ok(cudaMemcpyAsync(dc->aq,q,qb,cudaMemcpyHostToDevice,dc->stream),"kvdev q upload"))return 0;
    size_t shared=(size_t)(2*K+T+256)*sizeof(float);
    attention_absorb_batch_kernel<<<dim3(H,1),256,shared,dc->stream>>>(dc->ac,dc->aq,latent_dev,
        rope_dev,w->weights,w->scales,w->fmt,1,H,Q,R,V,K,T,scale,w->gs,w->ng);
    if(!cuda_ok(cudaGetLastError(),"kvdev absorb launch")||
       !cuda_ok(cudaMemcpyAsync(ctx,dc->ac,cb,cudaMemcpyDeviceToHost,dc->stream),"kvdev ctx download")||
       !cuda_ok(cudaStreamSynchronize(dc->stream),"kvdev absorb sync"))return 0;
    return 1;
}
extern "C" int coli_cuda_pipe_sync(int device){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    return cuda_ok(cudaDeviceSynchronize(),"pipe sync");
}
extern "C" int coli_cuda_error_state_ok(int device){
    DeviceContext *ctx=find_ctx(device); if(!select_ctx(ctx)) return 0;
    cudaError_t err=cudaPeekAtLastError();
    if(err==cudaSuccess) return 1;
    std::fprintf(stderr,"[CUDA] sticky error state: %s\n",cudaGetErrorString(err));
    (void)cudaGetLastError();
    return 0;
}
