#ifndef COLIBRI_BACKEND_CUDA_H
#define COLIBRI_BACKEND_CUDA_H

#include <stddef.h>
#include <stdint.h>

/* COLI_CUDA_DLLEXPORT marks functions exported from coli_cuda.dll on Windows.
 * Define COLI_CUDA_BUILDING_DLL when compiling the .cu into the DLL (so the
 * functions are __declspec(dllexport)); the host loader does NOT include this
 * header's declarations — it resolves symbols at runtime via GetProcAddress. */
#if defined(_WIN32) && defined(COLI_CUDA_BUILDING_DLL)
#define COLI_CUDA_DLLEXPORT __declspec(dllexport)
#else
#define COLI_CUDA_DLLEXPORT
#endif


#ifdef __cplusplus
extern "C" {
#endif

#define COLI_CUDA_MAX_DEVICES 16
#define COLI_CUDA_KIMI_EXPERT_MAX_CERTIFIED_BATCH 16

/* Opaque, persistent device copy of one resident quantized tensor. */
typedef struct ColiCudaTensor ColiCudaTensor;

/* Devices are CUDA ordinals, not positions in the input list. */
COLI_CUDA_DLLEXPORT int coli_cuda_init(const int *devices, int count);
COLI_CUDA_DLLEXPORT void coli_cuda_shutdown(void);
COLI_CUDA_DLLEXPORT int coli_cuda_device_count(void);
COLI_CUDA_DLLEXPORT int coli_cuda_device_at(int index);
COLI_CUDA_DLLEXPORT int coli_cuda_mem_info(int device, size_t *free_bytes, size_t *total_bytes);
/* Per-block shared-memory limits retained from device discovery.  The opt-in
 * value is the hard ceiling used by long-context MLA before any kernel launch. */
COLI_CUDA_DLLEXPORT int coli_cuda_device_shared_memory_limits(int device,
                            size_t *default_bytes, size_t *optin_bytes);
/* Research timer around one or more default-stream operations. The end call
 * synchronizes only the recorded stop event and returns elapsed kernel/queue time. */
COLI_CUDA_DLLEXPORT int coli_cuda_profile_begin(int device);
COLI_CUDA_DLLEXPORT int coli_cuda_profile_end(int device, double *elapsed_ms);
COLI_CUDA_DLLEXPORT int coli_cuda_device_integrated(int device);
COLI_CUDA_DLLEXPORT int coli_cuda_binary_min_compute_capability(void);
COLI_CUDA_DLLEXPORT int coli_cuda_binary_has_forward_ptx(void);
COLI_CUDA_DLLEXPORT int coli_cuda_binary_accepts_compute_capability(int major, int minor);
/* device < 0 returns aggregate statistics for all configured devices. */
COLI_CUDA_DLLEXPORT void coli_cuda_stats(int device, size_t *tensor_count, size_t *tensor_bytes);
COLI_CUDA_DLLEXPORT void coli_cuda_group_stats(uint64_t *calls, uint64_t *experts, uint64_t *rows,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms);
/* Kimi native-MXFP4 fixture telemetry.  Timing fields are populated only when
 * COLI_KIMI_CUDA_TELEMETRY=detailed; calls/rows/bytes are retained for both
 * production and detailed telemetry and are absent in minimal mode. */
COLI_CUDA_DLLEXPORT void coli_cuda_kimi_expert_stats(uint64_t *calls, uint64_t *rows,
                           uint64_t *h2d_bytes, uint64_t *d2h_bytes,
                           double *h2d_ms, double *kernel_ms, double *d2h_ms);
COLI_CUDA_DLLEXPORT void coli_cuda_kimi_expert_phase_stats(double *gate_ms,
                           double *up_ms, double *situ_ms, double *down_ms);
COLI_CUDA_DLLEXPORT void coli_cuda_kimi_expert_pair_stats(double *gate_up_pair_ms);
COLI_CUDA_DLLEXPORT void coli_cuda_kimi_expert_stats_reset(void);
/* mode: 0=minimal, 1=production counters, 2=detailed CUDA-event timing. */
COLI_CUDA_DLLEXPORT int coli_cuda_kimi_set_telemetry(int mode);
/* Research control used to distinguish launch-position from tensor effects. */
COLI_CUDA_DLLEXPORT int coli_cuda_kimi_set_up_first(int up_first);
COLI_CUDA_DLLEXPORT int coli_cuda_kimi_set_fused_gate_up(int fused);
COLI_CUDA_DLLEXPORT void coli_cuda_kimi_router_stats(uint64_t *calls,
                           double *logits_ms, double *selection_ms, double *d2h_ms);
COLI_CUDA_DLLEXPORT void coli_cuda_kimi_router_stats_reset(void);
COLI_CUDA_DLLEXPORT void coli_cuda_kimi_dense_stats(uint64_t *calls,
                           double *kernel_ms);
COLI_CUDA_DLLEXPORT void coli_cuda_kimi_dense_stats_reset(void);

/* Publish the E8 codebook (quant.h's e8_grid, 256x4 bytes) to every configured
 * device. Must be called after coli_cuda_init and before any fmt=6 upload; the
 * backend keeps no copy of the table so it cannot drift from the CPU decoder. */
COLI_CUDA_DLLEXPORT int coli_cuda_e8_set_grid(const void *grid);

/* Upload without executing, so capacity failures happen during model startup. */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload_g(ColiCudaTensor **tensor,
        const void *weights, const float *scales,
        int fmt, int I, int O, int device, int gs);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload(ColiCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);
#ifdef COLI_ANS
/* Experimental Linux-only GPU-resident entropy tier. The archive remains in
 * VRAM and is decoded into per-device scratch immediately before a grouped
 * expert launch. */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_upload_compressed(ColiCudaTensor **tensor,
                            const void *weights, const float *scales,
                            int fmt, int I, int O, int device);
#endif

/*
 * y[S,O] = x[S,I] @ W[O,I]^T.
 * fmt matches QT in glm.c: 0=f32, 1=int8, 2=int4, 3=int2, 4=grouped int4,
 * 6=E8/IQ3, 7=native Kimi MXFP4 E2M1/UE8M0.
 * gs is the group size for fmt=4 or fmt=7 (32 is required for fmt=7).
 * For fmt=7 `scales` points to uint8 UE8M0 values despite the legacy pointer
 * type; the byte-exact checkpoint layout is uploaded without conversion.
 * The first successful call uploads W and its scales; later calls reuse it.
 * Returns 1 on success and 0 when CUDA is not initialized or the format is invalid.
 */
COLI_CUDA_DLLEXPORT int coli_cuda_matmul(ColiCudaTensor **tensor,
                     float *y, const float *x,
                     const void *weights, const float *scales,
                     int fmt, int S, int I, int O, int device, int gs);

/* Fused expert pipeline: y = down(silu(gate(x)) * up(x)).  All three tensors
 * must already be resident on one device.  Activations cross PCIe once in
 * each direction instead of once per matrix. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_mlp(ColiCudaTensor *gate, ColiCudaTensor *up,
                         ColiCudaTensor *down, float *y, const float *x, int S);

/* Native Kimi routed expert: y = down(SiTU(gate(x), up(x))).  All weights are
 * resident fmt=7 tensors and every mathematical operation executes on CUDA.
 * beta and linear_beta are the checkpoint's activation_situ_beta and
 * activation_situ_linear_beta respectively. */
COLI_CUDA_DLLEXPORT int coli_cuda_kimi_expert_mlp(ColiCudaTensor *gate,
                         ColiCudaTensor *up, ColiCudaTensor *down,
                         float *y, const float *x, int S,
                         float beta, float linear_beta);
/* Highest Kimi expert batch size certified against the current native kernels.
 * Larger values are rejected before allocation or launch. */
COLI_CUDA_DLLEXPORT int coli_cuda_kimi_expert_max_certified_batch(void);
/* Exact-size query.  A maximum alone is insufficient while certification is
 * deliberately incremental over power-of-two production batches. */
COLI_CUDA_DLLEXPORT int coli_cuda_kimi_expert_supports_batch(int batch);
/* Persistent-stage variant: x_dev/y_dev are device pointers and no transfer or
 * synchronization occurs inside the call. */
COLI_CUDA_DLLEXPORT int coli_cuda_kimi_expert_mlp_dev(ColiCudaTensor *gate,
                         ColiCudaTensor *up, ColiCudaTensor *down,
                         float *y_dev, const float *x_dev, int S,
                         float beta, float linear_beta);
/* Resident BF16 vocabulary-table gather; token IDs and output stay on device. */
COLI_CUDA_DLLEXPORT int coli_cuda_kimi_embedding_bf16_dev(ColiCudaTensor *embedding,
                         float *output_dev, const int *token_ids_dev, int count);

/* Prefill-oriented shared expert path.  INT4 weights stay packed in global
 * memory, activations are converted to FP16 per tile, and Tensor Cores
 * accumulate into FP32.  Unlike COLI_CUDA_TC_INT4 this does not quantize the
 * activation to INT4. */
COLI_CUDA_DLLEXPORT int coli_cuda_shared_mlp_w4a16(ColiCudaTensor *gate, ColiCudaTensor *up,
                               ColiCudaTensor *down, float *y,
                               const float *x, int S);

/* Packed group of same-shaped experts. Inputs and outputs contain sum(rows)
 * consecutive [D] rows in call order. */
/* Async issue/take split of the group call below (Inc.4): issue launches on the
 * device stream and returns; take syncs and returns the pinned result rows (valid
 * until the next issue on that device). Small totals only (<=8 rows); one
 * outstanding issue per device. */
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_issue(ColiCudaTensor *const *gates,
                               ColiCudaTensor *const *ups,
                               ColiCudaTensor *const *downs,
                               const int *rows, int count, const float *x);
COLI_CUDA_DLLEXPORT const float *coli_cuda_expert_group_take(int device);

COLI_CUDA_DLLEXPORT int coli_cuda_expert_group(ColiCudaTensor *const *gates,
                           ColiCudaTensor *const *ups,
                           ColiCudaTensor *const *downs,
                           const int *rows, int count,
                           float *y, const float *x);

/* Decode-only MLA weight-absorption core for one token. kv_b is [H*(Q+V),K]. */
#define COLI_CUDA_KIMI_MLA_MAX_CERTIFIED_CONTEXT 16385
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb(ColiCudaTensor *kv_b,float *ctx,const float *q,
                               const float *latent,const float *rope,int H,int Q,
                               int R,int V,int K,int T,float attention_scale);

/* Causal MLA absorption for S contiguous rows from one sequence.  The KV
 * arrays contain T rows ending at the final query; query s attends T-S+s+1
 * rows.  One transfer and one launch replace S host round-trips. */
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb_batch(ColiCudaTensor *kv_b,float *ctx,const float *q,
                                     const float *latent,const float *rope,int S,
                                     int H,int Q,int R,int V,int K,int T,
                                     float attention_scale);

/* Same attention batch followed immediately by resident o_proj on the same
 * device.  Only the final [S,D] tensor crosses back to the host. */
COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_batch(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
                                      float *out,const float *q,const float *latent,
                                      const float *rope,int S,int H,int Q,int R,
                                      int V,int K,int T,float attention_scale);

COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_ragged(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q,const void *const *keys,
        const float *const *latent,const float *const *rope,
        const int *lengths,int S,int H,int Q,int R,int V,int K,int max_t,float attention_scale);

COLI_CUDA_DLLEXPORT void coli_cuda_tensor_free(ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT size_t coli_cuda_tensor_bytes(const ColiCudaTensor *tensor);
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_device(const ColiCudaTensor *tensor);

/* Replace a resident tensor's contents without reallocating its device slot. */
COLI_CUDA_DLLEXPORT int coli_cuda_tensor_update(ColiCudaTensor *tensor,
                            const void *weights, const float *scales);

/* ---- resident-pipeline primitives (Inc.0): device-pointer entry points ---- */
COLI_CUDA_DLLEXPORT float *coli_cuda_pipe_scratch(int device,int slot,size_t bytes);
COLI_CUDA_DLLEXPORT void *coli_cuda_pipe_alloc(int device,size_t bytes);
COLI_CUDA_DLLEXPORT void coli_cuda_pipe_free(int device,void *p);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_upload(int device,void *dst,const void *src,size_t bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_download(int device,const void *src,void *dst,size_t bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rmsnorm(int device,float *y_dev,const float *x_dev,
                           const float *w_dev,int S,int D,float eps);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rope(int device,float *v_dev,const int *pos_dev,int rows,
                        int stride,int offset,int R,int heads,float theta);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_silu_mul(int device,float *gate_dev,const float *up_dev,size_t n);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_add(int device,float *x_dev,const float *t_dev,size_t n);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rows_add(int device,float *x_dev,const float *partial_dev,
                            const int *rows_dev,int nrows,int D);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_gemm(ColiCudaTensor *t,float *y_dev,const float *x_dev,int S);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_gemm_rows_reuse(ColiCudaTensor *t,float *y_dev,
                                                       const float *x_dev,int S);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rmsnorm_s(int device,float *y_dev,const float *x_dev,
                             const float *w_dev,int S,int D,float eps,
                             int xstride,int ystride);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_rope_base(int device,float *v_dev,int pos_base,int rows,
                             int stride,int offset,int R,int heads,float theta);
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_resident_issue(ColiCudaTensor *const *gates,
        ColiCudaTensor *const *ups, ColiCudaTensor *const *downs,
        const float *weights, int count,
        int home_device, const float *x_src_dev, float *partial_slot_dev);
COLI_CUDA_DLLEXPORT int coli_cuda_expert_group_resident_take(int home_device,const int *devices,
        int n_issued,float *slots_dev,float *acc_dev,int D);
COLI_CUDA_DLLEXPORT int coli_cuda_kimi_moe_reduce_dev(int device,float *out_dev,
        const float *expert_rows_dev,const float *weights_dev,int count,int D);
COLI_CUDA_DLLEXPORT int coli_cuda_kimi_attnres_mix_dev(int device,float *out_dev,
        const float *prefix_dev,const float *block_residuals_dev,int block_count,
        const float *score_weight_dev,int D,float eps);
COLI_CUDA_DLLEXPORT int coli_cuda_kimi_kda_core_dev(int device,float *output_dev,
        float *q_dev,float *k_dev,float *v_dev,const float *gate_dev,
        const float *decay_dev,const float *beta_dev,const float *conv_q_dev,
        const float *conv_k_dev,const float *conv_v_dev,float *window_q_dev,
        float *window_k_dev,float *window_v_dev,float *state_dev,const float *dt_dev,
        const float *a_dev,const float *output_norm_dev,int heads,int head_dim,
        int conv_width,float gate_lower_bound,float eps);
COLI_CUDA_DLLEXPORT int coli_cuda_kimi_mla_cache_append_dev(int device,
        float *latent_row_dev,float *rope_row_dev,const float *compressed_kv_dev,
        const float *norm_dev,int K,int R,float eps);
COLI_CUDA_DLLEXPORT int coli_cuda_kimi_mla_absorb_dev(ColiCudaTensor *kv_b,
        float *ctx_dev,const float *q_dev,const float *latent_dev,
        const float *rope_dev,int H,int Q,int R,int V,int K,int T,float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_kimi_mla_sigmoid_gate_dev(int device,
        float *ctx_dev,const float *gate_dev,size_t n);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_router(int device,const float *x_dev,
        const void *rw_dev,const void *rb_dev,int D,int E,int Ksel,
        float topp,int norm_topk,float routed_scale,
        int *idx_host,float *w_host,int *keff_host);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_router_batch(int device,const float *x_dev,
        const void *rw_dev,const void *rb_dev,int rows,int D,int E,int Ksel,
        float topp,int norm_topk,float routed_scale,
        int *idx_host,float *w_host,int *keff_host);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_copy2d(int device,float *dst,int dpitch,const float *src,
                          int spitch,int width,int height);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_batch_dev(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb_batch_dev(ColiCudaTensor *kv_b_shard,float *ctx_dev,
        const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_absorb_kvdev(ColiCudaTensor *kv_b,float *ctx,const float *q,
        const float *latent_dev,const float *rope_dev,int H,int Q,int R,int V,int K,int T,
        float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_peer_copy(int dst_dev,float *dst,int src_dev,
                             const float *src,size_t bytes);
COLI_CUDA_DLLEXPORT int coli_cuda_attention_project_batch_dev_out(ColiCudaTensor *kv_b,ColiCudaTensor *o_proj,
        float *out_dev,const float *q_dev,const float *latent_dev,const float *rope_dev,
        int S,int H,int Q,int R,int V,int K,int T,float scale);
COLI_CUDA_DLLEXPORT int coli_cuda_pipe_sync(int device);
/* Non-launching sticky-error inspection used after each newly certified batch. */
COLI_CUDA_DLLEXPORT int coli_cuda_error_state_ok(int device);

#ifdef __cplusplus
}
#endif

#endif
