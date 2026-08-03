/*
 * Copyright 2026 swarm-inference-lab contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * C ABI for the canonical Experiment 010 SWARMEX1/SWARMT01 wire formats.
 * The Python definition in experiment_010/wire.py remains authoritative.
 */
#ifndef SWARM_EXPERT_WIRE_H
#define SWARM_EXPERT_WIRE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SWARM_EXPERT_WIRE_MAGIC "SWARMEX1"
#define SWARM_EXPERT_TENSOR_MAGIC "SWARMT01"
#define SWARM_EXPERT_MAX_FRAME_BYTES (1024ull * 1024ull * 1024ull)
#define SWARM_EXPERT_MAX_DIMS 4

typedef struct {
    const uint8_t *data;
    uint64_t length;
} swarm_expert_blob_view;

typedef struct {
    char kind[9];
    char *header_json;
    size_t header_length;
    char *semantic_json;
    size_t semantic_length;
    swarm_expert_blob_view *blobs;
    size_t blob_count;
} swarm_expert_packet;

typedef struct {
    uint8_t *data;
    size_t length;
} swarm_expert_owned_bytes;

typedef struct {
    char tensor_id[256];
    char request_id[256];
    char model_revision[256];
    char partition_hash[256];
    char dtype[32];
    int stage_id;
    int token_position;
    int sequence_length;
    int route_generation;
    int ndim;
    int64_t shape[SWARM_EXPERT_MAX_DIMS];
    const float *values;
    size_t value_count;
    const uint8_t *raw;
    size_t raw_length;
} swarm_expert_tensor_f32_view;

typedef enum {
    SWARM_EXPERT_RESPONSE_PER_EXPERT_EXACT = 1,
    SWARM_EXPERT_RESPONSE_PER_WORKER_FAST = 2
} swarm_expert_response_mode;

typedef enum {
    SWARM_EXPERT_EXECUTION_WHOLE = 1,
    SWARM_EXPERT_EXECUTION_MICROSHARD = 2
} swarm_expert_execution_mode;

typedef struct {
    char request_id[256];
    char model_id[256];
    char model_revision[256];
    char quantization_fingerprint[256];
    int layer_id;
    int batch_rows;
    int latent_dimension;
    int top_k;
    int *expert_ids_by_row;
    float *routing_weights_by_row;
    int *selected_rank_by_row;
    swarm_expert_response_mode response_mode;
    swarm_expert_execution_mode execution_mode;
    int hidden_start;
    int hidden_end;
    int microshard_final;
    swarm_expert_tensor_f32_view activation;
    swarm_expert_tensor_f32_view down_accumulators;
    int challenge;
} swarm_expert_route_request;

typedef struct {
    char request_id[256];
    char worker_id[256];
    char model_revision[256];
    char model_fingerprint[256];
    char result_hash[80];
    char status[16];
    char error[512];
    int layer_id;
    uint64_t bytes_read;
    uint64_t bytes_received;
    uint64_t bytes_sent;
    uint64_t compute_ns;
    uint64_t queue_ns;
    uint64_t transfer_ns;
    swarm_expert_tensor_f32_view result;
} swarm_expert_route_response;

int swarm_expert_wire_decode_packet(
    const uint8_t *payload,
    size_t payload_length,
    swarm_expert_packet *out,
    char *error,
    size_t error_capacity
);

int swarm_expert_wire_encode_packet(
    const char *kind,
    const char *semantic_json,
    const swarm_expert_blob_view *blobs,
    size_t blob_count,
    swarm_expert_owned_bytes *out,
    char *error,
    size_t error_capacity
);

void swarm_expert_wire_free_packet(swarm_expert_packet *packet);
void swarm_expert_wire_free_bytes(swarm_expert_owned_bytes *bytes);

int swarm_expert_wire_decode_tensor_f32(
    const uint8_t *payload,
    size_t payload_length,
    swarm_expert_tensor_f32_view *out,
    char *error,
    size_t error_capacity
);

int swarm_expert_wire_encode_tensor_f32(
    const char *tensor_id,
    const char *request_id,
    int stage_id,
    int token_position,
    int sequence_length,
    const char *model_revision,
    const char *partition_hash,
    int route_generation,
    const int64_t *shape,
    int ndim,
    const float *values,
    swarm_expert_owned_bytes *out,
    char *error,
    size_t error_capacity
);

int swarm_expert_wire_decode_route_request(
    const uint8_t *payload,
    size_t payload_length,
    swarm_expert_route_request *out,
    char *error,
    size_t error_capacity
);

int swarm_expert_wire_encode_route_request(
    const swarm_expert_route_request *request,
    uint64_t deadline_ns,
    const char *evidence_category,
    int exact_determinism,
    swarm_expert_owned_bytes *out,
    char *error,
    size_t error_capacity
);

void swarm_expert_wire_free_route_request(swarm_expert_route_request *request);

int swarm_expert_wire_decode_route_response(
    const uint8_t *payload,
    size_t payload_length,
    swarm_expert_route_response *out,
    char *error,
    size_t error_capacity
);

int swarm_expert_wire_encode_route_response(
    const swarm_expert_route_response *response,
    const int *experts_executed,
    int expert_count,
    const char *worker_signature,
    swarm_expert_owned_bytes *out,
    char *error,
    size_t error_capacity
);

void swarm_expert_wire_sha256_hex(
    const void *data,
    size_t length,
    char out[65]
);

int swarm_expert_wire_frame_with_length(
    const uint8_t *payload,
    size_t payload_length,
    swarm_expert_owned_bytes *out,
    char *error,
    size_t error_capacity
);

#ifdef __cplusplus
}
#endif
#endif
