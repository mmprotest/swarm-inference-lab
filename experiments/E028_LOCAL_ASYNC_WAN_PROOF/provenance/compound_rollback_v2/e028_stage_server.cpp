#include "ggml-backend.h"
#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <numeric>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
using socket_handle = SOCKET;
static constexpr socket_handle invalid_socket_handle = INVALID_SOCKET;
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
using socket_handle = int;
static constexpr socket_handle invalid_socket_handle = -1;
#endif

namespace {

constexpr uint32_t wire_magic = 0x37323045U; // "E027" in little-endian memory.
constexpr uint16_t wire_version = 1;
constexpr uint32_t max_payload_bytes = 256U * 1024U * 1024U;
constexpr uint32_t max_batch_tokens = 8192;
constexpr uint32_t max_top_k = 64;

enum class operation : uint16_t {
    infer = 1,
    reset = 2,
    ping = 3,
    tokenize = 4,
    shutdown = 5,
    stats = 6,
    checkpoint = 7,
    commit = 8,
    rollback = 9,
    fingerprint = 10,
};

enum class response_kind : uint32_t {
    hidden = 1,
    final = 2,
    tokens = 3,
    json = 4,
    empty = 5,
};

enum request_flags : uint32_t {
    input_tokens = 1U << 0,
    return_nextn = 1U << 1,
    return_full_logits = 1U << 2,
    add_special_tokens = 1U << 3,
    parse_special_tokens = 1U << 4,
    return_tap22 = 1U << 5,
    return_tap44 = 1U << 6,
    input_mtp = 1U << 7,
};

#pragma pack(push, 1)
struct request_header {
    uint32_t magic;
    uint16_t version;
    uint16_t op;
    uint64_t request_id;
    int64_t position;
    int64_t rewind_position;
    uint32_t n_tokens;
    uint32_t n_embd;
    uint32_t flags;
    uint32_t arg;
    uint64_t payload_bytes;
};

struct response_header {
    uint32_t magic;
    uint16_t version;
    uint16_t status;
    uint64_t request_id;
    uint32_t kind;
    uint32_t flags;
    uint32_t n_tokens;
    uint32_t n_embd;
    uint32_t n_vocab;
    uint32_t top_k;
    int32_t stage_start;
    int32_t stage_end;
    uint64_t payload_bytes;
    uint64_t compute_ns;
    uint64_t total_ns;
    uint64_t deserialize_ns;
    uint64_t serialize_ns;
};
#pragma pack(pop)

static_assert(sizeof(request_header) == 56, "wire request header changed");
static_assert(sizeof(response_header) == 88, "wire response header changed");

using clock_type = std::chrono::steady_clock;

uint64_t elapsed_ns(clock_type::time_point begin, clock_type::time_point end) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}

void close_socket(socket_handle value) {
    if (value == invalid_socket_handle) {
        return;
    }
#ifdef _WIN32
    closesocket(value);
#else
    close(value);
#endif
}

class socket_owner {
public:
    explicit socket_owner(socket_handle value = invalid_socket_handle) : value_(value) {}
    ~socket_owner() { close_socket(value_); }
    socket_owner(const socket_owner &) = delete;
    socket_owner & operator=(const socket_owner &) = delete;
    socket_owner(socket_owner && other) noexcept : value_(other.release()) {}
    socket_owner & operator=(socket_owner && other) noexcept {
        if (this != &other) {
            close_socket(value_);
            value_ = other.release();
        }
        return *this;
    }
    socket_handle get() const { return value_; }
    socket_handle release() {
        const socket_handle value = value_;
        value_ = invalid_socket_handle;
        return value;
    }
private:
    socket_handle value_;
};

bool recv_all(socket_handle sock, void * dst, size_t size) {
    auto * cursor = static_cast<uint8_t *>(dst);
    while (size > 0) {
#ifdef _WIN32
        const int chunk = recv(sock, reinterpret_cast<char *>(cursor),
            static_cast<int>(std::min<size_t>(size, INT_MAX)), 0);
#else
        const ssize_t chunk = recv(sock, cursor, size, 0);
#endif
        if (chunk == 0) {
            return false;
        }
        if (chunk < 0) {
#ifdef _WIN32
            if (WSAGetLastError() == WSAEINTR) {
#else
            if (errno == EINTR) {
#endif
                continue;
            }
            throw std::runtime_error("socket receive failed");
        }
        cursor += chunk;
        size -= static_cast<size_t>(chunk);
    }
    return true;
}

void send_all(socket_handle sock, const void * src, size_t size) {
    const auto * cursor = static_cast<const uint8_t *>(src);
    while (size > 0) {
#ifdef _WIN32
        const int chunk = send(sock, reinterpret_cast<const char *>(cursor),
            static_cast<int>(std::min<size_t>(size, INT_MAX)), 0);
#else
        const ssize_t chunk = send(sock, cursor, size, MSG_NOSIGNAL);
#endif
        if (chunk <= 0) {
#ifdef _WIN32
            if (chunk < 0 && WSAGetLastError() == WSAEINTR) {
#else
            if (chunk < 0 && errno == EINTR) {
#endif
                continue;
            }
            throw std::runtime_error("socket send failed");
        }
        cursor += chunk;
        size -= static_cast<size_t>(chunk);
    }
}

template <typename T>
void append_vector(std::vector<uint8_t> & destination, const std::vector<T> & source) {
    const size_t old_size = destination.size();
    const size_t added = source.size() * sizeof(T);
    destination.resize(old_size + added);
    if (added > 0) {
        std::memcpy(destination.data() + old_size, source.data(), added);
    }
}

struct options {
    std::string model_path;
    std::string listen_host = "127.0.0.1";
    uint16_t listen_port = 0;
    int32_t stage_start = -1;
    int32_t stage_end = -1;
    uint32_t n_ctx = 4096;
    uint32_t n_batch = 1024;
    uint32_t n_ubatch = 1024;
    uint32_t n_rs_seq = 16;
    int32_t gpu_layers = 999;
    bool mtp = false;
    bool serial_blocks = false;
    bool profile_layers = false;
    uint32_t checkpoint_slots = 18;
};

int32_t parse_i32(const char * text, const char * name) {
    try {
        size_t used = 0;
        const long long value = std::stoll(text, &used, 10);
        if (used != std::strlen(text) || value < std::numeric_limits<int32_t>::min() ||
                value > std::numeric_limits<int32_t>::max()) {
            throw std::out_of_range(name);
        }
        return static_cast<int32_t>(value);
    } catch (...) {
        throw std::runtime_error(std::string("invalid ") + name + ": " + text);
    }
}

options parse_options(int argc, char ** argv) {
    options result;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto next = [&]() -> const char * {
            if (++index >= argc) {
                throw std::runtime_error("missing value after " + arg);
            }
            return argv[index];
        };
        if (arg == "--model" || arg == "-m") {
            result.model_path = next();
        } else if (arg == "--mtp") {
            result.mtp = true;
        } else if (arg == "--checkpoint-slots") {
            result.checkpoint_slots = static_cast<uint32_t>(parse_i32(next(), "checkpoint-slots"));
        } else if (arg == "--profile-layers") {
            result.profile_layers = true;
        } else if (arg == "--serial-blocks") {
            result.serial_blocks = true;
        } else if (arg == "--host") {
            result.listen_host = next();
        } else if (arg == "--port") {
            const int32_t value = parse_i32(next(), "port");
            if (value <= 0 || value > 65535) {
                throw std::runtime_error("port must be in 1..65535");
            }
            result.listen_port = static_cast<uint16_t>(value);
        } else if (arg == "--stage-start") {
            result.stage_start = parse_i32(next(), "stage-start");
        } else if (arg == "--stage-end") {
            result.stage_end = parse_i32(next(), "stage-end");
        } else if (arg == "--n-ctx") {
            result.n_ctx = static_cast<uint32_t>(parse_i32(next(), "n-ctx"));
        } else if (arg == "--n-batch") {
            result.n_batch = static_cast<uint32_t>(parse_i32(next(), "n-batch"));
        } else if (arg == "--n-ubatch") {
            result.n_ubatch = static_cast<uint32_t>(parse_i32(next(), "n-ubatch"));
        } else if (arg == "--n-rs-seq") {
            result.n_rs_seq = static_cast<uint32_t>(parse_i32(next(), "n-rs-seq"));
        } else if (arg == "--gpu-layers" || arg == "-ngl") {
            result.gpu_layers = parse_i32(next(), "gpu-layers");
        } else if (arg == "--help" || arg == "-h") {
            std::printf(
                "Usage: llama-e027-stage --model FILE --host IP --port N "
                "--stage-start N --stage-end N [--n-ctx N --n-batch N "
                "--n-ubatch N --n-rs-seq N --gpu-layers N]\n");
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (result.model_path.empty() || result.listen_port == 0 ||
            result.stage_start < 0 || result.stage_end <= result.stage_start) {
        throw std::runtime_error("model, port, and a non-empty stage range are required");
    }
    if (result.n_ctx == 0 || result.n_batch == 0 || result.n_ubatch == 0 ||
            result.n_batch > max_batch_tokens || result.n_ubatch > result.n_batch) {
        throw std::runtime_error("invalid context/batch configuration");
    }
    return result;
}

void set_stage_environment(int32_t start, int32_t end) {
    const std::string start_text = std::to_string(start);
    const std::string end_text = std::to_string(end);
#ifdef _WIN32
    if (_putenv_s("E027_STAGE_START", start_text.c_str()) != 0 ||
            _putenv_s("E027_STAGE_END", end_text.c_str()) != 0) {
        throw std::runtime_error("could not configure E027 stage environment");
    }
#else
    if (setenv("E027_STAGE_START", start_text.c_str(), 1) != 0 ||
            setenv("E027_STAGE_END", end_text.c_str(), 1) != 0) {
        throw std::runtime_error("could not configure E027 stage environment");
    }
#endif
}

struct llama_owner {
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    ~llama_owner() {
        if (context) llama_free(context);
        if (model) llama_model_free(model);
    }
};

class stage_runtime {
public:
    explicit stage_runtime(options settings) : settings_(std::move(settings)) {
        if (!settings_.mtp) set_stage_environment(settings_.stage_start, settings_.stage_end);
        const auto load_begin = clock_type::now();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = settings_.gpu_layers;
        model_params.load_mtp = settings_.mtp;
        owner_.model = llama_model_load_from_file(settings_.model_path.c_str(), model_params);
        if (!owner_.model) {
            throw std::runtime_error("failed to load model");
        }
        n_embd_ = llama_model_n_embd(owner_.model);
        n_layers_ = llama_model_n_layer(owner_.model);
        const auto rope_type = llama_model_rope_type(owner_.model);
        n_pos_ = rope_type == LLAMA_ROPE_TYPE_MROPE || rope_type == LLAMA_ROPE_TYPE_IMROPE ? 4 : 1;
        vocab_ = llama_model_get_vocab(owner_.model);
        n_vocab_ = llama_vocab_n_tokens(vocab_);
        if (settings_.stage_end > n_layers_) {
            throw std::runtime_error("stage range exceeds model layer count");
        }
        final_stage_ = settings_.stage_end == n_layers_;

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = settings_.n_ctx;
        context_params.n_batch = settings_.n_batch;
        context_params.n_ubatch = settings_.n_ubatch;
        context_params.n_seq_max = 1;
        context_params.n_rs_seq = settings_.n_rs_seq;
        if (settings_.serial_blocks) {
            context_params.n_seq_max = settings_.checkpoint_slots + 1;
            context_params.n_rs_seq = 0;
            context_params.kv_unified = true;
        }
        checkpoint_positions_.assign(settings_.checkpoint_slots + 1, -1);
        if (settings_.profile_layers) {
            context_params.cb_eval = layer_callback;
            context_params.cb_eval_user_data = this;
        }
        if (settings_.mtp) context_params.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        context_params.no_perf = false;
        owner_.context = llama_init_from_model(owner_.model, context_params);
        if (!owner_.context) {
            throw std::runtime_error("failed to create context");
        }
        llama_set_embeddings(owner_.context, final_stage_ && !settings_.mtp);
        if (final_stage_) {
            llama_set_embeddings_nextn(owner_.context, true, settings_.mtp);
        } else {
            llama_set_embeddings_layer_inp(owner_.context,
                static_cast<uint32_t>(settings_.stage_end), true);
        }
        load_ns_ = elapsed_ns(load_begin, clock_type::now());
    }

    int32_t n_embd() const { return n_embd_; }
    int32_t n_layers() const { return n_layers_; }
    bool final_stage() const { return final_stage_; }
    uint64_t load_ns() const { return load_ns_; }
    uint64_t request_count() const { return request_count_; }

    std::vector<uint8_t> tokenize(const std::vector<uint8_t> & bytes, uint32_t flags) const {
        const char * text = reinterpret_cast<const char *>(bytes.data());
        const int32_t text_size = static_cast<int32_t>(bytes.size());
        const bool add_special = (flags & add_special_tokens) != 0;
        const bool parse_special = (flags & parse_special_tokens) != 0;
        std::vector<llama_token> tokens(bytes.size() + 16);
        int32_t count = llama_tokenize(vocab_, text, text_size, tokens.data(),
            static_cast<int32_t>(tokens.size()), add_special, parse_special);
        if (count < 0) {
            tokens.resize(static_cast<size_t>(-count));
            count = llama_tokenize(vocab_, text, text_size, tokens.data(),
                static_cast<int32_t>(tokens.size()), add_special, parse_special);
        }
        if (count < 0) {
            throw std::runtime_error("tokenization failed");
        }
        std::fprintf(stderr,
            "E027_TOKENIZE payload_bytes=%zu flags=%u token_count=%d\n",
            bytes.size(), flags, count);
        std::fflush(stderr);
        tokens.resize(static_cast<size_t>(count));
        std::vector<uint8_t> result;
        append_vector(result, tokens);
        return result;
    }

    void reset() {
        llama_memory_clear(llama_get_memory(owner_.context), true);
        std::fill(checkpoint_positions_.begin(), checkpoint_positions_.end(), -1);
        current_position_ = 0;
        input_cache_.clear();
    }

    std::vector<uint8_t> infer(
        const request_header & request,
        const std::vector<uint8_t> & payload,
        response_header & response) {
        if (request.n_tokens == 0 || request.n_tokens > settings_.n_batch) {
            throw std::runtime_error("request token count is outside configured batch bounds");
        }
        if (request.position < 0 ||
                request.position + static_cast<int64_t>(request.n_tokens) > settings_.n_ctx) {
            throw std::runtime_error("request positions are outside the context");
        }
        const bool tokens = (request.flags & input_tokens) != 0;
        const bool mtp = (request.flags & input_mtp) != 0;
        if (mtp != settings_.mtp) throw std::runtime_error("MTP input/context mismatch");
        if (tokens != (settings_.stage_start == 0)) {
            throw std::runtime_error("input kind does not match this stage");
        }
        const size_t token_bytes = static_cast<size_t>(request.n_tokens) * sizeof(llama_token);
        const size_t expected = mtp ? token_bytes + static_cast<size_t>(request.n_tokens) * n_embd_ * sizeof(float) : tokens
            ? static_cast<size_t>(request.n_tokens) * sizeof(llama_token)
            : static_cast<size_t>(request.n_tokens) * n_embd_ * sizeof(float);
        if (request.n_embd != static_cast<uint32_t>(n_embd_) || payload.size() != expected) {
            throw std::runtime_error("input tensor shape or payload size mismatch");
        }
        if (request.rewind_position < -1) commit_to(-request.rewind_position - 2);
        if (request.rewind_position >= 0) {
            if (settings_.serial_blocks) rollback_to(request.rewind_position);
            else if (!llama_memory_seq_rm(llama_get_memory(owner_.context), 0,
                        static_cast<llama_pos>(request.rewind_position), -1))
                throw std::runtime_error("draft state rollback failed");
            current_position_ = request.rewind_position;
        }
        if (request.position != current_position_)
            throw std::runtime_error("noncontiguous sequence position");
        if (request.flags & (1U << 8)) {
            checkpoint_at(request.position);
            input_cache_[request.position] = {request, payload};
        }

        // One WAN work unit, scalar local kernels. Each row's state stays here.
        if (settings_.serial_blocks && !(request.flags & (1U << 9)) && request.position > 0 && request.n_tokens > 1) {
            std::array<std::vector<uint8_t>, 6> columns;
            uint64_t compute_ns = 0;
            const size_t stride = expected / request.n_tokens;
            for (uint32_t i = 0; i < request.n_tokens; ++i) {
                request_header one = request;
                one.position += i;
                one.rewind_position = -1;
                one.flags &= ~(1U << 8);
                one.n_tokens = 1;
                one.payload_bytes = stride;
                std::vector<uint8_t> input(payload.begin() + i * stride,
                                           payload.begin() + (i + 1) * stride);
                response_header part{};
                auto row = infer(one, input, part);
                compute_ns += part.compute_ns;
                if (!final_stage_) {
                    columns[0].insert(columns[0].end(), row.begin(), row.end());
                } else {
                    std::array<size_t, 6> sizes = {
                        part.top_k * sizeof(int32_t), part.top_k * sizeof(float),
                        (part.flags & return_nextn) ? n_embd_ * sizeof(float) : 0,
                        (part.flags & return_full_logits) ? n_vocab_ * sizeof(float) : 0,
                        (part.flags & return_tap22) ? n_embd_ * sizeof(float) : 0,
                        (part.flags & return_tap44) ? n_embd_ * sizeof(float) : 0,
                    };
                    size_t offset = 0;
                    for (size_t c = 0; c < sizes.size(); ++c) {
                        columns[c].insert(columns[c].end(), row.begin() + offset,
                                          row.begin() + offset + sizes[c]);
                        offset += sizes[c];
                    }
                    if (offset != row.size()) throw std::runtime_error("scalar row layout mismatch");
                }
                response.kind = part.kind;
                response.flags = part.flags;
                response.n_embd = part.n_embd;
                response.n_vocab = part.n_vocab;
                response.top_k = part.top_k;
            }
            response.n_tokens = request.n_tokens;
            response.compute_ns = compute_ns;
            std::vector<uint8_t> result;
            for (auto & column : columns) result.insert(result.end(), column.begin(), column.end());
            return result;
        }

        llama_batch batch = llama_batch_init(
            static_cast<int32_t>(request.n_tokens), tokens && !mtp ? 0 : n_embd_, 1);
        if (mtp) batch.token = static_cast<llama_token *>(std::malloc(token_bytes));
        llama_pos * const allocated_positions = batch.pos;
        std::vector<llama_pos> hidden_positions;
        bool batch_freed = false;
        auto free_batch = [&]() {
            if (!batch_freed) {
                batch.pos = allocated_positions;
                llama_batch_free(batch);
                batch_freed = true;
            }
        };
        if ((tokens && !batch.token) || (!tokens && !batch.embd)) {
            free_batch();
            throw std::runtime_error("could not allocate llama batch");
        }
        batch.n_tokens = static_cast<int32_t>(request.n_tokens);
        if (mtp) {
            std::memcpy(batch.token, payload.data(), token_bytes);
            std::memcpy(batch.embd, payload.data() + token_bytes, expected - token_bytes);
        } else if (tokens) {
            std::memcpy(batch.token, payload.data(), expected);
        } else {
            std::memcpy(batch.embd, payload.data(), expected);
            hidden_positions.resize(static_cast<size_t>(request.n_tokens) * n_pos_);
            batch.pos = hidden_positions.data();
        }
        for (uint32_t index = 0; index < request.n_tokens; ++index) {
            const llama_pos position = static_cast<llama_pos>(request.position + index);
            if (tokens) {
                batch.pos[index] = position;
            } else {
                // Generic llama text-position layout, selected by public RoPE metadata.
                for (uint32_t dim = 0; dim < n_pos_; ++dim)
                    batch.pos[dim * request.n_tokens + index] = dim < 3 ? position : 0;
            }
            batch.n_seq_id[index] = 1;
            batch.seq_id[index][0] = 0;
            batch.logits[index] = 1;
        }

        std::fprintf(stderr,
            "E027_INFER_BEGIN stage=[%d,%d) position=%lld tokens=%u input=%s payload_bytes=%zu\n",
            settings_.stage_start, settings_.stage_end,
            static_cast<long long>(request.position), request.n_tokens,
            tokens ? "tokens" : "hidden", payload.size());
        std::fflush(stderr);
        const auto compute_begin = clock_type::now();
        layer_last_ = compute_begin;
        const int result = llama_decode(owner_.context, batch);
        if (result != 0) {
            free_batch();
            throw std::runtime_error("llama_decode failed with code " + std::to_string(result));
        }
        llama_synchronize(owner_.context);
        std::fprintf(stderr, "E027_INFER_END stage=[%d,%d) result=%d\n",
            settings_.stage_start, settings_.stage_end, result);
        std::fflush(stderr);
        response.compute_ns = elapsed_ns(compute_begin, clock_type::now());
        ++request_count_;
        current_position_ = request.position + request.n_tokens;

        std::vector<uint8_t> output;
        response.n_tokens = request.n_tokens;
        response.n_embd = static_cast<uint32_t>(n_embd_);
        if (!final_stage_) {
            response.kind = static_cast<uint32_t>(response_kind::hidden);
            std::vector<float> hidden(static_cast<size_t>(request.n_tokens) * n_embd_);
            const float * boundary = llama_get_embeddings_layer_inp(owner_.context,
                static_cast<uint32_t>(settings_.stage_end));
            if (!boundary) {
                free_batch();
                throw std::runtime_error("stage boundary layer input is unavailable");
            }
            std::memcpy(hidden.data(), boundary, hidden.size() * sizeof(float));
            append_vector(output, hidden);
            free_batch();
            return output;
        }

        response.kind = static_cast<uint32_t>(response_kind::final);
        response.n_vocab = static_cast<uint32_t>(n_vocab_);
        response.flags = request.flags & (return_nextn | return_full_logits | return_tap22 | return_tap44);
        const uint32_t top_k = std::max<uint32_t>(1,
            std::min<uint32_t>(request.arg == 0 ? 16 : request.arg, max_top_k));
        response.top_k = top_k;

        std::vector<int32_t> top_ids(static_cast<size_t>(request.n_tokens) * top_k);
        std::vector<float> top_values(static_cast<size_t>(request.n_tokens) * top_k);
        std::vector<float> full_logits;
        if ((request.flags & return_full_logits) != 0) {
            full_logits.resize(static_cast<size_t>(request.n_tokens) * n_vocab_);
        }
        std::vector<int32_t> indices(static_cast<size_t>(n_vocab_));
        std::iota(indices.begin(), indices.end(), 0);
        for (uint32_t row_index = 0; row_index < request.n_tokens; ++row_index) {
            const float * logits = llama_get_logits_ith(owner_.context,
                static_cast<int32_t>(row_index));
            if (!logits) {
                free_batch();
                throw std::runtime_error("final logits are unavailable");
            }
            std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                [logits](int32_t left, int32_t right) {
                    if (logits[left] == logits[right]) return left < right;
                    return logits[left] > logits[right];
                });
            for (uint32_t rank = 0; rank < top_k; ++rank) {
                const size_t offset = static_cast<size_t>(row_index) * top_k + rank;
                top_ids[offset] = indices[rank];
                top_values[offset] = logits[indices[rank]];
            }
            if (!full_logits.empty()) {
                std::memcpy(full_logits.data() + static_cast<size_t>(row_index) * n_vocab_,
                    logits, static_cast<size_t>(n_vocab_) * sizeof(float));
            }
        }
        append_vector(output, top_ids);
        append_vector(output, top_values);

        if ((request.flags & return_nextn) != 0) {
            std::vector<float> nextn(static_cast<size_t>(request.n_tokens) * n_embd_);
            for (uint32_t index = 0; index < request.n_tokens; ++index) {
                const float * row = llama_get_embeddings_nextn_ith(owner_.context,
                    static_cast<int32_t>(index));
                if (!row) {
                    free_batch();
                    throw std::runtime_error("target nextn hidden state is unavailable");
                }
                std::memcpy(nextn.data() + static_cast<size_t>(index) * n_embd_, row,
                    static_cast<size_t>(n_embd_) * sizeof(float));
            }
            append_vector(output, nextn);
        }
        append_vector(output, full_logits);
        free_batch();
        return output;
    }

private:
    options settings_;
    llama_owner owner_;
    const llama_vocab * vocab_ = nullptr;
    int32_t n_embd_ = 0;
    uint32_t n_pos_ = 1;
    struct cached_input { request_header request; std::vector<uint8_t> payload; };
    std::map<int64_t, cached_input> input_cache_;
    int32_t n_vocab_ = 0;
    int32_t n_layers_ = 0;
    bool final_stage_ = false;
    uint64_t load_ns_ = 0;
    uint64_t request_count_ = 0;
    std::vector<int64_t> checkpoint_positions_;
    int64_t current_position_ = 0;
    uint64_t checkpoint_ns_ = 0, commit_ns_ = 0, rollback_ns_ = 0;
    uint64_t checkpoint_count_ = 0, commit_count_ = 0, rollback_count_ = 0;
    std::vector<std::pair<int, uint64_t>> layer_samples_;
    clock_type::time_point layer_last_;
public:
    static bool layer_callback(ggml_tensor * tensor, bool ask, void * user) {
        int layer = -1;
        if (std::sscanf(tensor->name, "l_out-%d", &layer) != 1) return ask ? false : true;
        if (ask) return true;
        auto * self = static_cast<stage_runtime *>(user);
        const auto now = clock_type::now();
        self->layer_samples_.emplace_back(layer, elapsed_ns(self->layer_last_, now));
        self->layer_last_ = now;
        return true;
    }
    void checkpoint_at(int64_t position) {
        if (!settings_.serial_blocks || position != current_position_)
            throw std::runtime_error("invalid checkpoint request");
        auto begin = clock_type::now();
        auto mem = llama_get_memory(owner_.context);
        int slot = 1;
        for (int i = 1; i < (int) checkpoint_positions_.size(); ++i) {
            if (checkpoint_positions_[i] == position) return;
            if (checkpoint_positions_[i] < checkpoint_positions_[slot]) slot = i;
        }
        input_cache_.erase(checkpoint_positions_[slot]);
        llama_memory_seq_rm(mem, slot, -1, -1);
        llama_memory_seq_cp(mem, 0, slot, -1, -1);
        checkpoint_positions_[slot] = position;
        llama_synchronize(owner_.context);
        checkpoint_ns_ += elapsed_ns(begin, clock_type::now());
        ++checkpoint_count_;
    }
    void commit_to(int64_t position) {
        auto begin = clock_type::now();
        auto mem = llama_get_memory(owner_.context);
        for (int i = 1; i < (int) checkpoint_positions_.size(); ++i) {
            if (checkpoint_positions_[i] >= 0 && checkpoint_positions_[i] < position) {
                llama_memory_seq_rm(mem, i, -1, -1);
                input_cache_.erase(checkpoint_positions_[i]);
                checkpoint_positions_[i] = -1;
            }
        }
        llama_synchronize(owner_.context);
        commit_ns_ += elapsed_ns(begin, clock_type::now());
        ++commit_count_;
    }
    void rollback_to(int64_t position) {
        auto begin = clock_type::now();
        auto mem = llama_get_memory(owner_.context);
        int slot = -1;
        for (int i = 1; i < (int) checkpoint_positions_.size(); ++i)
            if (checkpoint_positions_[i] == position) slot = i;
        if (slot < 0) throw std::runtime_error("chunk checkpoint unavailable");
        llama_synchronize(owner_.context);
        if (!llama_memory_seq_rm(mem, 0, -1, -1)) throw std::runtime_error("rollback remove failed");
        llama_memory_seq_cp(mem, slot, 0, -1, -1);
        for (int i = 1; i < (int) checkpoint_positions_.size(); ++i) {
            if (checkpoint_positions_[i] >= position) {
                llama_memory_seq_rm(mem, i, -1, -1);
                input_cache_.erase(checkpoint_positions_[i]);
                checkpoint_positions_[i] = -1;
            }
        }
        current_position_ = position;
        llama_synchronize(owner_.context);
        rollback_ns_ += elapsed_ns(begin, clock_type::now());
        ++rollback_count_;
    }
    std::string fingerprint_json() {
        llama_synchronize(owner_.context);
        auto hash = [&](llama_state_seq_flags flags) {
            size_t n = llama_state_seq_get_size_ext(owner_.context, 0, flags);
            std::vector<uint8_t> data(n);
            if (llama_state_seq_get_data_ext(owner_.context, data.data(), n, 0, flags) != n)
                throw std::runtime_error("state export failed");
            uint64_t h = 14695981039346656037ULL;
            for (uint8_t b : data) h = (h ^ b) * 1099511628211ULL;
            return std::to_string(h);
        };
        return "{\"position\":" + std::to_string(current_position_) +
            ",\"memory_max_position\":" + std::to_string(llama_memory_seq_pos_max(llama_get_memory(owner_.context), 0)) +
            ",\"full_state_fnv64\":\"" + hash(LLAMA_STATE_SEQ_FLAGS_NONE) +
            "\",\"partial_state_fnv64\":\"" + hash(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) + "\"}";
    }
    std::vector<uint8_t> rollback_request(const request_header & request, response_header & response) {
        const auto found = input_cache_.find(request.position);
        if (found == input_cache_.end()) throw std::runtime_error("cached checkpoint input unavailable");
        auto cached = found->second;
        if (request.arg > cached.request.n_tokens) throw std::runtime_error("replay prefix exceeds cached chunk");
        rollback_to(request.position);
        const std::string boundary = (request.flags & (1U << 10)) ? fingerprint_json() : "null";
        uint64_t hash = 14695981039346656037ULL;
        if (request.arg) {
            auto one = cached.request;
            one.flags &= ~(1U << 8);
            one.rewind_position = -1;
            const size_t stride = cached.payload.size() / one.n_tokens;
            one.n_tokens = request.arg;
            cached.payload.resize(stride * one.n_tokens);
            one.payload_bytes = cached.payload.size();
            response_header part{};
            const auto output = infer(one, cached.payload, part);
            response.compute_ns = part.compute_ns;
            for (uint8_t b : output) hash = (hash ^ b) * 1099511628211ULL;
        }
        const std::string json = "{\"position\":" + std::to_string(current_position_) +
            ",\"replayed_rows\":" + std::to_string(request.arg) +
            ",\"replay_output_fnv64\":\"" + std::to_string(hash) +
            "\",\"boundary_state\":" + boundary + "}";
        response.kind = static_cast<uint32_t>(response_kind::json);
        return std::vector<uint8_t>(json.begin(), json.end());
    }
    std::string stats_json() {
        std::string s = "{\"position\":" + std::to_string(current_position_) +
          ",\"checkpoint_ns\":" + std::to_string(checkpoint_ns_) +
          ",\"checkpoint_count\":" + std::to_string(checkpoint_count_) +
          ",\"commit_ns\":" + std::to_string(commit_ns_) +
          ",\"commit_count\":" + std::to_string(commit_count_) +
          ",\"rollback_ns\":" + std::to_string(rollback_ns_) +
          ",\"rollback_count\":" + std::to_string(rollback_count_) + ",\"layer_samples\":[";
        for (size_t i = 0; i < layer_samples_.size(); ++i) {
            if (i) s += ",";
            s += "[" + std::to_string(layer_samples_[i].first) + "," + std::to_string(layer_samples_[i].second) + "]";
        }
        layer_samples_.clear();
        return s + "]}";
    }
};

socket_owner listen_socket(const std::string & host, uint16_t port) {
    socket_owner server(socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (server.get() == invalid_socket_handle) {
        throw std::runtime_error("could not create listen socket");
    }
    int enabled = 1;
    setsockopt(server.get(), SOL_SOCKET, SO_REUSEADDR,
        reinterpret_cast<const char *>(&enabled), sizeof(enabled));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
        throw std::runtime_error("listen host must be a numeric IPv4 address");
    }
    if (bind(server.get(), reinterpret_cast<const sockaddr *>(&address),
            sizeof(address)) != 0) {
        throw std::runtime_error("could not bind listen socket");
    }
    if (listen(server.get(), 8) != 0) {
        throw std::runtime_error("could not listen");
    }
    return server;
}

void send_response(socket_handle client, response_header response,
        const std::vector<uint8_t> & payload) {
    response.payload_bytes = payload.size();
    std::vector<uint8_t> frame(sizeof(response) + payload.size());
    std::memcpy(frame.data(), &response, sizeof(response));
    if (!payload.empty()) std::memcpy(frame.data() + sizeof(response), payload.data(), payload.size());
    send_all(client, frame.data(), frame.size());
}

bool serve_connection(socket_handle client, stage_runtime & runtime,
        const options & settings, bool & shutdown_requested) {
    int enabled = 1;
    setsockopt(client, IPPROTO_TCP, TCP_NODELAY,
        reinterpret_cast<const char *>(&enabled), sizeof(enabled));
    while (!shutdown_requested) {
        request_header request{};
        if (!recv_all(client, &request, sizeof(request))) {
            return true;
        }
        if (request.magic != wire_magic || request.version != wire_version ||
                request.payload_bytes > max_payload_bytes) {
            throw std::runtime_error("invalid E027 request frame");
        }
        const auto total_begin = clock_type::now();
        std::vector<uint8_t> payload(static_cast<size_t>(request.payload_bytes));
        const auto deserialize_begin = clock_type::now();
        if (!payload.empty() && !recv_all(client, payload.data(), payload.size())) {
            return false;
        }
        const auto deserialize_end = clock_type::now();

        response_header response{};
        response.magic = wire_magic;
        response.version = wire_version;
        response.request_id = request.request_id;
        response.stage_start = settings.stage_start;
        response.stage_end = settings.stage_end;
        response.n_embd = static_cast<uint32_t>(runtime.n_embd());
        response.deserialize_ns = elapsed_ns(deserialize_begin, deserialize_end);
        std::vector<uint8_t> result;
        try {
            switch (static_cast<operation>(request.op)) {
                case operation::infer:
                    result = runtime.infer(request, payload, response);
                    break;
                case operation::checkpoint:
                    runtime.checkpoint_at(request.position);
                    response.kind = static_cast<uint32_t>(response_kind::empty);
                    break;
                case operation::commit:
                    runtime.commit_to(request.position);
                    response.kind = static_cast<uint32_t>(response_kind::empty);
                    break;
                case operation::rollback:
                    result = runtime.rollback_request(request, response);
                    break;
                case operation::stats:
                case operation::fingerprint: {
                    auto json = request.op == static_cast<uint16_t>(operation::stats)
                        ? runtime.stats_json() : runtime.fingerprint_json();
                    result.assign(json.begin(), json.end());
                    response.kind = static_cast<uint32_t>(response_kind::json);
                    break;
                }
                case operation::reset:
                    if (!payload.empty()) throw std::runtime_error("reset payload must be empty");
                    runtime.reset();
                    response.kind = static_cast<uint32_t>(response_kind::empty);
                    break;
                case operation::ping: {
                    if (!payload.empty()) throw std::runtime_error("ping payload must be empty");
                    response.kind = static_cast<uint32_t>(response_kind::json);
                    const std::string json =
                        "{\"protocol\":\"e027-stage-v1\",\"stage_start\":" +
                        std::to_string(settings.stage_start) + ",\"stage_end\":" +
                        std::to_string(settings.stage_end) + ",\"model_layers\":" +
                        std::to_string(runtime.n_layers()) + ",\"n_embd\":" +
                        std::to_string(runtime.n_embd()) + ",\"final_stage\":" +
                        (runtime.final_stage() ? "true" : "false") +
                        ",\"load_ns\":" + std::to_string(runtime.load_ns()) +
                        ",\"request_count\":" + std::to_string(runtime.request_count()) + "}";
                    result.assign(json.begin(), json.end());
                    break;
                }
                case operation::tokenize:
                    response.kind = static_cast<uint32_t>(response_kind::tokens);
                    result = runtime.tokenize(payload, request.flags);
                    response.n_tokens = static_cast<uint32_t>(result.size() / sizeof(llama_token));
                    break;
                case operation::shutdown:
                    if (!payload.empty()) throw std::runtime_error("shutdown payload must be empty");
                    response.kind = static_cast<uint32_t>(response_kind::empty);
                    shutdown_requested = true;
                    break;
                default:
                    throw std::runtime_error("unknown E027 operation");
            }
        } catch (const std::exception & error) {
            response.status = 1;
            response.kind = static_cast<uint32_t>(response_kind::json);
            const std::string message = error.what();
            result.assign(message.begin(), message.end());
        }
        response.total_ns = elapsed_ns(total_begin, clock_type::now());
        const uint64_t accounted_ns = response.deserialize_ns + response.compute_ns;
        response.serialize_ns = response.total_ns > accounted_ns
            ? response.total_ns - accounted_ns : 0;
        send_response(client, response, result);
    }
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    try {
#ifdef _WIN32
        WSADATA data{};
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
            throw std::runtime_error("WSAStartup failed");
        }
#endif
        const options settings = parse_options(argc, argv);
        ggml_backend_load_all();
        llama_backend_init();
        stage_runtime runtime(settings);
        socket_owner server = listen_socket(settings.listen_host, settings.listen_port);
        std::fprintf(stderr,
            "E027_READY protocol=e027-stage-v1 host=%s port=%u stage=[%d,%d) "
            "layers=%d embd=%d final=%d load_ms=%.3f\n",
            settings.listen_host.c_str(), settings.listen_port, settings.stage_start,
            settings.stage_end, runtime.n_layers(), runtime.n_embd(),
            runtime.final_stage() ? 1 : 0, runtime.load_ns() / 1.0e6);
        std::fflush(stderr);

        bool shutdown_requested = false;
        while (!shutdown_requested) {
            sockaddr_in peer{};
#ifdef _WIN32
            int peer_size = sizeof(peer);
#else
            socklen_t peer_size = sizeof(peer);
#endif
            socket_owner client(accept(server.get(),
                reinterpret_cast<sockaddr *>(&peer), &peer_size));
            if (client.get() == invalid_socket_handle) {
                continue;
            }
            try {
                serve_connection(client.get(), runtime, settings, shutdown_requested);
            } catch (const std::exception & error) {
                std::fprintf(stderr, "E027_CONNECTION_ERROR %s\n", error.what());
                std::fflush(stderr);
            }
        }
#ifdef _WIN32
        WSACleanup();
#endif
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "E027_FATAL %s\n", error.what());
        return 1;
    }
}
