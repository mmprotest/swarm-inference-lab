#include "ggml-backend.h"
#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <memory>
#include <functional>
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
    bool mtp = false; bool dflash = false; std::string draft_path;
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
        } else if (arg == "--dflash") {
            result.dflash = true;
        } else if (arg == "--draft-model") {
            result.draft_path = next();
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

#include "tree_runtime.inc"

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
                case static_cast<operation>(11):
                case static_cast<operation>(12):
                case static_cast<operation>(13):
                case static_cast<operation>(14):
                case static_cast<operation>(15):
                    result = runtime.handle(request, payload, response);
                    break;
                case operation::stats:
                case operation::fingerprint: {
                    auto json = request.op == static_cast<uint16_t>(operation::stats)
                        ? runtime.stats_json() : runtime.fingerprint_json(request.arg);
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
