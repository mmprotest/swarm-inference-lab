#include "llama.h"
#include "../../.runtime/e026-llama.cpp/src/llama-ext.h"
#include "ggml-rpc.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using clock_type = std::chrono::steady_clock;

static double elapsed(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

static void log_stderr(ggml_log_level, const char * text, void *) {
    std::fputs(text, stderr);
}

static json read_json(const std::string & path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open JSON input");
    return json::parse(in);
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    llama_log_set(log_stderr, nullptr);
    ggml_log_set(log_stderr, nullptr);
    llama_backend_init();
    ggml_backend_load_all();
    llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    try {
        const json cfg = read_json(argv[1]);
        for (const auto & endpoint : cfg.value("rpc", std::vector<std::string>{})) {
            if (!ggml_backend_rpc_add_server(endpoint.c_str())) {
                throw std::runtime_error("Cannot register RPC backend");
            }
        }
        std::vector<ggml_backend_dev_t> devices;
        for (const auto & name : cfg.value("devices", std::vector<std::string>{})) {
            auto * dev = ggml_backend_dev_by_name(name.c_str());
            if (!dev) throw std::runtime_error("Requested device unavailable");
            devices.push_back(dev);
        }
        devices.push_back(nullptr);
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = cfg.value("n_gpu_layers", 99);
        mp.split_mode = LLAMA_SPLIT_MODE_LAYER;
        if (devices.size() > 1) mp.devices = devices.data();
        std::vector<float> splits = cfg.value("tensor_split", std::vector<float>{});
        if (!splits.empty()) {
            splits.resize(llama_max_devices(), 0.0f);
            mp.tensor_split = splits.data();
        }
        const auto load_start = clock_type::now();
        model = llama_model_load_from_file(cfg.at("model").get<std::string>().c_str(), mp);
        if (!model) throw std::runtime_error("Model load failed");
        const auto model_seconds = elapsed(load_start);
        auto cp = llama_context_default_params();
        cp.n_ctx = cfg.value("n_ctx", 12288);
        cp.n_batch = cfg.value("n_batch", 512);
        cp.n_ubatch = cfg.value("n_ubatch", 512);
        cp.n_seq_max = 1;
        cp.n_rs_seq = cfg.value("recurrent_snapshots", 0);
        cp.n_threads = 8;
        cp.n_threads_batch = 8;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.no_perf = false;
        ctx = llama_init_from_model(model, cp);
        if (!ctx) throw std::runtime_error("Context creation failed");
        if (cfg.value("embeddings_nextn", false)) llama_set_embeddings_nextn(ctx, true, false);
        const auto * vocab = llama_model_get_vocab(model);
        const int nv = llama_vocab_n_tokens(vocab);
        int position = 0;
        json ready = {{"event", "READY"}, {"model_load_s", model_seconds},
                      {"total_load_s", elapsed(load_start)}, {"vocab_size", nv},
                      {"n_layers", llama_model_n_layer(model)}};
        std::cout << ready.dump() << std::endl;

        auto evaluate = [&](std::vector<llama_token> & tokens, bool all_logits) {
            const auto start = clock_type::now();
            for (size_t offset = 0; offset < tokens.size(); offset += cp.n_batch) {
                const auto n = (int) std::min((size_t) cp.n_batch, tokens.size() - offset);
                auto batch = llama_batch_init(n, 0, 1);
                batch.n_tokens = n;
                for (int i = 0; i < n; ++i) {
                    batch.token[i] = tokens[offset + i];
                    batch.pos[i] = position + i;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = all_logits || i == n - 1;
                }
                const int ret = llama_decode(ctx, batch);
                llama_batch_free(batch);
                if (ret != 0) throw std::runtime_error("llama_decode failed: " + std::to_string(ret));
                position += n;
            }
            llama_synchronize(ctx);
            return elapsed(start);
        };

        std::string line;
        while (std::getline(std::cin, line)) {
            const auto start = clock_type::now();
            const auto req = json::parse(line);
            const auto op = req.at("op").get<std::string>();
            json result = {{"op", op}};
            if (op == "close") break;
            if (op == "clear") {
                llama_memory_clear(llama_get_memory(ctx), true);
                position = 0;
            } else if (op == "decode") {
                auto tokens = req.at("tokens").get<std::vector<llama_token>>();
                if (tokens.empty()) throw std::runtime_error("Empty decode");
                result["compute_s"] = evaluate(tokens, req.value("all_logits", false));
                const float * logits = llama_get_logits_ith(ctx, -1);
                const int greedy = std::max_element(logits, logits + nv) - logits;
                result["greedy_token"] = greedy;
                std::vector<int> indices(nv);
                std::iota(indices.begin(), indices.end(), 0);
                const int k = std::min(nv, req.value("top_k", 10));
                std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                                  [&](int a, int b) { return logits[a] > logits[b]; });
                json top = json::array();
                for (int i = 0; i < k; ++i) top.push_back({{"id", indices[i]}, {"logit", logits[indices[i]]}});
                result["top_logits"] = top;
                if (req.contains("logits_file")) {
                    std::ofstream out(req.at("logits_file").get<std::string>(), std::ios::binary);
                    out.write(reinterpret_cast<const char *>(logits), nv * sizeof(float));
                    if (!out) throw std::runtime_error("Logit write failed");
                }
            } else if (op == "save") {
                const auto full = llama_state_seq_get_size(ctx, 0);
                const auto partial = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                if (full == 0 || partial == 0 || partial > full) throw std::runtime_error("Invalid hybrid state sizes");
                std::vector<uint8_t> data(full);
                const auto serial_start = clock_type::now();
                const auto copied = llama_state_seq_get_data(ctx, data.data(), data.size(), 0);
                result["serialization_s"] = elapsed(serial_start);
                if (copied != full) throw std::runtime_error("Incomplete state export");
                std::ofstream out(req.at("path").get<std::string>(), std::ios::binary);
                out.write(reinterpret_cast<const char *>(data.data()), data.size());
                if (!out) throw std::runtime_error("State file write failed");
                result["state_bytes"] = full;
                result["recurrent_state_with_header_bytes"] = partial;
                result["attention_state_bytes"] = full - partial;
                result["flags"] = LLAMA_STATE_SEQ_FLAGS_NONE;
            } else if (op == "restore") {
                std::ifstream in(req.at("path").get<std::string>(), std::ios::binary | std::ios::ate);
                if (!in) throw std::runtime_error("State file unavailable");
                const auto size = in.tellg();
                if (size <= 0) throw std::runtime_error("Empty state file");
                std::vector<uint8_t> data((size_t) size);
                in.seekg(0);
                in.read(reinterpret_cast<char *>(data.data()), data.size());
                if (!in) throw std::runtime_error("State file read failed");
                llama_memory_clear(llama_get_memory(ctx), true);
                const auto restore_start = clock_type::now();
                const auto restored = llama_state_seq_set_data(ctx, data.data(), data.size(), 0);
                llama_synchronize(ctx);
                result["restore_s"] = elapsed(restore_start);
                result["restored_bytes"] = restored;
                if (restored != data.size()) throw std::runtime_error("Incomplete state restore");
                position = req.at("position").get<int>();
                if (llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != position - 1) {
                    throw std::runtime_error("Restored state position mismatch");
                }
            } else {
                throw std::runtime_error("Unknown operation");
            }
            result["position"] = position;
            result["wall_s"] = elapsed(start);
            std::cout << result.dump() << std::endl;
        }
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "E026 probe: " << error.what() << std::endl;
        std::cout << json({{"error", error.what()}}).dump() << std::endl;
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
        return 1;
    }
}
