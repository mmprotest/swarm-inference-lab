"""Generic compound rollback + cached prefix replay upgrade for the E028 service."""


def upgrade(source: str) -> str:
    def change(old, new):
        nonlocal source
        if old not in source:
            raise RuntimeError(f"native upgrade source drift: {old[:80]}")
        source = source.replace(old, new)

    change('#include <numeric>', '#include <numeric>\n#include <map>')
    change('        n_layers_ = llama_model_n_layer(owner_.model);', '''        n_layers_ = llama_model_n_layer(owner_.model);
        const auto rope_type = llama_model_rope_type(owner_.model);
        n_pos_ = rope_type == LLAMA_ROPE_TYPE_MROPE || rope_type == LLAMA_ROPE_TYPE_IMROPE ? 4 : 1;''')
    change('hidden_positions.resize(static_cast<size_t>(request.n_tokens) * 4);',
           'hidden_positions.resize(static_cast<size_t>(request.n_tokens) * n_pos_);')
    change('''                batch.pos[index] = position;
                batch.pos[request.n_tokens + index] = position;
                batch.pos[2 * request.n_tokens + index] = position;
                batch.pos[3 * request.n_tokens + index] = 0;''', '''                // Generic llama text-position layout, selected by public RoPE metadata.
                for (uint32_t dim = 0; dim < n_pos_; ++dim)
                    batch.pos[dim * request.n_tokens + index] = dim < 3 ? position : 0;''')
    change('        if (request.flags & (1U << 8)) checkpoint_at(request.position);', '''        if (request.flags & (1U << 8)) {
            checkpoint_at(request.position);
            input_cache_[request.position] = {request, payload};
        }''')
    change('        current_position_ = 0;', '        current_position_ = 0;\n        input_cache_.clear();')
    change('    int32_t n_embd_ = 0;', '''    int32_t n_embd_ = 0;
    uint32_t n_pos_ = 1;
    struct cached_input { request_header request; std::vector<uint8_t> payload; };
    std::map<int64_t, cached_input> input_cache_;''')
    change('''        llama_memory_seq_rm(mem, slot, -1, -1);
        llama_memory_seq_cp(mem, 0, slot, -1, -1);''', '''        input_cache_.erase(checkpoint_positions_[slot]);
        llama_memory_seq_rm(mem, slot, -1, -1);
        llama_memory_seq_cp(mem, 0, slot, -1, -1);''')
    change('''                llama_memory_seq_rm(mem, i, -1, -1);
                checkpoint_positions_[i] = -1;''', '''                llama_memory_seq_rm(mem, i, -1, -1);
                input_cache_.erase(checkpoint_positions_[i]);
                checkpoint_positions_[i] = -1;''')
    change('''                case operation::rollback:
                    runtime.rollback_to(request.position);
                    response.kind = static_cast<uint32_t>(response_kind::empty);
                    break;''', '''                case operation::rollback:
                    result = runtime.rollback_request(request, response);
                    break;''')
    change('    std::string stats_json() {', r'''    std::vector<uint8_t> rollback_request(const request_header & request, response_header & response) {
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
    std::string stats_json() {''')
    # Fixed diagnostic taps were an unused legacy debugging feature. They are
    # not exposed by E028 and do not belong in the generic stage service.
    start = source.index('        if ((request.flags & return_tap22) != 0) {')
    end = source.index('        free_batch();\n        return output;', start)
    source = source[:start] + source[end:]
    return source
