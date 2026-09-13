"""Derive the E028 native service from the existing local stage adapter.

Only experiment-local source is written. The llama library and E027 sources are
left intact. No network access or model acquisition is performed.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "native/experiment_027/e027_stage_server.cpp").read_text()

def replace(old, new):
    global source
    if old not in source:
        raise RuntimeError(f"source drift: {old[:100]}")
    source = source.replace(old, new)

replace("    shutdown = 5,", "    shutdown = 5,\n    stats = 6,\n    checkpoint = 7,\n    commit = 8,\n    rollback = 9,\n    fingerprint = 10,")
replace("    bool serial_blocks = false;", "    bool serial_blocks = false;\n    bool profile_layers = false;\n    uint32_t checkpoint_slots = 18;")
replace('        } else if (arg == "--serial-blocks") {', '''        } else if (arg == "--checkpoint-slots") {
            result.checkpoint_slots = static_cast<uint32_t>(parse_i32(next(), "checkpoint-slots"));
        } else if (arg == "--profile-layers") {
            result.profile_layers = true;
        } else if (arg == "--serial-blocks") {''')
replace('        context_params.n_seq_max = 10;', '        context_params.n_seq_max = settings_.checkpoint_slots + 1;')
replace('        if (settings_.mtp) context_params.ctx_type', '''        checkpoint_positions_.assign(settings_.checkpoint_slots + 1, -1);
        if (settings_.profile_layers) {
            context_params.cb_eval = layer_callback;
            context_params.cb_eval_user_data = this;
        }
        if (settings_.mtp) context_params.ctx_type''')
replace('        checkpoint_positions_.fill(-1);', '        std::fill(checkpoint_positions_.begin(), checkpoint_positions_.end(), -1);\n        current_position_ = 0;')
start = source.index('        if (request.rewind_position >= 0) {')
end = source.index('        // One WAN work unit', start)
source = source[:start] + '''        if (request.rewind_position < -1) commit_to(-request.rewind_position - 2);
        if (request.rewind_position >= 0) {
            if (settings_.serial_blocks) rollback_to(request.rewind_position);
            else if (!llama_memory_seq_rm(llama_get_memory(owner_.context), 0,
                        static_cast<llama_pos>(request.rewind_position), -1))
                throw std::runtime_error("draft state rollback failed");
            current_position_ = request.rewind_position;
        }
        if (request.position != current_position_)
            throw std::runtime_error("noncontiguous sequence position");
        if (request.flags & (1U << 8)) checkpoint_at(request.position);

''' + source[end:]
replace('                one.rewind_position = -1;', '                one.rewind_position = -1;\n                one.flags &= ~(1U << 8);')
replace('settings_.serial_blocks && request.position > 0 && request.n_tokens > 1', 'settings_.serial_blocks && !(request.flags & (1U << 9)) && request.position > 0 && request.n_tokens > 1')
start = source.index('        if (settings_.serial_blocks && request.position > 0) {')
end = source.index('        llama_batch batch', start)
source = source[:start] + source[end:]
replace('        ++request_count_;', '        ++request_count_;\n        current_position_ = request.position + request.n_tokens;')
replace('        const auto compute_begin = clock_type::now();', '        const auto compute_begin = clock_type::now();\n        layer_last_ = compute_begin;')
replace('    std::array<int64_t, 10> checkpoint_positions_ = {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1};', '''    std::vector<int64_t> checkpoint_positions_;
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
        return "{\\"position\\":" + std::to_string(current_position_) +
            ",\\"memory_max_position\\":" + std::to_string(llama_memory_seq_pos_max(llama_get_memory(owner_.context), 0)) +
            ",\\"full_state_fnv64\\":\\"" + hash(LLAMA_STATE_SEQ_FLAGS_NONE) +
            "\\",\\"partial_state_fnv64\\":\\"" + hash(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) + "\\"}";
    }
    std::string stats_json() {
        std::string s = "{\\"position\\":" + std::to_string(current_position_) +
          ",\\"checkpoint_ns\\":" + std::to_string(checkpoint_ns_) +
          ",\\"checkpoint_count\\":" + std::to_string(checkpoint_count_) +
          ",\\"commit_ns\\":" + std::to_string(commit_ns_) +
          ",\\"commit_count\\":" + std::to_string(commit_count_) +
          ",\\"rollback_ns\\":" + std::to_string(rollback_ns_) +
          ",\\"rollback_count\\":" + std::to_string(rollback_count_) + ",\\"layer_samples\\":[";
        for (size_t i = 0; i < layer_samples_.size(); ++i) {
            if (i) s += ",";
            s += "[" + std::to_string(layer_samples_[i].first) + "," + std::to_string(layer_samples_[i].second) + "]";
        }
        layer_samples_.clear();
        return s + "]}";
    }''')
replace('                case operation::reset:', '''                case operation::checkpoint:
                    runtime.checkpoint_at(request.position);
                    response.kind = static_cast<uint32_t>(response_kind::empty);
                    break;
                case operation::commit:
                    runtime.commit_to(request.position);
                    response.kind = static_cast<uint32_t>(response_kind::empty);
                    break;
                case operation::rollback:
                    runtime.rollback_to(request.position);
                    response.kind = static_cast<uint32_t>(response_kind::empty);
                    break;
                case operation::stats:
                case operation::fingerprint: {
                    auto json = request.op == static_cast<uint16_t>(operation::stats)
                        ? runtime.stats_json() : runtime.fingerprint_json();
                    result.assign(json.begin(), json.end());
                    response.kind = static_cast<uint32_t>(response_kind::json);
                    break;
                }
                case operation::reset:''')
# Remove unused fixed diagnostic taps from the derived Swarm adapter.
start = source.index('            if (settings_.stage_start == 0 && !settings_.mtp) {')
end = source.index('        } else {', start)
source = source[:start] + source[end:]
dest = ROOT / "native/experiment_028"
dest.mkdir(parents=True, exist_ok=True)
(dest / "e028_stage_server.cpp").write_text(source)
print(dest / "e028_stage_server.cpp")
