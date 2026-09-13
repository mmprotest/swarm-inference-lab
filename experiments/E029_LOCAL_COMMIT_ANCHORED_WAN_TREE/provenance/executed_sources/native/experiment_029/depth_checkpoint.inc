// Canonical single-sequence tree walk. Only backend-defined partial state is
// checkpointed (recurrent state for a hybrid); KV tails are truncated on return
// to an ancestor. At most one checkpoint per depth is live, not one context per
// node. Every node call, checkpoint, restore and accepted-path replay is charged.
std::vector<uint8_t> tree_input_;
bool tree_tokens_=false;
bool linear_pending_=false;
std::vector<std::vector<uint8_t>> linear_states_;
std::vector<uint8_t> partial_checkpoint(int slot){
    auto mem=llama_get_memory(owner_.context);llama_memory_seq_cp(mem,0,slot,-1,-1);
    auto flags=LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;
    size_t size=llama_state_seq_get_size_ext(owner_.context,slot,flags);std::vector<uint8_t> data(size);
    if(llama_state_seq_get_data_ext(owner_.context,data.data(),size,slot,flags)!=size)throw std::runtime_error("partial checkpoint failed");llama_memory_seq_rm(mem,slot,-1,-1);llama_synchronize(owner_.context);return data;
}
void partial_restore(const std::vector<uint8_t>& data,int64_t position){
    if(llama_state_seq_set_data_ext(owner_.context,data.data(),data.size(),0,LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE)!=data.size())throw std::runtime_error("partial restore failed");
    // Restore state position first. The generic hybrid API can then remove the
    // KV future without requesting an unsupported recurrent partial deletion.
    if(!llama_memory_seq_rm(llama_get_memory(owner_.context),0,position,-1))throw std::runtime_error("KV tail truncate after partial restore failed");
    llama_synchronize(owner_.context);
}
std::vector<uint8_t> tree_node(int i,std::vector<float>& features,uint64_t& ns){
    const size_t n=parents_.size(),stride=tree_tokens_?4:size_t(width_)*4;std::vector<int32_t> ids;std::vector<float> emb;
    if(tree_tokens_){int32_t token;std::memcpy(&token,tree_input_.data()+n*8+i*stride,4);ids.push_back(token);}
    else{emb.resize(width_);std::memcpy(emb.data(),tree_input_.data()+n*8+i*stride,stride);}
    return decode(ids,emb,{0},{int32_t(position_+depths_[i]-1)},width_,features,ns);
}
std::vector<uint8_t> tree_forward(const request_header& r,const std::vector<uint8_t>& payload,response_header& response){
    const size_t n=r.n_tokens;if(!n||n>settings_.checkpoint_slots||!parents_.empty()||r.position!=position_)throw std::runtime_error("invalid or overlapping tree round");
    tree_tokens_=(r.flags&input_tokens)!=0;const size_t stride=tree_tokens_?4:size_t(width_)*4;
    if(payload.size()!=n*(8+stride))throw std::runtime_error("invalid tree bundle size");
    parents_.resize(n);depths_.resize(n);std::memcpy(parents_.data(),payload.data(),n*4);std::memcpy(depths_.data(),payload.data()+n*4,n*4);tree_input_=payload;
    std::vector<std::vector<int>> children(n+1);
    for(size_t i=0;i<n;++i){if(parents_[i]<-1||parents_[i]>=int(i)||depths_[i]!=(parents_[i]<0?1:depths_[parents_[i]]+1))throw std::runtime_error("invalid tree topology");children[parents_[i]+1].push_back(int(i));}
    bool linear=children[0].size()==1;
    for(const auto& c:children)if(c.size()>1)linear=false;
    if(linear){
        linear_pending_=true;linear_states_.clear();auto mem=llama_get_memory(owner_.context);
        auto checkpoint_begin=clock_type::now();llama_memory_seq_cp(mem,0,9,-1,-1);llama_synchronize(owner_.context);
        uint64_t total=elapsed_ns(checkpoint_begin,clock_type::now()),all_checkpoints=total;
        std::vector<uint8_t> output;features_.clear();std::string levels="[";
        for(size_t i=0;i<n;++i){std::vector<float> captured;uint64_t compute=0;
            auto data=tree_node(int(i),captured,compute);output.insert(output.end(),data.begin(),data.end());features_.insert(features_.end(),captured.begin(),captured.end());
            auto start=clock_type::now();llama_memory_seq_cp(mem,0,int(i)+1,-1,-1);llama_synchronize(owner_.context);uint64_t checkpoint=elapsed_ns(start,clock_type::now());
            total+=compute+checkpoint;all_checkpoints+=checkpoint;if(i)levels+=",";
            levels+="{\"depth\":"+std::to_string(depths_[i])+",\"nodes\":["+std::to_string(i)+"],\"compute_ns\":"+std::to_string(compute)+",\"fork_ns\":"+std::to_string(checkpoint)+"}";
        }
        levels+="]";response.compute_ns=total;response.n_tokens=uint32_t(n);response.n_embd=width_;response.kind=uint32_t(response_kind::hidden);
        last_stats_="{\"levels\":"+levels+",\"taps\":"+ints(taps_)+",\"final\":"+(final_?"true":"false")+",\"physical_decode_calls\":"+std::to_string(n)+",\"state_checkpoint_ns\":"+std::to_string(all_checkpoints)+",\"state_restore_ns\":0,\"linear_checkpoint_mode\":true}";
        return pack(last_stats_,output);
    }
    linear_pending_=false;
    const size_t outstride=final_?4:size_t(width_)*4;std::vector<uint8_t> output(n*outstride);
    features_.resize(n*taps_.size()*width_);std::vector<uint64_t> compute(n,0),state(n,0);uint64_t checkpoint_ns=0,restore_ns=0;size_t max_checkpoint_bytes=0,live_bytes=0;
    auto save=[&](int slot){auto t=clock_type::now();auto data=partial_checkpoint(slot);auto elapsed=elapsed_ns(t,clock_type::now());checkpoint_ns+=elapsed;return std::make_pair(std::move(data),elapsed);};
    auto restore=[&](const std::vector<uint8_t>& data,int64_t p){auto t=clock_type::now();partial_restore(data,p);auto elapsed=elapsed_ns(t,clock_type::now());restore_ns+=elapsed;return elapsed;};
    std::function<void(int)> visit=[&](int i){
        std::vector<float> captured;auto data=tree_node(i,captured,compute[i]);std::memcpy(output.data()+size_t(i)*outstride,data.data(),outstride);
        if(!taps_.empty())std::memcpy(features_.data()+size_t(i)*taps_.size()*width_,captured.data(),captured.size()*4);
        if(children[i+1].size()==1){visit(children[i+1][0]);}
        else if(!children[i+1].empty()){
            auto saved=save(depths_[i]+1);live_bytes+=saved.first.size();max_checkpoint_bytes=std::max(max_checkpoint_bytes,live_bytes);
            const auto count=children[i+1].size();for(size_t j=0;j<count;++j){int child=children[i+1][j];state[child]+=saved.second/count+(j==0?saved.second%count:0);visit(child);state[child]+=restore(saved.first,position_+depths_[i]);}
            live_bytes-=saved.first.size();
        }
    };
    auto root=save(1);live_bytes=root.first.size();max_checkpoint_bytes=live_bytes;
    for(size_t j=0;j<children[0].size();++j){int child=children[0][j];state[child]+=root.second/children[0].size()+(j==0?root.second%children[0].size():0);visit(child);state[child]+=restore(root.first,position_);}
    std::string levels="[";uint64_t total=0;
    for(size_t i=0;i<n;++i){if(i)levels+=",";levels+="{\"depth\":"+std::to_string(depths_[i])+",\"nodes\":["+std::to_string(i)+"],\"compute_ns\":"+std::to_string(compute[i])+",\"fork_ns\":"+std::to_string(state[i])+"}";total+=compute[i]+state[i];}
    levels+="]";response.compute_ns=total;response.n_tokens=uint32_t(n);response.n_embd=width_;response.kind=uint32_t(response_kind::hidden);
    last_stats_="{\"levels\":"+levels+",\"taps\":"+ints(taps_)+",\"final\":"+(final_?"true":"false")+",\"physical_decode_calls\":"+std::to_string(n)+",\"state_checkpoint_ns\":"+std::to_string(checkpoint_ns)+",\"state_restore_ns\":"+std::to_string(restore_ns)+",\"max_live_checkpoint_metadata_bytes\":"+std::to_string(max_checkpoint_bytes)+"}";
    return pack(last_stats_,output);
}
std::vector<uint8_t> tree_commit(const request_header& r,response_header& response){
    if(r.arg>parents_.size())throw std::runtime_error("invalid commit node");std::vector<int32_t> path;int node=int(r.arg)-1;
    while(node>=0){path.push_back(node);node=parents_[node];}std::reverse(path.begin(),path.end());
    if(linear_pending_){
        auto begin=clock_type::now();auto mem=llama_get_memory(owner_.context);
        llama_memory_seq_rm(mem,0,-1,-1);llama_memory_seq_cp(mem,r.arg?r.arg:9,0,-1,-1);
        for(int slot=1;slot<=9;++slot)llama_memory_seq_rm(mem,slot,-1,-1);
        llama_synchronize(owner_.context);position_+=path.size();response.compute_ns=elapsed_ns(begin,clock_type::now());
        std::vector<float> f(features_.begin(),features_.begin()+path.size()*taps_.size()*width_);std::vector<uint8_t> out;append_vector(out,f);
        auto meta="{\"position\":"+std::to_string(position_)+",\"accepted_indices\":"+ints(path)+",\"taps\":"+ints(taps_)+",\"physical_decode_calls\":0,\"accepted_replay_compute_ns\":0,\"linear_checkpoint_mode\":true}";
        parents_.clear();depths_.clear();features_.clear();tree_input_.clear();linear_states_.clear();linear_pending_=false;return pack(meta,out);
    }
    auto begin=clock_type::now();std::vector<float> committed_features;uint64_t replay_ns=0;
    // Inputs for the accepted path were computed canonically during verification,
    // so each stage can replay its cached rows locally in the commit RPC.
    for(auto i:path){std::vector<float> captured;uint64_t ns=0;tree_node(i,captured,ns);replay_ns+=ns;committed_features.insert(committed_features.end(),captured.begin(),captured.end());}
    position_+=path.size();response.compute_ns=elapsed_ns(begin,clock_type::now());
    std::vector<uint8_t> out;append_vector(out,committed_features);
    std::string meta="{\"position\":"+std::to_string(position_)+",\"accepted_indices\":"+ints(path)+",\"taps\":"+ints(taps_)+",\"physical_decode_calls\":"+std::to_string(path.size())+",\"accepted_replay_compute_ns\":"+std::to_string(replay_ns)+"}";
    parents_.clear();depths_.clear();features_.clear();tree_input_.clear();return pack(meta,out);
}
