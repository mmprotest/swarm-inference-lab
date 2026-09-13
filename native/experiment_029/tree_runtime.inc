// Generic branch adapter. Model mathematics and DFlash feature semantics stay in llama.cpp.
// A tree request is one serialized activation bundle per stage. The canonical
// verifier executes scalar nodes and charges state restoration and commit replay.
// There is no overlap between unresolved tree rounds.
class stage_runtime {
    std::unique_ptr<stage_runtime> draft_;
    options settings_;
    llama_owner owner_;
    const llama_vocab * vocab_;
    int32_t width_=0, layers_=0, npos_=1, selector_=0, block_=8;
    int64_t position_=0;
    uint64_t load_=0, calls_=0;
    bool final_=false;
    std::vector<int32_t> taps_, parents_, depths_;
    std::vector<float> features_;
    std::string last_stats_="{}";
    static std::string ints(const std::vector<int32_t>& v) {
        std::string s="[";for(size_t i=0;i<v.size();++i){if(i)s+=",";s+=std::to_string(v[i]);}return s+"]";
    }
    std::vector<uint8_t> pack(const std::string& meta,const std::vector<uint8_t>& data={}) {
        std::vector<uint8_t> r; append_vector(r,std::vector<uint32_t>{uint32_t(meta.size())});
        r.insert(r.end(),meta.begin(),meta.end());r.insert(r.end(),data.begin(),data.end());return r;
    }
    void positions(llama_batch& b,std::vector<llama_pos>& p,const std::vector<int32_t>& values,bool embedding) {
        p.resize(values.size()*(embedding?npos_:1));
        for(size_t i=0;i<values.size();++i)for(int d=0;d<(embedding?npos_:1);++d)p[d*values.size()+i]=d==3?0:values[i];
        b.pos=p.data();
    }
    std::vector<uint8_t> decode(const std::vector<int32_t>& ids,const std::vector<float>& emb,
            const std::vector<int32_t>& seq,const std::vector<int32_t>& pos,int emb_width,
            std::vector<float>& captured,uint64_t& ns,bool draft_noise=false) {
        const int n=int(seq.size()); const bool embedding=!emb.empty();
        llama_batch b=llama_batch_init(n,embedding?emb_width:0,1); auto allocated=b.pos;
        std::vector<llama_pos> p;positions(b,p,pos,embedding);b.n_tokens=n;
        if(embedding)std::memcpy(b.embd,emb.data(),emb.size()*4);else std::memcpy(b.token,ids.data(),n*4);
        for(int i=0;i<n;++i){b.n_seq_id[i]=1;b.seq_id[i][0]=seq[i];b.logits[i]=!settings_.dflash;}
        auto begin=clock_type::now();int rc=llama_decode(owner_.context,b);llama_synchronize(owner_.context);
        ns=elapsed_ns(begin,clock_type::now());++calls_;
        b.pos=allocated;llama_batch_free(b);
        if(rc)throw std::runtime_error("llama_decode rc="+std::to_string(rc));
        std::vector<uint8_t> out;
        if(settings_.dflash){
            if(draft_noise){const float* x=llama_get_embeddings_nextn(owner_.context);if(!x)throw std::runtime_error("draft lattice unavailable");
                append_vector(out,std::vector<float>(x,x+size_t(n)*width_));}
            return out;
        }
        captured.assign(size_t(n)*taps_.size()*width_,0);
        for(size_t j=0;j<taps_.size();++j){
            const float* x=llama_get_embeddings_layer_inp(owner_.context,taps_[j]);
            if(!x)throw std::runtime_error("requested generic feature unavailable: "+std::to_string(taps_[j]));
            for(int i=0;i<n;++i)std::memcpy(captured.data()+(size_t(i)*taps_.size()+j)*width_,x+size_t(i)*width_,width_*4);
        }
        if(final_){std::vector<int32_t> greedy(n);
            for(int i=0;i<n;++i){const float* x=llama_get_logits_ith(owner_.context,i);if(!x)throw std::runtime_error("logits unavailable");
                greedy[i]=int32_t(std::max_element(x,x+llama_vocab_n_tokens(vocab_))-x);}
            append_vector(out,greedy);
        }else{
            const float* x=llama_get_embeddings_layer_inp(owner_.context,settings_.stage_end);
            if(!x)throw std::runtime_error("stage activation unavailable");
            append_vector(out,std::vector<float>(x,x+size_t(n)*width_));
        }return out;
    }
#include "depth_checkpoint.inc"
public:
    explicit stage_runtime(options settings, llama_context* other=nullptr):settings_(std::move(settings)){
        auto start=clock_type::now();
        if(!settings_.dflash)set_stage_environment(settings_.stage_start,settings_.stage_end);
        else { _putenv_s("E027_STAGE_START", ""); _putenv_s("E027_STAGE_END", ""); }
        auto mp=llama_model_default_params();mp.n_gpu_layers=settings_.gpu_layers;
        owner_.model=llama_model_load_from_file(settings_.model_path.c_str(),mp);
        if(!owner_.model)throw std::runtime_error("model load failed");
        width_=llama_model_n_embd(owner_.model);layers_=llama_model_n_layer(owner_.model);vocab_=llama_model_get_vocab(owner_.model);
        auto rope=llama_model_rope_type(owner_.model);npos_=(rope==LLAMA_ROPE_TYPE_MROPE||rope==LLAMA_ROPE_TYPE_IMROPE)?4:1;
        final_=!settings_.dflash && settings_.stage_end==layers_;
        auto cp=llama_context_default_params();cp.ctx_other=other;cp.n_ctx=settings_.n_ctx;cp.n_batch=settings_.n_batch;cp.n_ubatch=settings_.n_ubatch;
        
        cp.n_seq_max=settings_.dflash?1:10;cp.n_rs_seq=0;cp.kv_unified=true;cp.no_perf=false;
        owner_.context=llama_init_from_model(owner_.model,cp);if(!owner_.context)throw std::runtime_error("context load failed");
        if(settings_.dflash){
            selector_=llama_model_dflash_selector_top_k(owner_.model);
            const auto* t=llama_model_target_layer_ids(owner_.model);taps_.assign(t,t+llama_model_target_layer_ids_n(owner_.model));
            char v[64];if(llama_model_meta_val_str(owner_.model,"dflash.block_size",v,64)>0)block_=std::stoi(v);
            llama_set_embeddings_nextn(owner_.context,true,false);llama_set_causal_attn(owner_.context,false);
        }else if(!final_)llama_set_embeddings_layer_inp(owner_.context,settings_.stage_end,true);
        load_=elapsed_ns(start,clock_type::now());
        if(!settings_.draft_path.empty()){auto ds=settings_;ds.dflash=true;ds.model_path=ds.draft_path;ds.draft_path.clear();draft_=std::make_unique<stage_runtime>(ds,owner_.context);set_stage_environment(settings_.stage_start,settings_.stage_end);}
    }
    ~stage_runtime(){draft_.reset();}
    int32_t n_embd()const{return width_;}int32_t n_layers()const{return layers_;}bool final_stage()const{return final_;}
    uint64_t load_ns()const{return load_;}uint64_t request_count()const{return calls_;}
    void reset(){linear_pending_=false;linear_states_.clear();if(draft_)draft_->reset();llama_memory_clear(llama_get_memory(owner_.context),true);position_=0;parents_.clear();depths_.clear();features_.clear();}
    std::vector<uint8_t> tokenize(const std::vector<uint8_t>& text,uint32_t flags)const{
        std::vector<llama_token> t(text.size()+32);int n=llama_tokenize(vocab_,(const char*)text.data(),int(text.size()),t.data(),int(t.size()),flags&add_special_tokens,flags&parse_special_tokens);
        if(n<0){t.resize(-n);n=llama_tokenize(vocab_,(const char*)text.data(),int(text.size()),t.data(),int(t.size()),flags&add_special_tokens,flags&parse_special_tokens);}if(n<0)throw std::runtime_error("tokenize failed");t.resize(n);std::vector<uint8_t> o;append_vector(o,t);return o;
    }
    std::string stats_json(){if(draft_)return draft_->stats_json();return "{\"position\":"+std::to_string(position_)+",\"selector_top_k\":"+std::to_string(selector_)+",\"block_size\":"+std::to_string(block_)+",\"target_layers\":"+ints(taps_)+",\"last\":"+last_stats_+"}";}
    std::string fingerprint_json(int seq=0){if(seq==0 && linear_pending_)seq=9;

        llama_synchronize(owner_.context);auto hash=[&](llama_state_seq_flags flags){
            size_t n=llama_state_seq_get_size_ext(owner_.context,seq,flags);std::vector<uint8_t> d(n);
            if(llama_state_seq_get_data_ext(owner_.context,d.data(),n,seq,flags)!=n)throw std::runtime_error("state export failed");
            uint64_t h=14695981039346656037ULL;for(auto b:d)h=(h^b)*1099511628211ULL;return std::to_string(h);};
        return "{\"position\":"+std::to_string(position_)+",\"memory_max_position\":"+std::to_string(llama_memory_seq_pos_max(llama_get_memory(owner_.context),seq))+",\"full_state_fnv64\":\""+hash(LLAMA_STATE_SEQ_FLAGS_NONE)+"\",\"partial_state_fnv64\":\""+hash(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)+"\"}";
    }
    std::vector<uint8_t> handle(const request_header& r,const std::vector<uint8_t>& payload,response_header& response){
        if(draft_ && (r.op==14||r.op==15))return draft_->handle(r,payload,response);
        response.kind=uint32_t(response_kind::hidden);response.n_tokens=r.n_tokens;response.n_embd=width_;
        auto mem=llama_get_memory(owner_.context);
        if(r.op==13){taps_.resize(payload.size()/4);std::memcpy(taps_.data(),payload.data(),payload.size());
            for(auto t:taps_)llama_set_embeddings_layer_inp(owner_.context,t,true);return pack("{}");}
        if(r.op==11)return tree_forward(r,payload,response);
        if(r.op==12)return tree_commit(r,response);
        if(r.op==14||r.op==15){ // Drafter adapter; no tree/scheduling decisions here.
            auto restore_start=clock_type::now();llama_memory_seq_rm(mem,0,r.position,-1);llama_synchronize(owner_.context);const auto restore_ns=elapsed_ns(restore_start,clock_type::now());
            std::vector<int32_t> ids,seq(r.n_tokens,0),pos(r.n_tokens);std::iota(pos.begin(),pos.end(),int(r.position));
            std::vector<float> e,capture;
            if(r.op==14){if(payload.size()!=size_t(r.n_tokens)*r.n_embd*4)throw std::runtime_error("bad draft features");e.resize(payload.size()/4);std::memcpy(e.data(),payload.data(),payload.size());}
            else{ids.assign(r.n_tokens,llama_vocab_mask(vocab_));ids[0]=r.arg;}
            auto out=decode(ids,e,seq,pos,r.n_embd,capture,response.compute_ns,r.op==15);response.compute_ns+=restore_ns;
            return pack("{\"state_restore_ns\":"+std::to_string(restore_ns)+",\"selector_top_k\":"+std::to_string(selector_)+",\"width\":"+std::to_string(width_)+",\"taps\":"+ints(taps_)+"}",out);
        }
        const bool tree=r.op==11,tokens=(r.flags&input_tokens)!=0;const size_t n=r.n_tokens;
        if(r.position!=position_)throw std::runtime_error("root position mismatch");
        if(tree && (!parents_.empty()||n>=settings_.checkpoint_slots))throw std::runtime_error("uncommitted tree or node budget exceeded");
        const size_t offset=tree?n*8:0, stride=tokens?4:size_t(width_)*4;
        if(payload.size()!=offset+n*stride)throw std::runtime_error("invalid node payload");
        std::vector<int32_t> par(n,-1),dep(n);if(tree){std::memcpy(par.data(),payload.data(),n*4);std::memcpy(dep.data(),payload.data()+n*4,n*4);}
        else std::iota(dep.begin(),dep.end(),1);
        if(tree)for(size_t i=0;i<n;++i)if(par[i]>=int(i)||par[i]<-1||dep[i]!=(par[i]<0?1:dep[par[i]]+1))throw std::runtime_error("non-topological tree");
        if(tree)llama_memory_seq_cp(mem,0,settings_.checkpoint_slots,-1,-1);
        std::vector<uint8_t> outputs(n*(final_?4:size_t(width_)*4));
        std::vector<float> all_features(n*taps_.size()*width_);std::string levels="[";uint64_t total=0;
        const int maxd=tree?int(n):1; // Stable scalar kernels across every branch/history.
        for(int d=1;d<=maxd;++d){
            std::vector<int32_t> selected,seq,pos,ids;std::vector<float> emb,captured;
            auto forkstart=clock_type::now();
            for(size_t i=0;i<n;++i)if(!tree||int(i)==d-1){
                selected.push_back(int(i));int s=0;seq.push_back(s);pos.push_back(int(r.position)+(tree?dep[i]-1:int(i)));
                if(tree){llama_memory_seq_rm(mem,0,-1,-1);llama_memory_seq_cp(mem,par[i]<0?settings_.checkpoint_slots:par[i]+1,0,-1,-1);}
                if(tokens){int32_t v;std::memcpy(&v,payload.data()+offset+i*stride,4);ids.push_back(v);}
                else {auto start=emb.size();emb.resize(start+width_);std::memcpy(emb.data()+start,payload.data()+offset+i*stride,stride);}
            }
            llama_synchronize(owner_.context);uint64_t fork_ns=elapsed_ns(forkstart,clock_type::now()), ns=0;
            auto out=decode(ids,emb,seq,pos,width_,captured,ns);
            if(tree){auto checkpoint_start=clock_type::now();llama_memory_seq_cp(mem,0,selected[0]+1,-1,-1);llama_synchronize(owner_.context);fork_ns+=elapsed_ns(checkpoint_start,clock_type::now());}
            for(size_t j=0;j<selected.size();++j){auto i=selected[j];size_t outstride=final_?4:size_t(width_)*4;
                std::memcpy(outputs.data()+i*outstride,out.data()+j*outstride,outstride);
                if(!taps_.empty())std::memcpy(all_features.data()+size_t(i)*taps_.size()*width_,captured.data()+j*taps_.size()*width_,taps_.size()*width_*4);
            }
            if(d>1)levels+=",";levels+="{\"depth\":"+std::to_string(tree?dep[selected[0]]:0)+",\"nodes\":"+ints(selected)+",\"compute_ns\":"+std::to_string(ns)+",\"fork_ns\":"+std::to_string(fork_ns)+"}";
            total+=ns+fork_ns;
        }
        levels+="]";response.compute_ns=total;
        if(tree){parents_=par;depths_=dep;features_=std::move(all_features);}else{position_+=n;append_vector(outputs,all_features);}
        last_stats_="{\"levels\":"+levels+",\"taps\":"+ints(taps_)+",\"final\":"+(final_?"true":"false")+",\"physical_decode_calls\":"+std::to_string(maxd)+"}";
        return pack(last_stats_,outputs);
    }
};
