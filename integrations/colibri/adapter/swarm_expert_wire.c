/*
 * Copyright 2026 swarm-inference-lab contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 * Byte-compatible C implementation of experiment_010/wire.py and
 * protocol/tensor_codec.py.  This is the sole C wire implementation copied
 * into the pinned Colibri export by integrations/colibri/build.{ps1,sh}.
 */
#include "swarm_expert_wire.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { const char *begin, *end; } json_span;

static int wire_error(char *error, size_t capacity, const char *fmt, ...) {
    if (error && capacity) {
        va_list ap; va_start(ap, fmt); vsnprintf(error, capacity, fmt, ap); va_end(ap);
    }
    return -1;
}

static uint32_t read_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | p[3];
}
static uint64_t read_be64(const uint8_t *p) {
    uint64_t value = 0; for (int i = 0; i < 8; i++) value = (value << 8) | p[i]; return value;
}
static void write_be32(uint8_t *p, uint32_t value) {
    p[0]=(uint8_t)(value>>24);p[1]=(uint8_t)(value>>16);p[2]=(uint8_t)(value>>8);p[3]=(uint8_t)value;
}
static void write_be64(uint8_t *p, uint64_t value) {
    for(int i=7;i>=0;i--){p[i]=(uint8_t)value;value>>=8;}
}
static int add_size(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) return -1;
    *out = a + b;
    return 0;
}

static const char *skip_ws(const char *p, const char *end) {
    while (p < end && isspace((unsigned char)*p)) p++;
    return p;
}
static int hex_value(char c) {
    if(c>='0'&&c<='9')return c-'0';
    if(c>='a'&&c<='f')return c-'a'+10;
    if(c>='A'&&c<='F')return c-'A'+10;
    return -1;
}

static const char *scan_json_string(const char *p, const char *end) {
    if (p >= end || *p != '"') return NULL;
    p++;
    while (p < end) {
        unsigned char c = (unsigned char)*p++;
        if (c == '"') return p;
        if (c < 0x20) return NULL;
        if (c == '\\') {
            if (p >= end) return NULL;
            char e = *p++;
            if (strchr("\"\\/bfnrt", e)) continue;
            if (e != 'u' || end - p < 4) return NULL;
            for (int i=0;i<4;i++) if(hex_value(p[i])<0)return NULL;
            p += 4;
        }
    }
    return NULL;
}

static const char *scan_json_value(const char *p, const char *end, int depth) {
    if (depth > 64) return NULL;
    p = skip_ws(p,end);
    if(p>=end)return NULL;
    if (*p == '"') return scan_json_string(p,end);
    if (*p == '{') {
        p=skip_ws(p+1,end); if(p<end&&*p=='}')return p+1;
        for (;;) {
            p=scan_json_string(p,end);if(!p)return NULL;p=skip_ws(p,end);
            if(p>=end||*p++!=':')return NULL;
            p=scan_json_value(p,end,depth+1);if(!p)return NULL;p=skip_ws(p,end);
            if(p<end&&*p==','){p=skip_ws(p+1,end);continue;}
            if(p<end&&*p=='}')return p+1;
            return NULL;
        }
    }
    if (*p == '[') {
        p=skip_ws(p+1,end);if(p<end&&*p==']')return p+1;
        for(;;){p=scan_json_value(p,end,depth+1);if(!p)return NULL;p=skip_ws(p,end);
            if(p<end&&*p==','){p=skip_ws(p+1,end);continue;}
            if(p<end&&*p==']')return p+1;
            return NULL;}
    }
    if(end-p>=4&&!memcmp(p,"true",4))return p+4;
    if(end-p>=5&&!memcmp(p,"false",5))return p+5;
    if(end-p>=4&&!memcmp(p,"null",4))return p+4;
    const char *start=p;
    if(*p=='-')p++;
    if(p>=end)return NULL;
    if(*p=='0')p++;else{if(!isdigit((unsigned char)*p))return NULL;while(p<end&&isdigit((unsigned char)*p))p++;}
    if(p<end&&*p=='.'){p++;if(p>=end||!isdigit((unsigned char)*p))return NULL;while(p<end&&isdigit((unsigned char)*p))p++;}
    if(p<end&&(*p=='e'||*p=='E')){p++;if(p<end&&(*p=='+'||*p=='-'))p++;
        if(p>=end||!isdigit((unsigned char)*p))return NULL;
        while(p<end&&isdigit((unsigned char)*p))p++;}
    return p>start?p:NULL;
}

static int json_complete(const char *json, json_span *span) {
    if(!json)return -1;
    const char *begin=skip_ws(json,json+strlen(json));
    const char *end=scan_json_value(begin,json+strlen(json),0);if(!end)return -1;
    if(skip_ws(end,json+strlen(json))!=json+strlen(json))return -1;
    if(span){span->begin=begin;span->end=end;}return 0;
}

static int json_is_null(json_span value) {
    const char *begin=skip_ws(value.begin,value.end);
    const char *end=value.end;
    while(end>begin&&isspace((unsigned char)end[-1]))end--;
    return end-begin==4&&!memcmp(begin,"null",4);
}

static int json_boolean(json_span value, int *out) {
    const char *begin=skip_ws(value.begin,value.end),*end=value.end;
    while(end>begin&&isspace((unsigned char)end[-1]))end--;
    if(end-begin==4&&!memcmp(begin,"true",4)){*out=1;return 0;}
    if(end-begin==5&&!memcmp(begin,"false",5)){*out=0;return 0;}
    return -1;
}

static int json_key_equals(json_span encoded, const char *key) {
    size_t n=(size_t)(encoded.end-encoded.begin);
    return n==strlen(key)+2 && encoded.begin[0]=='"' &&
           !memcmp(encoded.begin+1,key,n-2);
}

static int json_object_find(json_span object, const char *key, json_span *value) {
    const char *p=skip_ws(object.begin,object.end);
    if(p>=object.end||*p!='{')return -1;
    p=skip_ws(p+1,object.end);
    if(p<object.end&&*p=='}')return 1;
    for(;;){
        const char *kb=p,*ke=scan_json_string(p,object.end);if(!ke)return -1;
        json_span ks={kb,ke};p=skip_ws(ke,object.end);if(p>=object.end||*p++!=':')return -1;
        p=skip_ws(p,object.end);const char *vb=p,*ve=scan_json_value(p,object.end,1);if(!ve)return -1;
        if(json_key_equals(ks,key)){value->begin=vb;value->end=ve;return 0;}
        p=skip_ws(ve,object.end);if(p<object.end&&*p==','){p=skip_ws(p+1,object.end);continue;}
        if(p<object.end&&*p=='}')return 1;
        return -1;
    }
}

static int append_utf8(uint32_t cp, char *out, size_t capacity, size_t *used) {
    uint8_t bytes[4];int n;
    if(cp<0x80){bytes[0]=(uint8_t)cp;n=1;}
    else if(cp<0x800){bytes[0]=(uint8_t)(0xc0|(cp>>6));bytes[1]=(uint8_t)(0x80|(cp&63));n=2;}
    else if(cp<0x10000){bytes[0]=(uint8_t)(0xe0|(cp>>12));bytes[1]=(uint8_t)(0x80|((cp>>6)&63));bytes[2]=(uint8_t)(0x80|(cp&63));n=3;}
    else if(cp<=0x10ffff){bytes[0]=(uint8_t)(0xf0|(cp>>18));bytes[1]=(uint8_t)(0x80|((cp>>12)&63));bytes[2]=(uint8_t)(0x80|((cp>>6)&63));bytes[3]=(uint8_t)(0x80|(cp&63));n=4;}
    else return -1;
    if(*used+(size_t)n>=capacity)return -1;
    memcpy(out+*used,bytes,(size_t)n);*used+=(size_t)n;return 0;
}

static int json_string_copy(json_span span, char *out, size_t capacity) {
    if(capacity<1||span.end-span.begin<2||*span.begin!='"'||span.end[-1]!='"')return -1;
    const char *p=span.begin+1,*end=span.end-1;size_t used=0;
    while(p<end){unsigned char c=(unsigned char)*p++;
        if(c!='\\'){if(used+1>=capacity)return -1;out[used++]=(char)c;continue;}
        if(p>=end)return -1;
        char e=*p++;
        if(e=='u'){if(end-p<4)return -1;uint32_t cp=0;for(int i=0;i<4;i++){int h=hex_value(*p++);if(h<0)return -1;cp=(cp<<4)|(uint32_t)h;}
            if(cp>=0xd800&&cp<=0xdbff&&end-p>=6&&p[0]=='\\'&&p[1]=='u'){p+=2;uint32_t lo=0;for(int i=0;i<4;i++){int h=hex_value(*p++);if(h<0)return -1;lo=(lo<<4)|(uint32_t)h;}if(lo<0xdc00||lo>0xdfff)return -1;cp=0x10000+((cp-0xd800)<<10)+(lo-0xdc00);}
            if(append_utf8(cp,out,capacity,&used)!=0)return -1;
            continue;}
        char v;switch(e){case '"':v='"';break;case '\\':v='\\';break;case '/':v='/';break;
            case 'b':v='\b';break;case 'f':v='\f';break;case 'n':v='\n';break;case 'r':v='\r';break;case 't':v='\t';break;default:return -1;}
        if(used+1>=capacity)return -1;
        out[used++]=v;
    }
    out[used]=0;return 0;
}

static int json_integer(json_span span, int *out) {
    char buf[64];size_t n=(size_t)(span.end-span.begin);if(n<1||n>=sizeof(buf))return -1;
    memcpy(buf,span.begin,n);buf[n]=0;char *end=NULL;errno=0;long v=strtol(buf,&end,10);
    if(errno||!end||*end||v<INT_MIN||v>INT_MAX)return -1;
    *out=(int)v;return 0;
}
static int json_u64(json_span span, uint64_t *out) {
    char buf[64];size_t n=(size_t)(span.end-span.begin);if(n<1||n>=sizeof(buf)||span.begin[0]=='-')return -1;
    memcpy(buf,span.begin,n);buf[n]=0;char *end=NULL;errno=0;unsigned long long v=strtoull(buf,&end,10);
    if(errno||!end||*end)return -1;
    *out=(uint64_t)v;return 0;
}
static int json_float(json_span span, float *out) {
    char buf[96];size_t n=(size_t)(span.end-span.begin);if(n<1||n>=sizeof(buf))return -1;
    memcpy(buf,span.begin,n);buf[n]=0;char *end=NULL;errno=0;float v=strtof(buf,&end);
    if(errno||!end||*end)return -1;
    *out=v;return 0;
}

typedef struct {uint32_t h[8];uint64_t bits;uint8_t block[64];size_t used;} wire_sha256;
static uint32_t rotr32(uint32_t x,unsigned n){return(x>>n)|(x<<(32-n));}
static void sha_block(wire_sha256*s,const uint8_t b[64]){
    static const uint32_t k[64]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    uint32_t w[64];for(int i=0;i<16;i++)w[i]=((uint32_t)b[i*4]<<24)|((uint32_t)b[i*4+1]<<16)|((uint32_t)b[i*4+2]<<8)|b[i*4+3];
    for(int i=16;i<64;i++){uint32_t a=rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3),z=rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);w[i]=w[i-16]+a+w[i-7]+z;}
    uint32_t a=s->h[0],c=s->h[2],b0=s->h[1],d=s->h[3],e=s->h[4],f=s->h[5],g=s->h[6],h=s->h[7];
    for(int i=0;i<64;i++){uint32_t s1=rotr32(e,6)^rotr32(e,11)^rotr32(e,25),ch=(e&f)^((~e)&g),t1=h+s1+ch+k[i]+w[i],s0=rotr32(a,2)^rotr32(a,13)^rotr32(a,22),maj=(a&b0)^(a&c)^(b0&c),t2=s0+maj;h=g;g=f;f=e;e=d+t1;d=c;c=b0;b0=a;a=t1+t2;}
    s->h[0]+=a;s->h[1]+=b0;s->h[2]+=c;s->h[3]+=d;s->h[4]+=e;s->h[5]+=f;s->h[6]+=g;s->h[7]+=h;
}
static void sha_init(wire_sha256*s){static const uint32_t v[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};memcpy(s->h,v,sizeof(v));s->bits=0;s->used=0;}
static void sha_update(wire_sha256*s,const void*data_,size_t n){const uint8_t*d=data_;s->bits+=(uint64_t)n*8;while(n){size_t take=64-s->used;if(take>n)take=n;memcpy(s->block+s->used,d,take);s->used+=take;d+=take;n-=take;if(s->used==64){sha_block(s,s->block);s->used=0;}}}
static void sha_final(wire_sha256*s,uint8_t out[32]){s->block[s->used++]=0x80;if(s->used>56){while(s->used<64)s->block[s->used++]=0;sha_block(s,s->block);s->used=0;}while(s->used<56)s->block[s->used++]=0;for(int i=7;i>=0;i--)s->block[s->used++]=(uint8_t)(s->bits>>(i*8));sha_block(s,s->block);for(int i=0;i<8;i++){out[i*4]=(uint8_t)(s->h[i]>>24);out[i*4+1]=(uint8_t)(s->h[i]>>16);out[i*4+2]=(uint8_t)(s->h[i]>>8);out[i*4+3]=(uint8_t)s->h[i];}}
static void sha_hex(const void*data,size_t n,char out[65]){wire_sha256 s;uint8_t d[32];static const char h[]="0123456789abcdef";sha_init(&s);sha_update(&s,data,n);sha_final(&s,d);for(int i=0;i<32;i++){out[i*2]=h[d[i]>>4];out[i*2+1]=h[d[i]&15];}out[64]=0;}

void swarm_expert_wire_sha256_hex(const void *data,size_t length,char out[65]) {
    if(!out)return;
    if(!data&&length){out[0]=0;return;}
    sha_hex(data,length,out);
}

void swarm_expert_wire_free_packet(swarm_expert_packet *packet) {
    if(!packet)return;
    free(packet->header_json);free(packet->semantic_json);free(packet->blobs);memset(packet,0,sizeof(*packet));
}
void swarm_expert_wire_free_bytes(swarm_expert_owned_bytes *bytes) {
    if(!bytes)return;
    free(bytes->data);bytes->data=NULL;bytes->length=0;
}

int swarm_expert_wire_decode_packet(const uint8_t *payload,size_t payload_length,
        swarm_expert_packet *out,char *error,size_t error_capacity) {
    const size_t minimum=8+8;
    if(!payload||!out)return wire_error(error,error_capacity,"null packet argument");
    memset(out,0,sizeof(*out));
    if(payload_length<minimum||memcmp(payload,SWARM_EXPERT_WIRE_MAGIC,8))
        return wire_error(error,error_capacity,"invalid or truncated expert frame");
    if(payload_length>SWARM_EXPERT_MAX_FRAME_BYTES)
        return wire_error(error,error_capacity,"expert frame exceeds maximum size");
    uint32_t header_length=read_be32(payload+8),lengths_length=read_be32(payload+12);
    size_t header_end,lengths_end;
    if(add_size(minimum,header_length,&header_end)!=0||
       add_size(header_end,lengths_length,&lengths_end)!=0||lengths_end>payload_length||
       lengths_length%8)
        return wire_error(error,error_capacity,"expert frame length table is invalid");
    out->header_json=(char*)malloc((size_t)header_length+1);
    if(!out->header_json)return wire_error(error,error_capacity,"out of memory decoding expert header");
    memcpy(out->header_json,payload+minimum,header_length);out->header_json[header_length]=0;
    out->header_length=header_length;
    json_span root;
    if(json_complete(out->header_json,&root)!=0||root.begin>=root.end||*root.begin!='{'){
        swarm_expert_wire_free_packet(out);return wire_error(error,error_capacity,"expert header is not valid JSON object");}
    json_span count_span,kind_span,semantic_span;
    uint64_t count;
    if(json_object_find(root,"blob_count",&count_span)!=0||json_u64(count_span,&count)!=0||
       count>SIZE_MAX/sizeof(*out->blobs)||count>UINT32_MAX||
       json_object_find(root,"kind",&kind_span)!=0||json_string_copy(kind_span,out->kind,sizeof(out->kind))!=0||
       (strcmp(out->kind,"request")&&strcmp(out->kind,"response")&&strcmp(out->kind,"control"))||
       json_object_find(root,"semantic",&semantic_span)!=0||*semantic_span.begin!='{'){
        swarm_expert_wire_free_packet(out);return wire_error(error,error_capacity,"expert header semantic fields are invalid");}
    if(lengths_length!=count*8){swarm_expert_wire_free_packet(out);return wire_error(error,error_capacity,"blob count does not match length table");}
    out->semantic_length=(size_t)(semantic_span.end-semantic_span.begin);
    out->semantic_json=(char*)malloc(out->semantic_length+1);
    out->blobs=(swarm_expert_blob_view*)calloc((size_t)count,sizeof(*out->blobs));
    if(!out->semantic_json||(count&&!out->blobs)){swarm_expert_wire_free_packet(out);return wire_error(error,error_capacity,"out of memory decoding packet");}
    memcpy(out->semantic_json,semantic_span.begin,out->semantic_length);out->semantic_json[out->semantic_length]=0;
    out->blob_count=(size_t)count;size_t cursor=lengths_end;
    for(size_t i=0;i<out->blob_count;i++){
        uint64_t length=read_be64(payload+header_end+i*8);
        if(length>SIZE_MAX||cursor>payload_length-(size_t)length){swarm_expert_wire_free_packet(out);return wire_error(error,error_capacity,"expert frame blob is truncated");}
        out->blobs[i].data=payload+cursor;out->blobs[i].length=length;cursor+=(size_t)length;
    }
    if(cursor!=payload_length){swarm_expert_wire_free_packet(out);return wire_error(error,error_capacity,"expert frame has trailing bytes");}
    return 0;
}

int swarm_expert_wire_encode_packet(const char *kind,const char *semantic_json,
        const swarm_expert_blob_view *blobs,size_t blob_count,swarm_expert_owned_bytes *out,
        char *error,size_t error_capacity) {
    if(!kind||!semantic_json||!out||(blob_count&&!blobs))return wire_error(error,error_capacity,"null packet encode argument");
    memset(out,0,sizeof(*out));
    if(strcmp(kind,"request")&&strcmp(kind,"response")&&strcmp(kind,"control"))
        return wire_error(error,error_capacity,"unknown expert packet kind");
    json_span semantic;
    if(json_complete(semantic_json,&semantic)!=0||*semantic.begin!='{')
        return wire_error(error,error_capacity,"semantic payload must be one JSON object");
    size_t semantic_length=(size_t)(semantic.end-semantic.begin);
    int prefix=snprintf(NULL,0,"{\"blob_count\":%llu,\"kind\":\"%s\",\"semantic\":",
        (unsigned long long)blob_count,kind);
    if(prefix<0)return wire_error(error,error_capacity,"cannot format expert header");
    size_t header_length=(size_t)prefix+semantic_length+1;
    if(header_length>UINT32_MAX||blob_count>UINT32_MAX/8)
        return wire_error(error,error_capacity,"expert header or length table is too large");
    size_t payload_length=16+header_length+blob_count*8;
    for(size_t i=0;i<blob_count;i++){
        if(blobs[i].length>SIZE_MAX||add_size(payload_length,(size_t)blobs[i].length,&payload_length)!=0)
            return wire_error(error,error_capacity,"expert packet size overflows");
    }
    if(payload_length>SWARM_EXPERT_MAX_FRAME_BYTES)
        return wire_error(error,error_capacity,"expert frame exceeds maximum size");
    uint8_t *encoded=(uint8_t*)malloc(payload_length);if(!encoded)return wire_error(error,error_capacity,"out of memory encoding packet");
    memcpy(encoded,SWARM_EXPERT_WIRE_MAGIC,8);write_be32(encoded+8,(uint32_t)header_length);write_be32(encoded+12,(uint32_t)(blob_count*8));
    int wrote=snprintf((char*)encoded+16,(size_t)prefix+1,"{\"blob_count\":%llu,\"kind\":\"%s\",\"semantic\":",
        (unsigned long long)blob_count,kind);
    if(wrote!=prefix){free(encoded);return wire_error(error,error_capacity,"expert header formatting failed");}
    memcpy(encoded+16+prefix,semantic.begin,semantic_length);encoded[16+prefix+semantic_length]='}';
    size_t header_end=16+header_length,cursor=header_end+blob_count*8;
    for(size_t i=0;i<blob_count;i++){write_be64(encoded+header_end+i*8,blobs[i].length);memcpy(encoded+cursor,blobs[i].data,(size_t)blobs[i].length);cursor+=(size_t)blobs[i].length;}
    out->data=encoded;out->length=payload_length;return 0;
}

int swarm_expert_wire_frame_with_length(const uint8_t *payload,size_t payload_length,
        swarm_expert_owned_bytes *out,char *error,size_t error_capacity) {
    if(!payload||!out||payload_length>SWARM_EXPERT_MAX_FRAME_BYTES)
        return wire_error(error,error_capacity,"invalid length-framed payload");
    if(payload_length>SIZE_MAX-8)return wire_error(error,error_capacity,"length frame overflows");
    out->data=(uint8_t*)malloc(payload_length+8);if(!out->data)return wire_error(error,error_capacity,"out of memory framing packet");
    write_be64(out->data,(uint64_t)payload_length);memcpy(out->data+8,payload,payload_length);out->length=payload_length+8;return 0;
}

static int parse_int_array(json_span array,int64_t *values,int capacity,int *count) {
    const char*p=skip_ws(array.begin,array.end);if(p>=array.end||*p!='[')return -1;p=skip_ws(p+1,array.end);int n=0;
    if(p<array.end&&*p==']'){*count=0;return 0;}
    for(;;){const char*b=p,*e=scan_json_value(p,array.end,1);json_span item={b,e};int v;if(!e||json_integer(item,&v)!=0||n>=capacity)return -1;values[n++]=v;p=skip_ws(e,array.end);
        if(p<array.end&&*p==','){p=skip_ws(p+1,array.end);continue;}if(p<array.end&&*p==']'){*count=n;return 0;}return -1;}
}

int swarm_expert_wire_decode_tensor_f32(const uint8_t *payload,size_t payload_length,
        swarm_expert_tensor_f32_view *out,char *error,size_t error_capacity) {
    if(!payload||!out||payload_length<12||memcmp(payload,SWARM_EXPERT_TENSOR_MAGIC,8))
        return wire_error(error,error_capacity,"invalid activation tensor magic or truncated envelope");
    memset(out,0,sizeof(*out));uint32_t header_length=read_be32(payload+8);
    size_t header_end;if(add_size(12,header_length,&header_end)!=0||header_end>payload_length)
        return wire_error(error,error_capacity,"activation header length exceeds envelope size");
    char *header=(char*)malloc((size_t)header_length+1);if(!header)return wire_error(error,error_capacity,"out of memory decoding tensor header");
    memcpy(header,payload+12,header_length);header[header_length]=0;json_span root;
    if(json_complete(header,&root)!=0||*root.begin!='{'){free(header);return wire_error(error,error_capacity,"invalid activation tensor JSON header");}
    json_span span;uint64_t declared_length;char checksum[65],byte_order[16];int rc=-1;
#define GET_STRING(name,dest) (json_object_find(root,name,&span)==0&&json_string_copy(span,dest,sizeof(dest))==0)
    if(!GET_STRING("tensor_id",out->tensor_id)||!GET_STRING("request_id",out->request_id)||
       !GET_STRING("model_revision",out->model_revision)||!GET_STRING("partition_hash",out->partition_hash)||
       !GET_STRING("dtype",out->dtype)||!GET_STRING("byte_order",byte_order)||
       !GET_STRING("checksum",checksum)||strcmp(out->dtype,"float32")||strcmp(byte_order,"little")||
       json_object_find(root,"stage_id",&span)!=0||json_integer(span,&out->stage_id)!=0||
       json_object_find(root,"token_position",&span)!=0||json_integer(span,&out->token_position)!=0||
       json_object_find(root,"sequence_length",&span)!=0||json_integer(span,&out->sequence_length)!=0||
       json_object_find(root,"route_generation",&span)!=0||json_integer(span,&out->route_generation)!=0||
       json_object_find(root,"payload_length",&span)!=0||json_u64(span,&declared_length)!=0||declared_length!=payload_length-header_end)
        goto done;
    if(json_object_find(root,"shape",&span)!=0)goto done;
    int count=0;
    if(parse_int_array(span,out->shape,SWARM_EXPERT_MAX_DIMS,&count)!=0||count<1)goto done;
    out->ndim=count;
    size_t elements=1;for(int i=0;i<count;i++){if(out->shape[i]<=0||(uint64_t)out->shape[i]>SIZE_MAX/elements)goto done;elements*=(size_t)out->shape[i];}
    if(elements>SIZE_MAX/sizeof(float)||elements*sizeof(float)!=(size_t)declared_length)goto done;
    char actual[65];sha_hex(payload+header_end,(size_t)declared_length,actual);if(strcmp(actual,checksum))goto done;
    {uint16_t endian=1;if(*(uint8_t*)&endian!=1)goto done;}
    out->raw=payload+header_end;out->raw_length=(size_t)declared_length;out->values=(const float*)(const void*)out->raw;out->value_count=elements;rc=0;
done:
#undef GET_STRING
    free(header);if(rc!=0)return wire_error(error,error_capacity,"activation tensor metadata, shape, or checksum is invalid");return 0;
}

static char *json_escape_ascii(const char *source) {
    if(!source)return NULL;
    size_t n=strlen(source);if(n>SIZE_MAX/6-1)return NULL;
    char*out=(char*)malloc(n*6+1);if(!out)return NULL;size_t used=0;
    static const char hex[]="0123456789abcdef";
    for(size_t i=0;i<n;i++){unsigned char c=(unsigned char)source[i];
        if(c=='"'||c=='\\'){out[used++]='\\';out[used++]=(char)c;}
        else if(c=='\b'){out[used++]='\\';out[used++]='b';}else if(c=='\f'){out[used++]='\\';out[used++]='f';}
        else if(c=='\n'){out[used++]='\\';out[used++]='n';}else if(c=='\r'){out[used++]='\\';out[used++]='r';}
        else if(c=='\t'){out[used++]='\\';out[used++]='t';}
        else if(c<0x20||c>=0x80){out[used++]='\\';out[used++]='u';out[used++]='0';out[used++]='0';out[used++]=hex[c>>4];out[used++]=hex[c&15];}
        else out[used++]=(char)c;
    }out[used]=0;return out;
}

static int format_i64_array(const int64_t *values,int count,char **out) {
    size_t capacity=(size_t)count*32+3;char*text=(char*)malloc(capacity);if(!text)return -1;
    size_t used=0;text[used++]='[';for(int i=0;i<count;i++){int n=snprintf(text+used,capacity-used,"%s%lld",i?",":"",(long long)values[i]);if(n<0||(size_t)n>=capacity-used){free(text);return -1;}used+=(size_t)n;}text[used++]=']';text[used]=0;*out=text;return 0;
}

int swarm_expert_wire_encode_tensor_f32(const char *tensor_id,const char *request_id,
        int stage_id,int token_position,int sequence_length,const char *model_revision,
        const char *partition_hash,int route_generation,const int64_t *shape,int ndim,
        const float *values,swarm_expert_owned_bytes *out,char *error,size_t error_capacity) {
    if(!tensor_id||!request_id||!model_revision||!partition_hash||!shape||!values||!out||ndim<1||ndim>SWARM_EXPERT_MAX_DIMS)
        return wire_error(error,error_capacity,"invalid tensor encode argument");
    memset(out,0,sizeof(*out));uint16_t endian=1;if(*(uint8_t*)&endian!=1)return wire_error(error,error_capacity,"float32 tensor codec requires little-endian host");
    size_t elements=1;int64_t strides[SWARM_EXPERT_MAX_DIMS];int64_t stride=4;
    for(int i=ndim-1;i>=0;i--){if(shape[i]<=0||(uint64_t)shape[i]>SIZE_MAX/elements)return wire_error(error,error_capacity,"tensor shape overflows");strides[i]=stride;elements*=(size_t)shape[i];if(i&&shape[i]>INT64_MAX/stride)return wire_error(error,error_capacity,"tensor stride overflows");stride*=shape[i];}
    if(elements>SIZE_MAX/sizeof(float))return wire_error(error,error_capacity,"tensor byte count overflows");
    size_t raw_length=elements*sizeof(float);
    char checksum[65];sha_hex(values,raw_length,checksum);char*shape_json=NULL,*strides_json=NULL;
    char*et=json_escape_ascii(tensor_id),*er=json_escape_ascii(request_id),*em=json_escape_ascii(model_revision),*ep=json_escape_ascii(partition_hash);
    if(!et||!er||!em||!ep||format_i64_array(shape,ndim,&shape_json)!=0||format_i64_array(strides,ndim,&strides_json)!=0){free(et);free(er);free(em);free(ep);free(shape_json);free(strides_json);return wire_error(error,error_capacity,"out of memory encoding tensor JSON");}
    int header_length=snprintf(NULL,0,
        "{\"byte_order\":\"little\",\"checksum\":\"%s\",\"dtype\":\"float32\",\"model_revision\":\"%s\",\"partition_hash\":\"%s\",\"payload_length\":%llu,\"request_id\":\"%s\",\"route_generation\":%d,\"sequence_length\":%d,\"shape\":%s,\"stage_id\":%d,\"strides\":%s,\"tensor_id\":\"%s\",\"token_position\":%d}",
        checksum,em,ep,(unsigned long long)raw_length,er,route_generation,sequence_length,shape_json,stage_id,strides_json,et,token_position);
    if(header_length<0||(uint64_t)header_length>UINT32_MAX||raw_length>SIZE_MAX-12-(size_t)header_length){free(et);free(er);free(em);free(ep);free(shape_json);free(strides_json);return wire_error(error,error_capacity,"tensor header is too large");}
    size_t total=12+(size_t)header_length+raw_length;uint8_t*encoded=(uint8_t*)malloc(total);if(!encoded){free(et);free(er);free(em);free(ep);free(shape_json);free(strides_json);return wire_error(error,error_capacity,"out of memory encoding tensor");}
    memcpy(encoded,SWARM_EXPERT_TENSOR_MAGIC,8);write_be32(encoded+8,(uint32_t)header_length);
    int wrote=snprintf((char*)encoded+12,(size_t)header_length+1,
        "{\"byte_order\":\"little\",\"checksum\":\"%s\",\"dtype\":\"float32\",\"model_revision\":\"%s\",\"partition_hash\":\"%s\",\"payload_length\":%llu,\"request_id\":\"%s\",\"route_generation\":%d,\"sequence_length\":%d,\"shape\":%s,\"stage_id\":%d,\"strides\":%s,\"tensor_id\":\"%s\",\"token_position\":%d}",
        checksum,em,ep,(unsigned long long)raw_length,er,route_generation,sequence_length,shape_json,stage_id,strides_json,et,token_position);
    free(et);free(er);free(em);free(ep);free(shape_json);free(strides_json);
    if(wrote!=header_length){free(encoded);return wire_error(error,error_capacity,"tensor header formatting failed");}
    memcpy(encoded+12+header_length,values,raw_length);out->data=encoded;out->length=total;return 0;
}

typedef struct {char *data;size_t length,capacity;} wire_text;
static void wire_text_free(wire_text *text){if(text){free(text->data);memset(text,0,sizeof(*text));}}
static int wire_text_reserve(wire_text *text,size_t extra){
    if(extra>SIZE_MAX-text->length-1)return -1;
    size_t needed=text->length+extra+1;
    if(needed<=text->capacity)return 0;
    size_t capacity=text->capacity?text->capacity:1024;
    while(capacity<needed){if(capacity>SIZE_MAX/2){capacity=needed;break;}capacity*=2;}
    char*grown=(char*)realloc(text->data,capacity);if(!grown)return -1;text->data=grown;text->capacity=capacity;return 0;
}
static int wire_text_append(wire_text *text,const char *value){
    size_t n=strlen(value);if(wire_text_reserve(text,n)!=0)return -1;
    memcpy(text->data+text->length,value,n);text->length+=n;text->data[text->length]=0;return 0;
}
static int wire_text_appendf(wire_text *text,const char *fmt,...){
    va_list ap,copy;va_start(ap,fmt);va_copy(copy,ap);int n=vsnprintf(NULL,0,fmt,copy);va_end(copy);
    if(n<0||wire_text_reserve(text,(size_t)n)!=0){va_end(ap);return -1;}
    int wrote=vsnprintf(text->data+text->length,text->capacity-text->length,fmt,ap);va_end(ap);
    if(wrote!=n)return -1;
    text->length+=(size_t)n;return 0;
}

int swarm_expert_wire_encode_route_request(const swarm_expert_route_request *request,
        uint64_t deadline_ns,const char *evidence_category,int exact_determinism,
        swarm_expert_owned_bytes *out,char *error,size_t error_capacity){
    if(!request||!out||!evidence_category||request->batch_rows<1||request->latent_dimension<1||
       request->top_k<1||!request->expert_ids_by_row||!request->routing_weights_by_row||
       !request->selected_rank_by_row||!request->activation.values||deadline_ns<1)
        return wire_error(error,error_capacity,"invalid route request encode argument");
    if(request->execution_mode!=SWARM_EXPERT_EXECUTION_WHOLE&&
       request->execution_mode!=SWARM_EXPERT_EXECUTION_MICROSHARD)
        return wire_error(error,error_capacity,"invalid route request execution mode");
    if((request->execution_mode==SWARM_EXPERT_EXECUTION_MICROSHARD&&
        (request->hidden_start<0||request->hidden_end<=request->hidden_start))||
       (request->execution_mode==SWARM_EXPERT_EXECUTION_WHOLE&&
        (request->hidden_start!=0||request->hidden_end!=0||request->microshard_final||
         request->down_accumulators.values)))
        return wire_error(error,error_capacity,"invalid route request hidden range");
    if(request->microshard_final!=0&&request->microshard_final!=1)
        return wire_error(error,error_capacity,"invalid microshard final marker");
    if(request->challenge!=0&&request->challenge!=1)
        return wire_error(error,error_capacity,"invalid challenge marker");
    int needs_accumulator=request->execution_mode==SWARM_EXPERT_EXECUTION_MICROSHARD&&
        request->response_mode==SWARM_EXPERT_RESPONSE_PER_EXPERT_EXACT&&request->hidden_start>0;
    if(request->execution_mode==SWARM_EXPERT_EXECUTION_MICROSHARD&&
       request->response_mode==SWARM_EXPERT_RESPONSE_PER_WORKER_FAST&&request->microshard_final)
        return wire_error(error,error_capacity,"fast microshard cannot carry exact chain state");
    if(needs_accumulator!=!!request->down_accumulators.values)
        return wire_error(error,error_capacity,"microshard accumulator presence differs from hidden range");
    memset(out,0,sizeof(*out));
    if((size_t)request->batch_rows>SIZE_MAX/(size_t)request->top_k)
        return wire_error(error,error_capacity,"route request matrix is too large");
    size_t routes=(size_t)request->batch_rows*(size_t)request->top_k;
    if(routes>SWARM_EXPERT_MAX_FRAME_BYTES/12)
        return wire_error(error,error_capacity,"route request matrix is too large");
    for(size_t i=0;i<routes;i++)if(request->expert_ids_by_row[i]<0||
       request->selected_rank_by_row[i]!=(int)(i%(size_t)request->top_k)||
       !isfinite(request->routing_weights_by_row[i]))
        return wire_error(error,error_capacity,"route request ranks, IDs, or weights are invalid");
    int64_t shape[2]={request->batch_rows,request->latent_dimension};
    char tensor_id[512];int idn=snprintf(tensor_id,sizeof(tensor_id),"%s:expert-activation",request->request_id);
    if(idn<0||(size_t)idn>=sizeof(tensor_id))return wire_error(error,error_capacity,"request tensor identity is too long");
    swarm_expert_owned_bytes tensor={0},accumulator_tensor={0};
    if(swarm_expert_wire_encode_tensor_f32(tensor_id,request->request_id,request->layer_id,
            request->activation.token_position,request->batch_rows,request->model_revision,
            request->quantization_fingerprint,request->activation.route_generation,shape,2,
            request->activation.values,&tensor,error,error_capacity)!=0)return -1;
    char outer_checksum[65],accumulator_checksum[65]={0};sha_hex(tensor.data,tensor.length,outer_checksum);
    if(needs_accumulator){
        if(routes>SIZE_MAX/(size_t)request->latent_dimension){
            swarm_expert_wire_free_bytes(&tensor);
            return wire_error(error,error_capacity,"microshard accumulator shape overflows");
        }
        size_t accumulator_values=routes*(size_t)request->latent_dimension;
        if(request->down_accumulators.value_count&&
           request->down_accumulators.value_count!=accumulator_values){
            swarm_expert_wire_free_bytes(&tensor);
            return wire_error(error,error_capacity,"microshard accumulator value count differs from geometry");
        }
        int64_t accumulator_shape[3]={request->batch_rows,request->top_k,request->latent_dimension};
        int accumulator_idn=snprintf(tensor_id,sizeof(tensor_id),"%s:down-accumulators",request->request_id);
        if(accumulator_idn<0||(size_t)accumulator_idn>=sizeof(tensor_id)||
           swarm_expert_wire_encode_tensor_f32(tensor_id,request->request_id,request->layer_id,
                request->activation.token_position,request->batch_rows,request->model_revision,
                request->quantization_fingerprint,request->activation.route_generation,
                accumulator_shape,3,request->down_accumulators.values,&accumulator_tensor,
                error,error_capacity)!=0){
            swarm_expert_wire_free_bytes(&tensor);return -1;
        }
        sha_hex(accumulator_tensor.data,accumulator_tensor.length,accumulator_checksum);
    }
    char *request_id=json_escape_ascii(request->request_id),*model_id=json_escape_ascii(request->model_id);
    char *revision=json_escape_ascii(request->model_revision),*quant=json_escape_ascii(request->quantization_fingerprint);
    char *evidence=json_escape_ascii(evidence_category);wire_text semantic={0};int rc=-1;
    if(!request_id||!model_id||!revision||!quant||!evidence)goto done;
    const char *mode=request->response_mode==SWARM_EXPERT_RESPONSE_PER_EXPERT_EXACT?"per_expert_exact":"per_worker_fast";
    const char *execution=request->execution_mode==SWARM_EXPERT_EXECUTION_MICROSHARD?"microshard":"whole_expert";
    char hidden_start[32]="null",hidden_end[32]="null";
    if(request->execution_mode==SWARM_EXPERT_EXECUTION_MICROSHARD){
        snprintf(hidden_start,sizeof(hidden_start),"%d",request->hidden_start);
        snprintf(hidden_end,sizeof(hidden_end),"%d",request->hidden_end);
    }
    if(wire_text_appendf(&semantic,
        "{\"schema_version\":\"1.0\",\"request_id\":\"%s\",\"model_id\":\"%s\",\"model_revision\":\"%s\",\"quantization_fingerprint\":\"%s\",\"layer_id\":%d,\"batch_rows\":%d,\"latent_dimension\":%d,\"expert_ids\":[],\"routing_weights\":[],\"top_k\":%d,\"expert_ids_by_row\":[",
        request_id,model_id,revision,quant,request->layer_id,request->batch_rows,
        request->latent_dimension,request->top_k)!=0)goto done;
    for(int row=0;row<request->batch_rows;row++){if(row&&wire_text_append(&semantic,",")!=0)goto done;if(wire_text_append(&semantic,"[")!=0)goto done;
        for(int rank=0;rank<request->top_k;rank++){if(rank&&wire_text_append(&semantic,",")!=0)goto done;if(wire_text_appendf(&semantic,"%d",request->expert_ids_by_row[(size_t)row*request->top_k+rank])!=0)goto done;}
        if(wire_text_append(&semantic,"]")!=0)goto done;}
    if(wire_text_append(&semantic,"],\"routing_weights_by_row\":[")!=0)goto done;
    for(int row=0;row<request->batch_rows;row++){if(row&&wire_text_append(&semantic,",")!=0)goto done;if(wire_text_append(&semantic,"[")!=0)goto done;
        for(int rank=0;rank<request->top_k;rank++){if(rank&&wire_text_append(&semantic,",")!=0)goto done;if(wire_text_appendf(&semantic,"%.9g",(double)request->routing_weights_by_row[(size_t)row*request->top_k+rank])!=0)goto done;}
        if(wire_text_append(&semantic,"]")!=0)goto done;}
    if(wire_text_append(&semantic,"],\"selected_rank_by_row\":[")!=0)goto done;
    for(int row=0;row<request->batch_rows;row++){if(row&&wire_text_append(&semantic,",")!=0)goto done;if(wire_text_append(&semantic,"[")!=0)goto done;
        for(int rank=0;rank<request->top_k;rank++){if(rank&&wire_text_append(&semantic,",")!=0)goto done;if(wire_text_appendf(&semantic,"%d",request->selected_rank_by_row[(size_t)row*request->top_k+rank])!=0)goto done;}
        if(wire_text_append(&semantic,"]")!=0)goto done;}
    size_t raw_bytes=(size_t)request->batch_rows*(size_t)request->latent_dimension*sizeof(float);
    if(wire_text_appendf(&semantic,
        "],\"response_mode\":\"%s\",\"activations\":{\"name\":\"activations\",\"envelope\":\"SWARMT01\",\"dtype\":\"float32\",\"shape\":[%d,%d],\"codec\":\"raw_fp32\",\"payload_index\":0,\"raw_bytes\":%llu,\"encoded_bytes\":%llu,\"scale\":null,\"checksum\":\"%s\"},\"deadline_ns\":%llu,\"execution_mode\":\"%s\",\"determinism_mode\":\"%s\",\"compression\":\"raw_fp32\",\"hidden_start\":%s,\"hidden_end\":%s,\"microshard_final\":%s,\"reduction_mode\":\"fixed_order_fp32\",\"challenge\":%s,\"metadata\":{\"evidence_category\":\"%s\",\"token_position\":%d,\"exact_contribution_representation\":\"%s\"},\"down_accumulators\":",
        mode,request->batch_rows,request->latent_dimension,(unsigned long long)raw_bytes,
        (unsigned long long)tensor.length,outer_checksum,(unsigned long long)deadline_ns,
        execution,exact_determinism?"exact":"quality_bounded",hidden_start,hidden_end,
        request->microshard_final?"true":"false",request->challenge?"true":"false",
        evidence,request->activation.token_position,
        request->response_mode==SWARM_EXPERT_RESPONSE_PER_EXPERT_EXACT?
            "unweighted_expert_output":"worker_weighted_sum")!=0)goto done;
    if(needs_accumulator){
        size_t accumulator_raw=routes*(size_t)request->latent_dimension*sizeof(float);
        if(wire_text_appendf(&semantic,
            "{\"name\":\"down_accumulators\",\"envelope\":\"SWARMT01\",\"dtype\":\"float32\",\"shape\":[%d,%d,%d],\"codec\":\"raw_fp32\",\"payload_index\":1,\"raw_bytes\":%llu,\"encoded_bytes\":%llu,\"scale\":null,\"checksum\":\"%s\"}",
            request->batch_rows,request->top_k,request->latent_dimension,
            (unsigned long long)accumulator_raw,
            (unsigned long long)accumulator_tensor.length,accumulator_checksum)!=0)goto done;
    }else if(wire_text_append(&semantic,"null")!=0)goto done;
    if(wire_text_append(&semantic,"}")!=0)goto done;
    {swarm_expert_blob_view blobs[2]={{tensor.data,tensor.length},
                                     {accumulator_tensor.data,accumulator_tensor.length}};
     if(swarm_expert_wire_encode_packet("request",semantic.data,blobs,
            needs_accumulator?2:1,out,error,error_capacity)!=0)goto done;}
    rc=0;
done:
    free(request_id);free(model_id);free(revision);free(quant);free(evidence);
    wire_text_free(&semantic);swarm_expert_wire_free_bytes(&tensor);
    swarm_expert_wire_free_bytes(&accumulator_tensor);
    if(rc!=0&&error&&error_capacity&&!error[0])
        wire_error(error,error_capacity,"cannot encode route request semantic JSON");
    return rc;
}

int swarm_expert_wire_encode_route_response(const swarm_expert_route_response *response,
        const int *experts_executed,int expert_count,const char *worker_signature,
        swarm_expert_owned_bytes *out,char *error,size_t error_capacity){
    if(!response||!out||!worker_signature||expert_count<0||(expert_count&&!experts_executed)||
       !response->result.values||response->result.ndim<1||response->result.ndim>SWARM_EXPERT_MAX_DIMS)
        return wire_error(error,error_capacity,"invalid route response encode argument");
    size_t result_values=1;
    for(int i=0;i<response->result.ndim;i++){
        if(response->result.shape[i]<=0||(uint64_t)response->result.shape[i]>SIZE_MAX/result_values)
            return wire_error(error,error_capacity,"route response shape overflows");
        result_values*=(size_t)response->result.shape[i];
    }
    if(response->result.value_count&&response->result.value_count!=result_values)
        return wire_error(error,error_capacity,"route response value count differs from shape");
    memset(out,0,sizeof(*out));char tensor_id[512];int idn=snprintf(tensor_id,sizeof(tensor_id),"%s:expert-result",response->request_id);
    if(idn<0||(size_t)idn>=sizeof(tensor_id))return wire_error(error,error_capacity,"response tensor identity is too long");
    swarm_expert_owned_bytes tensor={0};
    if(swarm_expert_wire_encode_tensor_f32(tensor_id,response->request_id,response->layer_id,
            response->result.token_position,(int)response->result.shape[0],response->model_revision,
            response->model_fingerprint,response->result.route_generation,response->result.shape,
            response->result.ndim,response->result.values,&tensor,error,error_capacity)!=0)return -1;
    char outer_checksum[65],raw_hash[65];sha_hex(tensor.data,tensor.length,outer_checksum);
    sha_hex(response->result.values,result_values*sizeof(float),raw_hash);
    char *request_id=json_escape_ascii(response->request_id),*worker_id=json_escape_ascii(response->worker_id);
    char *revision=json_escape_ascii(response->model_revision),*fingerprint=json_escape_ascii(response->model_fingerprint);
    char *signature=json_escape_ascii(worker_signature);wire_text semantic={0};int rc=-1;
    if(!request_id||!worker_id||!revision||!fingerprint||!signature)goto done;
    if(wire_text_appendf(&semantic,
        "{\"schema_version\":\"1.0\",\"request_id\":\"%s\",\"worker_id\":\"%s\",\"model_revision\":\"%s\",\"layer_id\":%d,\"result\":{\"name\":\"result\",\"envelope\":\"SWARMT01\",\"dtype\":\"float32\",\"shape\":[",
        request_id,worker_id,revision,response->layer_id)!=0)goto done;
    for(int i=0;i<response->result.ndim;i++){if(i&&wire_text_append(&semantic,",")!=0)goto done;if(wire_text_appendf(&semantic,"%lld",(long long)response->result.shape[i])!=0)goto done;}
    if(wire_text_appendf(&semantic,
        "],\"codec\":\"raw_fp32\",\"payload_index\":0,\"raw_bytes\":%llu,\"encoded_bytes\":%llu,\"scale\":null,\"checksum\":\"%s\"},\"execution_metadata\":{\"experts_executed\":[",
        (unsigned long long)(result_values*sizeof(float)),
        (unsigned long long)tensor.length,outer_checksum)!=0)goto done;
    for(int i=0;i<expert_count;i++){if(i&&wire_text_append(&semantic,",")!=0)goto done;if(wire_text_appendf(&semantic,"%d",experts_executed[i])!=0)goto done;}
    if(wire_text_appendf(&semantic,
        "],\"bytes_read\":%llu,\"bytes_received\":%llu,\"bytes_sent\":%llu,\"cache_hits\":0,\"cache_misses\":0,\"compute_ns\":%llu,\"queue_ns\":%llu,\"transfer_ns\":%llu,\"serialisation_ns\":0,\"copy_ns\":0,\"kernel_transition_ns\":0,\"backend\":\"native_colibri_cpu\",\"device\":\"cpu\",\"resident_tensor_bytes\":0,\"expert_resident_bytes\":0,\"fallback_events\":[]},\"integrity\":{\"result_hash\":\"sha256:%s\",\"model_fingerprint\":\"%s\",\"worker_signature\":\"%s\"},\"status\":\"ok\",\"error\":null}",
        (unsigned long long)response->bytes_read,(unsigned long long)response->bytes_received,
        (unsigned long long)response->bytes_sent,(unsigned long long)response->compute_ns,
        (unsigned long long)response->queue_ns,(unsigned long long)response->transfer_ns,
        raw_hash,fingerprint,signature)!=0)goto done;
    {swarm_expert_blob_view blob={tensor.data,tensor.length};
     if(swarm_expert_wire_encode_packet("response",semantic.data,&blob,1,out,error,error_capacity)!=0)goto done;}
    rc=0;
done:
    free(request_id);free(worker_id);free(revision);free(fingerprint);free(signature);
    wire_text_free(&semantic);swarm_expert_wire_free_bytes(&tensor);
    if(rc!=0&&error&&error_capacity&&!error[0])
        wire_error(error,error_capacity,"cannot encode route response semantic JSON");
    return rc;
}

static int parse_float_array(json_span array,float *values,int capacity,int *count) {
    const char*p=skip_ws(array.begin,array.end);if(p>=array.end||*p!='[')return -1;p=skip_ws(p+1,array.end);int n=0;
    if(p<array.end&&*p==']'){*count=0;return 0;}
    for(;;){const char*b=p,*e=scan_json_value(p,array.end,1);json_span item={b,e};float v;if(!e||json_float(item,&v)!=0||!isfinite(v)||n>=capacity)return -1;values[n++]=v;p=skip_ws(e,array.end);
        if(p<array.end&&*p==','){p=skip_ws(p+1,array.end);continue;}if(p<array.end&&*p==']'){*count=n;return 0;}return -1;}
}

static int parse_int_matrix(json_span matrix,int rows,int columns,int *values) {
    const char*p=skip_ws(matrix.begin,matrix.end);if(p>=matrix.end||*p!='[')return -1;p=skip_ws(p+1,matrix.end);
    for(int row=0;row<rows;row++){const char*b=p,*e=scan_json_value(p,matrix.end,1);int count=0;if(!e)return -1;
        /* Parse through a temporary int64 row to keep range checks explicit. */
        int64_t*tmp=(int64_t*)malloc((size_t)columns*sizeof(*tmp));if(!tmp)return -1;
        if(parse_int_array((json_span){b,e},tmp,columns,&count)!=0||count!=columns){free(tmp);return -1;}
        for(int col=0;col<columns;col++){if(tmp[col]<INT_MIN||tmp[col]>INT_MAX){free(tmp);return -1;}values[row*columns+col]=(int)tmp[col];}free(tmp);
        p=skip_ws(e,matrix.end);if(row+1<rows){if(p>=matrix.end||*p!=',')return -1;p=skip_ws(p+1,matrix.end);}}
    return p<matrix.end&&*p==']'&&skip_ws(p+1,matrix.end)==matrix.end?0:-1;
}

static int parse_float_matrix(json_span matrix,int rows,int columns,float *values) {
    const char*p=skip_ws(matrix.begin,matrix.end);if(p>=matrix.end||*p!='[')return -1;p=skip_ws(p+1,matrix.end);
    for(int row=0;row<rows;row++){const char*b=p,*e=scan_json_value(p,matrix.end,1);int count=0;if(!e||parse_float_array((json_span){b,e},values+row*columns,columns,&count)!=0||count!=columns)return -1;
        p=skip_ws(e,matrix.end);if(row+1<rows){if(p>=matrix.end||*p!=',')return -1;p=skip_ws(p+1,matrix.end);}}
    return p<matrix.end&&*p==']'&&skip_ws(p+1,matrix.end)==matrix.end?0:-1;
}

static int json_array_count(json_span array,int *count) {
    const char*p=skip_ws(array.begin,array.end);if(p>=array.end||*p!='[')return -1;p=skip_ws(p+1,array.end);int n=0;
    if(p<array.end&&*p==']'){*count=0;return 0;}
    for(;;){p=scan_json_value(p,array.end,1);if(!p||n==INT_MAX)return -1;n++;p=skip_ws(p,array.end);
        if(p<array.end&&*p==','){p=skip_ws(p+1,array.end);continue;}if(p<array.end&&*p==']'){*count=n;return skip_ws(p+1,array.end)==array.end?0:-1;}return -1;}
}

static int copy_flat_ints(json_span array,int *out,int count) {
    int64_t*tmp=(int64_t*)malloc((size_t)count*sizeof(*tmp));if(!tmp)return -1;int found=0;
    int rc=parse_int_array(array,tmp,count,&found);if(rc==0&&found==count)for(int i=0;i<count;i++){if(tmp[i]<INT_MIN||tmp[i]>INT_MAX){rc=-1;break;}out[i]=(int)tmp[i];}
    free(tmp);return rc==0&&found==count?0:-1;
}

void swarm_expert_wire_free_route_request(swarm_expert_route_request *request) {
    if(!request)return;
    free(request->expert_ids_by_row);free(request->routing_weights_by_row);free(request->selected_rank_by_row);memset(request,0,sizeof(*request));
}

int swarm_expert_wire_decode_route_request(const uint8_t *payload,size_t payload_length,
        swarm_expert_route_request *out,char *error,size_t error_capacity) {
    if(!out)return wire_error(error,error_capacity,"null route request output");
    memset(out,0,sizeof(*out));
    swarm_expert_packet packet;
    if(swarm_expert_wire_decode_packet(payload,payload_length,&packet,error,error_capacity)!=0)return -1;
    int rc=-1;json_span root,span;
    if(strcmp(packet.kind,"request")||(packet.blob_count!=1&&packet.blob_count!=2)||
       json_complete(packet.semantic_json,&root)!=0||*root.begin!='{'){
        wire_error(error,error_capacity,"expected one- or two-blob expert request");goto done;}
#define REQUEST_STRING(name,dest) (json_object_find(root,name,&span)==0&&json_string_copy(span,dest,sizeof(dest))==0)
    if(!REQUEST_STRING("request_id",out->request_id)||!REQUEST_STRING("model_id",out->model_id)||
       !REQUEST_STRING("model_revision",out->model_revision)||!REQUEST_STRING("quantization_fingerprint",out->quantization_fingerprint)||
       json_object_find(root,"layer_id",&span)!=0||json_integer(span,&out->layer_id)!=0||out->layer_id<0||
       json_object_find(root,"batch_rows",&span)!=0||json_integer(span,&out->batch_rows)!=0||out->batch_rows<1||out->batch_rows>(1<<20)||
       json_object_find(root,"latent_dimension",&span)!=0||json_integer(span,&out->latent_dimension)!=0||out->latent_dimension<1||out->latent_dimension>(1<<20)){
        wire_error(error,error_capacity,"expert request identity or geometry is invalid");goto done;}
    out->response_mode=SWARM_EXPERT_RESPONSE_PER_WORKER_FAST;
    if(json_object_find(root,"response_mode",&span)==0&&span.begin<span.end&&*span.begin=='"'){
        char mode[32];if(json_string_copy(span,mode,sizeof(mode))!=0){wire_error(error,error_capacity,"invalid response mode");goto done;}
        if(!strcmp(mode,"per_expert_exact"))out->response_mode=SWARM_EXPERT_RESPONSE_PER_EXPERT_EXACT;
        else if(strcmp(mode,"per_worker_fast")){wire_error(error,error_capacity,"unsupported response mode");goto done;}
    }
    out->execution_mode=SWARM_EXPERT_EXECUTION_WHOLE;
    if(json_object_find(root,"execution_mode",&span)==0){
        char execution[32];
        if(json_string_copy(span,execution,sizeof(execution))!=0){wire_error(error,error_capacity,"invalid execution mode");goto done;}
        if(!strcmp(execution,"microshard"))out->execution_mode=SWARM_EXPERT_EXECUTION_MICROSHARD;
        else if(strcmp(execution,"whole_expert")){wire_error(error,error_capacity,"unsupported execution mode");goto done;}
    }
    json_span hidden_start,hidden_end;
    int has_start=json_object_find(root,"hidden_start",&hidden_start)==0&&!json_is_null(hidden_start);
    int has_end=json_object_find(root,"hidden_end",&hidden_end)==0&&!json_is_null(hidden_end);
    if(out->execution_mode==SWARM_EXPERT_EXECUTION_MICROSHARD){
        if(!has_start||!has_end||json_integer(hidden_start,&out->hidden_start)!=0||
           json_integer(hidden_end,&out->hidden_end)!=0||out->hidden_start<0||
           out->hidden_end<=out->hidden_start){wire_error(error,error_capacity,"microshard hidden range is invalid");goto done;}
    }else if(has_start||has_end){wire_error(error,error_capacity,"whole expert request carries a hidden range");goto done;}
    if(json_object_find(root,"microshard_final",&span)==0){
        if(json_boolean(span,&out->microshard_final)!=0){wire_error(error,error_capacity,"microshard final marker is invalid");goto done;}
    }
    if(json_object_find(root,"challenge",&span)==0){
        if(json_boolean(span,&out->challenge)!=0){wire_error(error,error_capacity,"challenge marker is invalid");goto done;}
    }
    if(out->execution_mode==SWARM_EXPERT_EXECUTION_WHOLE&&out->microshard_final){
        wire_error(error,error_capacity,"whole expert request carries microshard chain state");goto done;}
    json_span per_ids;int per_status=json_object_find(root,"expert_ids_by_row",&per_ids);
    int per_row=per_status==0&&per_ids.begin<per_ids.end&&*per_ids.begin=='[';
    if(per_row){
        if(json_object_find(root,"top_k",&span)!=0||json_integer(span,&out->top_k)!=0||out->top_k<1)
            {wire_error(error,error_capacity,"per-row route requires positive top_k");goto done;}
    }else{
        if(json_object_find(root,"expert_ids",&span)!=0||json_array_count(span,&out->top_k)!=0||out->top_k<1)
            {wire_error(error,error_capacity,"flat route is absent");goto done;}
        json_span declared;if(json_object_find(root,"top_k",&declared)==0&&!json_is_null(declared)){
            int top;if(json_integer(declared,&top)!=0||top!=out->top_k){wire_error(error,error_capacity,"flat top_k mismatch");goto done;}}
    }
    if(out->top_k>(1<<20)||(size_t)out->batch_rows>SIZE_MAX/(size_t)out->top_k){wire_error(error,error_capacity,"expert route matrix is too large");goto done;}
    size_t route_count=(size_t)out->batch_rows*(size_t)out->top_k;
    if(route_count>SWARM_EXPERT_MAX_FRAME_BYTES/12){wire_error(error,error_capacity,"expert route matrix is too large");goto done;}
    out->expert_ids_by_row=(int*)malloc(route_count*sizeof(int));out->routing_weights_by_row=(float*)malloc(route_count*sizeof(float));out->selected_rank_by_row=(int*)malloc(route_count*sizeof(int));
    if(!out->expert_ids_by_row||!out->routing_weights_by_row||!out->selected_rank_by_row){wire_error(error,error_capacity,"out of memory decoding routes");goto done;}
    if(per_row){json_span weights,ranks;
        if(parse_int_matrix(per_ids,out->batch_rows,out->top_k,out->expert_ids_by_row)!=0||
           json_object_find(root,"routing_weights_by_row",&weights)!=0||parse_float_matrix(weights,out->batch_rows,out->top_k,out->routing_weights_by_row)!=0||
           json_object_find(root,"selected_rank_by_row",&ranks)!=0||parse_int_matrix(ranks,out->batch_rows,out->top_k,out->selected_rank_by_row)!=0){wire_error(error,error_capacity,"per-row route matrices are malformed");goto done;}
    }else{json_span ids,weights;int weight_count;
        if(json_object_find(root,"expert_ids",&ids)!=0||copy_flat_ints(ids,out->expert_ids_by_row,out->top_k)!=0||
           json_object_find(root,"routing_weights",&weights)!=0||parse_float_array(weights,out->routing_weights_by_row,out->top_k,&weight_count)!=0||weight_count!=out->top_k){wire_error(error,error_capacity,"flat route vectors are malformed");goto done;}
        for(int rank=0;rank<out->top_k;rank++)out->selected_rank_by_row[rank]=rank;
        for(int row=1;row<out->batch_rows;row++){memcpy(out->expert_ids_by_row+(size_t)row*out->top_k,out->expert_ids_by_row,(size_t)out->top_k*sizeof(int));memcpy(out->routing_weights_by_row+(size_t)row*out->top_k,out->routing_weights_by_row,(size_t)out->top_k*sizeof(float));for(int rank=0;rank<out->top_k;rank++)out->selected_rank_by_row[(size_t)row*out->top_k+rank]=rank;}
    }
    for(size_t i=0;i<route_count;i++)if(out->expert_ids_by_row[i]<0||out->selected_rank_by_row[i]!=(int)(i%(size_t)out->top_k)){wire_error(error,error_capacity,"route IDs or selected ranks are invalid");goto done;}
    json_span metadata;if(json_object_find(root,"activations",&metadata)!=0||*metadata.begin!='{'){wire_error(error,error_capacity,"activation metadata is absent");goto done;}
    char envelope[32],outer_checksum[65];int payload_index;uint64_t encoded_bytes;
#define META_STRING(name,dest) (json_object_find(metadata,name,&span)==0&&json_string_copy(span,dest,sizeof(dest))==0)
    if(!META_STRING("envelope",envelope)||strcmp(envelope,"SWARMT01")||!META_STRING("checksum",outer_checksum)||
       json_object_find(metadata,"payload_index",&span)!=0||json_integer(span,&payload_index)!=0||payload_index!=0||
       json_object_find(metadata,"encoded_bytes",&span)!=0||json_u64(span,&encoded_bytes)!=0||encoded_bytes!=packet.blobs[0].length){wire_error(error,error_capacity,"activation outer metadata is invalid");goto done;}
    char actual_outer[65];sha_hex(packet.blobs[0].data,(size_t)packet.blobs[0].length,actual_outer);if(strcmp(actual_outer,outer_checksum)){wire_error(error,error_capacity,"activation outer checksum mismatch");goto done;}
    if(swarm_expert_wire_decode_tensor_f32(packet.blobs[0].data,(size_t)packet.blobs[0].length,&out->activation,error,error_capacity)!=0)goto done;
    if(strcmp(out->activation.request_id,out->request_id)||out->activation.stage_id!=out->layer_id||
       strcmp(out->activation.model_revision,out->model_revision)||strcmp(out->activation.partition_hash,out->quantization_fingerprint)||
       out->activation.ndim!=2||out->activation.shape[0]!=out->batch_rows||out->activation.shape[1]!=out->latent_dimension||
       out->activation.sequence_length!=out->batch_rows){wire_error(error,error_capacity,"activation tensor identity does not match expert request");goto done;}
    json_span accumulator_meta;
    int accumulator_status=json_object_find(root,"down_accumulators",&accumulator_meta);
    int needs_accumulator=out->execution_mode==SWARM_EXPERT_EXECUTION_MICROSHARD&&
        out->response_mode==SWARM_EXPERT_RESPONSE_PER_EXPERT_EXACT&&out->hidden_start>0;
    if(out->execution_mode==SWARM_EXPERT_EXECUTION_MICROSHARD&&
       out->response_mode==SWARM_EXPERT_RESPONSE_PER_WORKER_FAST&&out->microshard_final){
        wire_error(error,error_capacity,"fast microshard carries exact chain state");goto done;}
    if(needs_accumulator){
        if(packet.blob_count!=2||accumulator_status!=0||json_is_null(accumulator_meta)||
           *accumulator_meta.begin!='{'){
            wire_error(error,error_capacity,"non-initial microshard has no down accumulator");goto done;}
        char accumulator_envelope[32],accumulator_checksum[65];int accumulator_index;
        uint64_t accumulator_encoded;
        if(json_object_find(accumulator_meta,"envelope",&span)!=0||
           json_string_copy(span,accumulator_envelope,sizeof(accumulator_envelope))!=0||
           strcmp(accumulator_envelope,"SWARMT01")||
           json_object_find(accumulator_meta,"checksum",&span)!=0||
           json_string_copy(span,accumulator_checksum,sizeof(accumulator_checksum))!=0||
           json_object_find(accumulator_meta,"payload_index",&span)!=0||
           json_integer(span,&accumulator_index)!=0||accumulator_index!=1||
           json_object_find(accumulator_meta,"encoded_bytes",&span)!=0||
           json_u64(span,&accumulator_encoded)!=0||accumulator_encoded!=packet.blobs[1].length){
            wire_error(error,error_capacity,"down accumulator outer metadata is invalid");goto done;}
        char accumulator_actual[65];
        sha_hex(packet.blobs[1].data,(size_t)packet.blobs[1].length,accumulator_actual);
        if(strcmp(accumulator_actual,accumulator_checksum)){
            wire_error(error,error_capacity,"down accumulator outer checksum mismatch");goto done;}
        if(swarm_expert_wire_decode_tensor_f32(packet.blobs[1].data,
                (size_t)packet.blobs[1].length,&out->down_accumulators,
                error,error_capacity)!=0)goto done;
        if(strcmp(out->down_accumulators.request_id,out->request_id)||
           out->down_accumulators.stage_id!=out->layer_id||
           strcmp(out->down_accumulators.model_revision,out->model_revision)||
           strcmp(out->down_accumulators.partition_hash,out->quantization_fingerprint)||
           out->down_accumulators.ndim!=3||
           out->down_accumulators.shape[0]!=out->batch_rows||
           out->down_accumulators.shape[1]!=out->top_k||
           out->down_accumulators.shape[2]!=out->latent_dimension||
           out->down_accumulators.sequence_length!=out->batch_rows){
            wire_error(error,error_capacity,"down accumulator identity does not match expert request");goto done;}
    }else if(packet.blob_count!=1||(accumulator_status==0&&!json_is_null(accumulator_meta))){
        wire_error(error,error_capacity,"initial or whole expert request carries a down accumulator");goto done;
    }
    rc=0;
done:
#undef REQUEST_STRING
#undef META_STRING
    swarm_expert_wire_free_packet(&packet);if(rc!=0)swarm_expert_wire_free_route_request(out);return rc;
}

int swarm_expert_wire_decode_route_response(const uint8_t *payload,size_t payload_length,
        swarm_expert_route_response *out,char *error,size_t error_capacity) {
    if(!out)return wire_error(error,error_capacity,"null route response output");
    memset(out,0,sizeof(*out));
    swarm_expert_packet packet;
    if(swarm_expert_wire_decode_packet(payload,payload_length,&packet,error,error_capacity)!=0)return -1;
    int rc=-1;json_span root,span,result_meta,integrity,execution;
    if(strcmp(packet.kind,"response")||packet.blob_count!=1||
       json_complete(packet.semantic_json,&root)!=0||*root.begin!='{'){
        wire_error(error,error_capacity,"expected one-blob expert response");goto done;}
#define RESPONSE_STRING(object,name,dest) (json_object_find(object,name,&span)==0&&json_string_copy(span,dest,sizeof(dest))==0)
    if(!RESPONSE_STRING(root,"request_id",out->request_id)||
       !RESPONSE_STRING(root,"worker_id",out->worker_id)||
       !RESPONSE_STRING(root,"model_revision",out->model_revision)||
       json_object_find(root,"layer_id",&span)!=0||json_integer(span,&out->layer_id)!=0||out->layer_id<0){
        wire_error(error,error_capacity,"expert response identity is invalid");goto done;}
    strcpy(out->status,"ok");
    if(json_object_find(root,"status",&span)==0&&json_string_copy(span,out->status,sizeof(out->status))!=0){
        wire_error(error,error_capacity,"expert response status is invalid");goto done;}
    if(strcmp(out->status,"ok")&&strcmp(out->status,"error")){
        wire_error(error,error_capacity,"expert response has unsupported status");goto done;}
    if(!strcmp(out->status,"error")){
        if(json_object_find(root,"error",&span)!=0||json_string_copy(span,out->error,sizeof(out->error))!=0){
            wire_error(error,error_capacity,"error response has no reason");goto done;}
        rc=0;goto done;
    }
    if(json_object_find(root,"result",&result_meta)!=0||*result_meta.begin!='{'||
       json_object_find(root,"integrity",&integrity)!=0||*integrity.begin!='{'||
       json_object_find(root,"execution_metadata",&execution)!=0||*execution.begin!='{'){
        wire_error(error,error_capacity,"expert response metadata objects are absent");goto done;}
    char envelope[32],dtype[32],outer_checksum[80];uint64_t payload_index,encoded_bytes,raw_bytes;
    if(!RESPONSE_STRING(result_meta,"envelope",envelope)||strcmp(envelope,"SWARMT01")||
       !RESPONSE_STRING(result_meta,"dtype",dtype)||strcmp(dtype,"float32")||
       !RESPONSE_STRING(result_meta,"checksum",outer_checksum)||
       json_object_find(result_meta,"payload_index",&span)!=0||json_u64(span,&payload_index)!=0||payload_index!=0||
       json_object_find(result_meta,"encoded_bytes",&span)!=0||json_u64(span,&encoded_bytes)!=0||encoded_bytes!=packet.blobs[0].length||
       json_object_find(result_meta,"raw_bytes",&span)!=0||json_u64(span,&raw_bytes)!=0){
        wire_error(error,error_capacity,"expert result tensor metadata is invalid");goto done;}
    char actual_outer[65];sha_hex(packet.blobs[0].data,(size_t)packet.blobs[0].length,actual_outer);
    if(strcmp(actual_outer,outer_checksum)){
        wire_error(error,error_capacity,"expert result outer checksum mismatch");goto done;}
    if(swarm_expert_wire_decode_tensor_f32(packet.blobs[0].data,(size_t)packet.blobs[0].length,
            &out->result,error,error_capacity)!=0)goto done;
    if(out->result.raw_length!=raw_bytes||strcmp(out->result.request_id,out->request_id)||
       out->result.stage_id!=out->layer_id||strcmp(out->result.model_revision,out->model_revision)){
        wire_error(error,error_capacity,"expert result tensor identity differs from response");goto done;}
    if(json_object_find(result_meta,"shape",&span)!=0){wire_error(error,error_capacity,"expert result shape is absent");goto done;}
    int64_t declared_shape[SWARM_EXPERT_MAX_DIMS];int declared_dims=0;
    if(parse_int_array(span,declared_shape,SWARM_EXPERT_MAX_DIMS,&declared_dims)!=0||
       declared_dims!=out->result.ndim){wire_error(error,error_capacity,"expert result shape is invalid");goto done;}
    for(int i=0;i<declared_dims;i++)if(declared_shape[i]!=out->result.shape[i]){
        wire_error(error,error_capacity,"expert result shape differs from tensor envelope");goto done;}
    if(!RESPONSE_STRING(integrity,"model_fingerprint",out->model_fingerprint)||
       !RESPONSE_STRING(integrity,"result_hash",out->result_hash)||
       strcmp(out->result.partition_hash,out->model_fingerprint)){
        wire_error(error,error_capacity,"expert result integrity identity is invalid");goto done;}
    char raw_hash[65];sha_hex(out->result.raw,out->result.raw_length,raw_hash);
    const char *recorded_hash=out->result_hash;
    if(!strncmp(recorded_hash,"sha256:",7))recorded_hash+=7;
    if(strcmp(recorded_hash,raw_hash)){
        wire_error(error,error_capacity,"expert result raw hash mismatch");goto done;}
#define RESPONSE_U64(name,dest) do{if(json_object_find(execution,name,&span)!=0||json_u64(span,&dest)!=0){wire_error(error,error_capacity,"expert execution metadata is invalid");goto done;}}while(0)
    RESPONSE_U64("bytes_read",out->bytes_read);
    RESPONSE_U64("bytes_received",out->bytes_received);
    RESPONSE_U64("bytes_sent",out->bytes_sent);
    RESPONSE_U64("compute_ns",out->compute_ns);
    RESPONSE_U64("queue_ns",out->queue_ns);
    RESPONSE_U64("transfer_ns",out->transfer_ns);
#undef RESPONSE_U64
    rc=0;
done:
#undef RESPONSE_STRING
    swarm_expert_wire_free_packet(&packet);return rc;
}
