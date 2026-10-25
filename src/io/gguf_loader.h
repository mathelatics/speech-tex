/* gguf_loader.h — minimal GGUF v2/v3 reader (metadata + tensor registry).
 * Internal header; not part of the public API. */
#ifndef PHONOS_GGUF_LOADER_H
#define PHONOS_GGUF_LOADER_H

#include <ggml.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ph_gguf_type {
    PH_GGUF_TYPE_UINT8   = 0,
    PH_GGUF_TYPE_INT8    = 1,
    PH_GGUF_TYPE_UINT16  = 2,
    PH_GGUF_TYPE_INT16   = 3,
    PH_GGUF_TYPE_UINT32  = 4,
    PH_GGUF_TYPE_INT32   = 5,
    PH_GGUF_TYPE_FLOAT32 = 6,
    PH_GGUF_TYPE_BOOL    = 7,
    PH_GGUF_TYPE_STRING  = 8,
    PH_GGUF_TYPE_ARRAY   = 9,
    PH_GGUF_TYPE_UINT64  = 10,
    PH_GGUF_TYPE_INT64   = 11,
    PH_GGUF_TYPE_FLOAT64 = 12,
    PH_GGUF_TYPE_COUNT   = 13,
};

static inline size_t gguf_type_size(enum ph_gguf_type t) {
    static const size_t sz[PH_GGUF_TYPE_COUNT] = {1,1,2,2,4,4,4,4,sizeof(void*),0,8,8,8};
    return t < PH_GGUF_TYPE_COUNT ? sz[t] : 0;
}

struct ph_gguf_array;  /* defined in .c; opaque here except via accessors */

struct gguf_kv {
    const char*     key;
    enum ph_gguf_type  type;
    size_t          n;      /* scalar: 1; array: 1 (payload in gguf_array) */
    void*           data;
};

struct gguf_tensor_info {
    const char*    name;
    uint32_t       n_dims;
    uint64_t       ne[4];        /* ggml order: fastest dim first */
    enum ggml_type type;
    uint64_t       offset;       /* relative to data_start */
    size_t         size;         /* bytes */
};

struct gguf_file {
    uint32_t version;
    size_t   n_tensors_total;

    struct gguf_kv* kvs;
    size_t          n_kv, cap_kv;

    struct gguf_tensor_info* tensors;
    size_t                   n_tensors;

    uint64_t          data_start;
    const uint8_t*    weights_base;   /* points at data_start region */
    uint64_t          weights_len;
    int               weights_mmapped;
    int               fd;             /* unix mmap handle */
#ifdef _WIN32
    void*             hFile;
    void*             hMap;
#endif
};

/* prefer_mmap: fall back to malloc+read when unsupported. Returns NULL on error. */
struct gguf_file* gguf_file_load(const char* path, int prefer_mmap);
void              gguf_file_free(struct gguf_file* g);

const struct gguf_kv*    gguf_find(const struct gguf_file* g, const char* key);
const char*              gguf_get_str(const struct gguf_file* g, const char* key);
int64_t                  gguf_get_int(const struct gguf_file* g, const char* key, int64_t dflt);
float                    gguf_get_f32(const struct gguf_file* g, const char* key, float dflt);
const struct ph_gguf_array* gguf_get_arr(const struct gguf_file* g, const char* key);

/* array internals exposed to model code (tokenizer vocab etc.) */
struct ph_gguf_array {
    enum ph_gguf_type elem_type;
    uint64_t n;
    void* elems;      /* n * size(elem_type); for STRING: n char* */
};

#ifdef __cplusplus
}
#endif

#endif /* PHONOS_GGUF_LOADER_H */
