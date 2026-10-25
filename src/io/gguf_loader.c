/* gguf_loader.c — GGUF container reader: metadata KV + tensor registry.
 * Deliberately dumb: it does NOT interpret arch semantics (that's model/).
 * Supports mmap or plain-read weight ingest; all quantized types pass through
 * as native ggml tensors (no dequantize-on-load — that would kill perf). */
#include "gguf_loader.h"
#include "../core/phonos_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#  include <fcntl.h>
#endif

#define GGUF_MAGIC   0x46554747u  /* "GGUF" little-endian */
#define GGUF_V_MIN   2
#define GGUF_V_MAX   3

/* array payload: parsed into a flat buffer of element values; only scalar
 * element types supported here (strings-in-arrays handled for tokenizer). */
struct gguf_array {
    enum gguf_type elem_type;
    uint64_t n;
    void* elems;      /* n * size(elem_type); for STRING: n char* */
};

/* ---------------- aligned file reader ----------------------------------- */
struct reader {
    FILE*  f;
    long   offset;      /* tracked for padding math */
    int    eof_error;
};

static size_t rd(struct reader* r, void* dst, size_t n) {
    size_t got = fread(dst, 1, n, r->f);
    r->offset += (long)got;
    if (got != n) r->eof_error = 1;
    return got;
}

static void pad_to(struct reader* r, size_t align) {
    long m = r->offset % (long)align;
    if (m) {
        char tmp[64];
        size_t need = align - (size_t)m;
        while (need) { size_t c = need > 64 ? 64 : need; rd(r, tmp, c); need -= c; }
    }
}

/* ---------------- kv parsing -------------------------------------------- */
static uint32_t rd_u32(struct reader* r) { uint32_t v=0; rd(r,&v,4); return v; }
static uint64_t rd_u64(struct reader* r) { uint64_t v=0; rd(r,&v,8); return v; }

static char* rd_str(struct reader* r) {
    uint64_t len = rd_u64(r);
    if (len > (1u<<24)) return NULL;         /* absurd string => corrupt */
    char* s = (char*)malloc(len + 1);
    if (!s) return NULL;
    if (len && rd(r, s, (size_t)len) != len) { free(s); return NULL; }
    s[len] = '\0';
    return s;
}

static void kv_add(struct gguf_file* g, const char* key, enum gguf_type t,
                   const void* data, size_t n_elems) {
    if (g->n_kv >= g->cap_kv) {
        size_t nc = g->cap_kv ? g->cap_kv*2 : 64;
        struct gguf_kv* p = realloc(g->kvs, nc * sizeof(*p));
        if (!p) { free((void*)key); return; }  /* best-effort: drop entry */
        g->kvs = p; g->cap_kv = nc;
    }
    struct gguf_kv* e = &g->kvs[g->n_kv++];
    e->key = key; e->type = t; e->n = n_elems;
    e->data = NULL;
    switch (t) {
        case GGUF_TYPE_STRING: e->data = (void*)*((const char**)data); break; /* already malloc'd */
        case GGUF_TYPE_ARRAY:  e->data = (void*)data; break;                  /* ownership transferred */
        default: {
            size_t sz = gguf_type_size(t) * (n_elems ? n_elems : 1);
            void* buf = malloc(sz);
            if (buf) memcpy(buf, data, sz);
            e->data = buf;
        }
    }
}

/* array payload access via struct gguf_array (declared in header) */

static int rd_array(struct reader* r, struct gguf_array* out) {
    uint32_t t = rd_u32(r);
    uint64_t n = rd_u64(r);
    if (t >= GGUF_TYPE_COUNT) return 0;
    size_t esz = gguf_type_size((enum gguf_type)t);
    if (n > 0 && (uint64_t)(SIZE_MAX/esz - 1) < n) return 0;   /* overflow guard */
    out->elem_type = (enum gguf_type)t;
    out->n = n;
    out->elems = calloc(n ? n : 1, esz ? esz : 1);
    if (!out->elems) return 0;
    for (uint64_t i = 0; i < n; i++) {
        switch ((enum gguf_type)t) {
            case GGUF_TYPE_UINT8:   rd(r, (uint8_t*)out->elems + i,   1); break;
            case GGUF_TYPE_INT8:    rd(r, (int8_t*)out->elems + i,    1); break;
            case GGUF_TYPE_UINT16:  rd(r, (uint16_t*)out->elems + i,  2); break;
            case GGUF_TYPE_INT16:   rd(r, (int16_t*)out->elems + i,   2); break;
            case GGUF_TYPE_UINT32:  rd(r, (uint32_t*)out->elems + i,  4); break;
            case GGUF_TYPE_INT32:   rd(r, (int32_t*)out->elems + i,   4); break;
            case GGUF_TYPE_FLOAT32: rd(r, (float*)out->elems + i,     4); break;
            case GGUF_TYPE_BOOL:    rd(r, (int32_t*)out->elems + i,   4); break;
            case GGUF_TYPE_UINT64:  rd(r, (uint64_t*)out->elems + i,  8); break;
            case GGUF_TYPE_INT64:   rd(r, (int64_t*)out->elems + i,   8); break;
            case GGUF_TYPE_FLOAT64: rd(r, (double*)out->elems + i,    8); break;
            case GGUF_TYPE_STRING: {
                char* s = rd_str(r);
                if (!s) { free(out->elems); out->elems=NULL; return 0; }
                ((char**)out->elems)[i] = s;
                break;
            }
            default: free(out->elems); out->elems = NULL; return 0; /* nested arrays unsupported */
        }
    }
    return 1;
}

/* ---------------- public API --------------------------------------------- */
void gguf_file_free(struct gguf_file* g) {
    if (!g) return;
    for (size_t i = 0; i < g->n_kv; i++) {
        free((void*)g->kvs[i].key);
        if (g->kvs[i].type == GGUF_TYPE_ARRAY && g->kvs[i].data) {
            struct gguf_array* a = (struct gguf_array*)g->kvs[i].data;
            if (a->elem_type == GGUF_TYPE_STRING && a->elems) {
                for (uint64_t j = 0; j < a->n; j++) free(((char**)a->elems)[j]);
            }
            free(a->elems); free(a);
        } else {
            free(g->kvs[i].data);
        }
    }
    free(g->kvs);
    for (size_t i = 0; i < g->n_tensors; i++) free((void*)g->tensors[i].name);
    free(g->tensors);
    if (g->weights_mmapped) {
#ifdef _WIN32
        UnmapViewOfFile(g->weights_base); CloseHandle(g->hMap); CloseHandle(g->hFile);
#else
        /* whole-file map; weights_base is offset into it — unmap from page base */
        uintptr_t p = (uintptr_t)g->weights_base - (uintptr_t)g->data_start;
        munmap((void*)p, (size_t)g->weights_len + g->data_start); close(g->fd);
#endif
    } else {
        free((void*)g->weights_base);
    }
    free(g);
}

struct gguf_file* gguf_file_load(const char* path, int prefer_mmap) {
    struct gguf_file* g = calloc(1, sizeof(*g));
    if (!g) return NULL;

    struct reader r = {0};
    r.f = fopen(path, "rb");
    if (!r.f) { phonos_log(PHONOS_LOG_ERROR, "gguf: cannot open %s", path); free(g); return NULL; }
    r.offset = 0;

    uint32_t magic = rd_u32(&r);
    uint32_t ver   = rd_u32(&r);
    if (magic != GGUF_MAGIC || ver < GGUF_V_MIN || ver > GGUF_V_MAX) {
        phonos_log(PHONOS_LOG_ERROR, "gguf: bad magic 0x%08x or version %u", magic, ver);
        fclose(r.f); free(g); return NULL;
    }
    g->version = ver;
    g->n_tensors_total = (size_t)rd_u64(&r);

    uint64_t nkv = rd_u64(&r);
    if (nkv > (1u<<20)) { fclose(r.f); free(g); return NULL; }   /* sanity */

    for (uint64_t i = 0; i < nkv && !r.eof_error; i++) {
        char* key = rd_str(&r);
        if (!key) break;
        uint32_t t = rd_u32(&r);
        if (t >= GGUF_TYPE_COUNT) { free(key); break; }
        char stackval[8];
        if ((enum gguf_type)t == GGUF_TYPE_STRING) {
            char* s = rd_str(&r);
            if (!s) { free(key); break; }
            kv_add(g, key, GGUF_TYPE_STRING, &s, 1);
        } else if ((enum gguf_type)t == GGUF_TYPE_ARRAY) {
            struct gguf_array* a = calloc(1, sizeof(*a));
            if (!a || !rd_array(&r, a)) { free(a); free(key); break; }
            kv_add(g, key, GGUF_TYPE_ARRAY, a, 1); /* transfers ownership */
        } else {
            size_t sz = gguf_type_size((enum gguf_type)t);
            if (sz <= 8) { memset(stackval,0,8); rd(&r, stackval, sz); kv_add(g, key, (enum gguf_type)t, stackval, 1); }
            else { void* buf = malloc(sz); if (!buf) { free(key); break; } rd(&r, buf, sz); kv_add(g, key, (enum gguf_type)t, buf, 1); free(buf); }
        }
    }

    /* ---- tensor infos ---- */
    g->tensors = calloc(g->n_tensors_total, sizeof(*g->tensors));
    if (!g->tensors) { fclose(r.f); gguf_file_free(g); return NULL; }
    size_t max_align = 32;
    const char* al = gguf_get_str(g, "general.alignment");
    if (al) { unsigned a = (unsigned)atoi(al); if (a && (a & (a-1))==0) max_align = a; }

    uint64_t min_off = UINT64_MAX;
    for (size_t i = 0; i < g->n_tensors_total && !r.eof_error; i++) {
        struct gguf_tensor_info* ti = &g->tensors[g->n_tensors++];
        ti->name = rd_str(&r);
        uint32_t nd = rd_u32(&r);
        if (nd > 4) { phonos_log(PHONOS_LOG_ERROR,"gguf: tensor %s rank %u too high", ti->name?ti->name:"?", nd); goto fail; }
        ti->n_dims = nd;
        for (uint32_t d = 0; d < nd; d++) ti->ne[d] = rd_u64(&r);
        ti->type = (enum ggml_type)rd_u32(&r);
        ti->offset = rd_u64(&r);
        if (ti->offset < min_off) min_off = ti->offset;
        size_t row = ggml_row_size(ti->type, ti->n_dims ? ti->ne[0] : 1);
        for (uint32_t d = 1; d < nd; d++) row *= (size_t)ti->ne[d];
        ti->size = row;
    }
    if (r.eof_error) goto fail;

    pad_to(&r, max_align);
    g->data_start = (uint64_t)r.offset;
    fclose(r.f); r.f = NULL;

    /* ---- weights region ---- */
    uint64_t wbytes = 0;
    for (size_t i = 0; i < g->n_tensors; i++) {
        uint64_t end = g->tensors[i].offset + g->tensors[i].size;
        if (end > wbytes) wbytes = end;
    }
    g->weights_len = wbytes;

    if (prefer_mmap) {
#ifdef _WIN32
        HANDLE h = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (h != INVALID_HANDLE_VALUE) {
            LARGE_INTEGER szli; GetFileSizeEx(h, &szli);
            HANDLE m = CreateFileMappingA(h, NULL, PAGE_READONLY, 0, 0, NULL);
            if (m) {
                void* base = MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
                if (base) {
                    g->weights_base = (const uint8_t*)base + g->data_start;
                    g->weights_mmapped = 1; g->hFile = h; g->hMap = m;
                } else { CloseHandle(m); CloseHandle(h); }
            } else CloseHandle(h);
        }
#else
        int fd = open(path, O_RDONLY);
        if (fd >= 0) {
            struct stat st;
            if (fstat(fd, &st) == 0 && (uint64_t)st.st_size >= g->data_start + wbytes) {
                void* base = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
                if (base != MAP_FAILED) {
                    g->weights_base = (const uint8_t*)base + g->data_start;
                    g->weights_mmapped = 1; g->fd = fd;
                } else close(fd);
            } else close(fd);
        }
#endif
    }

    if (!g->weights_mmapped) {
        FILE* f = fopen(path, "rb");
        if (!f) goto fail_nofclose;
        fseek(f, (long)g->data_start, SEEK_SET);
        void* buf = malloc((size_t)wbytes + 64);
        if (!buf) { fclose(f); goto fail_nofclose; }
        size_t got = fread(buf, 1, (size_t)wbytes, f);
        fclose(f);
        if (got != (size_t)wbytes) { free(buf); phonos_log(PHONOS_LOG_ERROR, "gguf: truncated weight section"); free(g); return NULL; }
        g->weights_base = (const uint8_t*)buf;
    }

    phonos_log(PHONOS_LOG_INFO, "gguf: loaded %s — %zu tensors, %.1f MiB weights (%s)",
               path, g->n_tensors, (double)wbytes/1048576.0,
               g->weights_mmapped ? "mmap" : "read");
    return g;

fail:
    if (r.f) fclose(r.f);
fail_nofclose:
    gguf_file_free(g);
    return NULL;
}

/* ---------------- kv accessors ------------------------------------------- */
const struct gguf_kv* gguf_find(const struct gguf_file* g, const char* key) {
    for (size_t i = 0; i < g->n_kv; i++)
        if (strcmp(g->kvs[i].key, key) == 0) return &g->kvs[i];
    return NULL;
}

const char* gguf_get_str(const struct gguf_file* g, const char* key) {
    const struct gguf_kv* k = gguf_find(g, key);
    return (k && k->type == GGUF_TYPE_STRING) ? (const char*)k->data : NULL;
}

int64_t gguf_get_int(const struct gguf_file* g, const char* key, int64_t dflt) {
    const struct gguf_kv* k = gguf_find(g, key);
    if (!k || !k->data) return dflt;
    switch (k->type) {
        case GGUF_TYPE_UINT8:  return *(uint8_t*)k->data;
        case GGUF_TYPE_INT8:   return *(int8_t*)k->data;
        case GGUF_TYPE_UINT16: return *(uint16_t*)k->data;
        case GGUF_TYPE_INT16:  return *(int16_t*)k->data;
        case GGUF_TYPE_UINT32: return *(uint32_t*)k->data;
        case GGUF_TYPE_INT32:  return *(int32_t*)k->data;
        case GGUF_TYPE_UINT64: return (int64_t)*(uint64_t*)k->data;
        case GGUF_TYPE_INT64:  return *(int64_t*)k->data;
        case GGUF_TYPE_FLOAT32: return (int64_t)*(float*)k->data;
        case GGUF_TYPE_BOOL:   return *(int32_t*)k->data;
        default: return dflt;
    }
}

float gguf_get_f32(const struct gguf_file* g, const char* key, float dflt) {
    const struct gguf_kv* k = gguf_find(g, key);
    if (!k || !k->data) return dflt;
    if (k->type == GGUF_TYPE_FLOAT32) return *(float*)k->data;
    if (k->type == GGUF_TYPE_FLOAT64) return (float)*(double*)k->data;
    return dflt;
}

const struct gguf_array* gguf_get_arr(const struct gguf_file* g, const char* key) {
    const struct gguf_kv* k = gguf_find(g, key);
    return (k && k->type == GGUF_TYPE_ARRAY) ? (const struct gguf_array*)k->data : NULL;
}
