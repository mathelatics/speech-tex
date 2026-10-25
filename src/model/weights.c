/* weights.c — materialize ggml weight tensors from a parsed GGUF file.
 * Types are preserved exactly (Q4_0/Q8_0/F32...): no conversion on load, so
 * RAM footprint == quantized size and startup is one memcpy pass per tensor.
 * GPU offload (Phase 4) will move these to backend buffers via
 * ggml_backend_dev_cpy_tensor; the CPU path here needs no changes for that. */
#include "model.h"
#include "../io/gguf_loader.h"
#include "../core/phonos_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* lookup index or -1 */
static int tidx(const struct gguf_file* g, const char* name) {
    for (size_t i = 0; i < g->n_tensors; i++)
        if (strcmp(g->tensors[i].name, name) == 0) return (int)i;
    return -1;
}

/* create tensor of same type/shape as GGUF entry idx; payload written into the
 * contiguous weight buffer (native types preserved, zero dequant on load) */
static struct ggml_tensor* dup_tensor(struct ggml_context* ctx,
                                      const struct gguf_file* g, int idx,
                                      struct ph_weights* w) {
    const struct gguf_tensor_info* ti = &g->tensors[idx];
    int64_t ne[4] = {1,1,1,1};
    for (uint32_t d = 0; d < ti->n_dims && d < 4; d++) ne[d] = (int64_t)ti->ne[d];
    struct ggml_tensor* t = ggml_new_tensor(ctx, ti->type, (int)ti->n_dims, ne);
    if (!t) return NULL;
    ggml_set_name(t, ti->name);

    /* 64B-align each tensor within the arena (quant blocks like F32 rows want it) */
    size_t off = (w->buf_cursor + 63) & ~(size_t)63;
    if (off + ti->size > ggml_backend_buffer_get_size(w->buf)) {
        phonos_log(PHONOS_LOG_ERROR, "weights: arena overflow at '%s'", ti->name);
        return NULL;
    }
    if (ggml_backend_tensor_alloc(w->buf, t, (char*)ggml_backend_buffer_get_base(w->buf) + off)
        != GGML_STATUS_SUCCESS) {
        phonos_log(PHONOS_LOG_ERROR, "weights: tensor_alloc failed for '%s'", ti->name);
        return NULL;
    }
    ggml_backend_buffer_write(w->buf, g->weights_base + ti->offset, off, ti->size);
    w->buf_cursor = off + ti->size;
    return t;
}

#define LOAD(dst_field, NAME) do { \
    int _i = tidx(g, (NAME)); \
    if (_i < 0) { phonos_log(PHONOS_LOG_ERROR, "weights: missing tensor '%s'", (NAME)); return -1; } \
    dst_field = dup_tensor(w->ctx_data, g, _i, w); \
    if (!(dst_field)) return -1; \
} while (0)

/* Weight storage strategy (PLAN.md §quantization-first):
 *   - ONE contiguous host buffer (ggml_backend_cpu_buffer) holds every tensor
 *     payload exactly as it appears in the GGUF file (native quant types stay
 *     native — zero dequant on load).
 *   - Tensor structs live in a no_alloc ggml_context; their ->data points into
 *     the buffer. On CPU this is zero-copy; Phase-4 GPU offload will instead
 *     allocate a device buffer and async-copy each tensor once at load time.
 * This gives us: deterministic RSS, one free() at teardown, and a single place
 * to swap host<->device placement without touching any model code.          */
int ph_weights_load(struct gguf_file* g, const struct ph_hparams* hp, struct ph_weights* w) {
    memset(w, 0, sizeof(*w));

    size_t total = 0;
    for (size_t i = 0; i < g->n_tensors; i++) total += g->tensors[i].size;
    w->total_bytes = total;

    /* contiguous host buffer; tensor payloads are written into it below */
    const size_t align = 64;
    const size_t buf_size = (total + align - 1) & ~(align - 1);
    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) { phonos_log(PHONOS_LOG_ERROR, "weights: no CPU device registered"); return -1; }
    w->buf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(cpu_dev), buf_size);
    if (!w->buf) {
        phonos_log(PHONOS_LOG_ERROR, "weights: buffer alloc (%zu B) failed", buf_size);
        return -1;
    }
    w->buf_cursor = 0;

    /* tensor structs only — data pointers assigned per tensor below */
    const size_t overhead = g->n_tensors * 96 + 4096;
    struct ggml_init_params p = { .mem_size = overhead, .no_alloc = true };
    w->ctx_data = ggml_init(p);
    if (!w->ctx_data) {
        phonos_log(PHONOS_LOG_ERROR, "weights: ggml_init(%zu B) failed", p.mem_size);
        return -1;
    }

    LOAD(w->tok_emb,    "token_embd.weight");
    LOAD(w->out_norm_w, "output_norm.weight");
    LOAD(w->out_norm_b, "output_norm.bias");
    LOAD(w->lm_head,    "output.weight");

    w->attn_norm_w = calloc(hp->n_layers, sizeof(struct ggml_tensor*));
    w->attn_norm_b = calloc(hp->n_layers, sizeof(struct ggml_tensor*));
    w->wq = calloc(hp->n_layers, sizeof(struct ggml_tensor*));
    w->wk = calloc(hp->n_layers, sizeof(struct ggml_tensor*));
    w->wv = calloc(hp->n_layers, sizeof(struct ggml_tensor*));
    w->wo = calloc(hp->n_layers, sizeof(struct ggml_tensor*));
    w->ffn_norm_w = calloc(hp->n_layers, sizeof(struct ggml_tensor*));
    w->ffn_norm_b = calloc(hp->n_layers, sizeof(struct ggml_tensor*));
    w->w1 = calloc(hp->n_layers, sizeof(struct ggml_tensor*));
    w->w2 = calloc(hp->n_layers, sizeof(struct ggml_tensor*));
    if (!w->attn_norm_w || !w->w2) return -1;

    char nm[96];
    for (int L = 0; L < hp->n_layers; L++) {
        struct ggml_tensor** slots[10] = {
            &w->attn_norm_w[L], &w->attn_norm_b[L], &w->wq[L], &w->wk[L], &w->wv[L],
            &w->wo[L], &w->ffn_norm_w[L], &w->ffn_norm_b[L], &w->w1[L], &w->w2[L] };
        const char* suffixes[10] = {
            "attn_norm.weight", "attn_norm.bias", "attn_q.weight", "attn_k.weight", "attn_v.weight",
            "attn_o.weight", "ffn_norm.weight", "ffn_norm.bias", "ffn_up.weight", "ffn_down.weight" };
        for (int s = 0; s < 10; s++) {
            snprintf(nm, sizeof nm, "blk.%d.%s", L, suffixes[s]);
            LOAD(*slots[s], nm);
        }
    }

    if (hp->has_speech) {
        LOAD(w->enc_proj_w, "enc_proj.weight");
        LOAD(w->enc_proj_b, "enc_proj.bias");
        LOAD(w->cond_w,     "cond_proj.weight");
        LOAD(w->cond_b,     "cond_proj.bias");
    }
#undef LOAD

    phonos_log(PHONOS_LOG_INFO, "weights: %zu tensors, %.2f MiB ingested",
               g->n_tensors, (double)total/1048576.0);
    return 0;
}

void ph_weights_free(struct ph_weights* w) {
    if (!w) return;
    if (w->ctx_data) ggml_free(w->ctx_data);
    if (w->buf) ggml_backend_buffer_free(w->buf);
    free(w->attn_norm_w); free(w->attn_norm_b);
    free(w->wq); free(w->wk); free(w->wv); free(w->wo);
    free(w->ffn_norm_w); free(w->ffn_norm_b);
    free(w->w1); free(w->w2);
    memset(w, 0, sizeof(*w));
}

/* declarations for functions implemented in graph_text.c / graph_speech.c */
int  ph_hparams_read(struct gguf_file* g, struct ph_hparams* hp);
void ph_vocab_load(struct gguf_file* g, struct ph_hparams* hp, struct ph_vocab* v);
void ph_vocab_free(struct ph_vocab* v);
int  ph_tokenize(const struct ph_vocab* v, const char* text, int* ids, int max_n);
const char* ph_detokenize_piece(const struct ph_vocab* v, int id, char* buf, size_t bufsz);

