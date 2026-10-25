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

/* create tensor of same type/shape as GGUF entry idx and copy payload in */
static struct ggml_tensor* dup_tensor(struct ggml_context* ctx,
                                      const struct gguf_file* g, int idx) {
    const struct gguf_tensor_info* ti = &g->tensors[idx];
    int64_t ne[4] = {1,1,1,1};
    for (uint32_t d = 0; d < ti->n_dims && d < 4; d++) ne[d] = (int64_t)ti->ne[d];
    struct ggml_tensor* t = ggml_new_tensor(ctx, ti->type, (int)ti->n_dims, ne);
    if (!t) return NULL;
    ggml_set_name(t, ti->name);
    memcpy(t->data, g->weights_base + ti->offset, ti->size);
    return t;
}

#define LOAD(dst_field, NAME) do { \
    int _i = tidx(g, (NAME)); \
    if (_i < 0) { phonos_log(PHONOS_LOG_ERROR, "weights: missing tensor '%s'", (NAME)); return -1; } \
    dst_field = dup_tensor(w->ctx_data, g, _i); \
    if (!(dst_field)) return -1; \
} while (0)

int ph_weights_load(struct gguf_file* g, const struct ph_hparams* hp, struct ph_weights* w) {
    memset(w, 0, sizeof(*w));

    size_t total = 0;
    for (size_t i = 0; i < g->n_tensors; i++) total += g->tensors[i].size;
    w->total_bytes = total;

    /* per-tensor overhead: ggml_object header + tensor struct alignment (~96B safe) */
    const size_t overhead = g->n_tensors * 96 + 4096;
    struct ggml_init_params p = { .mem_size = total + overhead, .no_alloc = false };
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

