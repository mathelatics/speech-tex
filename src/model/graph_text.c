/* graph_text.c — compute-graph builder for the phonon2 text decoder.
 *
 * Layout conventions (ggml column-major ne[] order):
 *   weights : w_proj [out, in]  -> ggml_mul_mat(w, x) with x [in, n_tokens]
 *   cache   : K/V per layer     [n_ctx, HD]           (MQA: one KV head;
 *             dim0 = keys so softmax normalizes over dim0)
 *   scores  : [n_ctx, n_tokens] (keys along dim0)
 *
 * Performance rules baked in:
 *   - step context is arena-reset by runtime.c each call; no malloc here;
 *   - KV written via ggml_cpy into views -> buffers never reallocate;
 *   - decode path (logits_last_only) computes lm_head for the final row only;
 *   - causal mask built once per step as a small f32 tensor [nkv, n].
 */
#include "model.h"
#include "../core/phonos_log.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GRAPH_SIZE_BUILD 8192

static struct ggml_tensor* norm_affine(struct ggml_context* c, struct ggml_tensor* x,
                                       struct ggml_tensor* w, struct ggml_tensor* b, float eps) {
    struct ggml_tensor* xn = ggml_rms_norm(c, x, eps);
    struct ggml_tensor* r  = ggml_mul(c, xn, w);      /* w [E] broadcast over columns */
    return ggml_add(c, r, b);
}

void ph_rope_tables(float* cos_t, float* sin_t, int n_tok, int hd, int pos_begin, float theta) {
    const int half = hd/2;
    for (int t = 0; t < n_tok; t++) {
        float pos = (float)(pos_begin + t);
        for (int i = 0; i < half; i++) {
            float inv = 1.0f / powf(theta, (float)(2*i)/hd);
            float f = pos * inv;
            cos_t[t*half + i] = cosf(f);
            sin_t[t*half + i] = sinf(f);
        }
    }
}

/* rotate-half RoPE on x [hd, n_tok] with tables [hd/2, n_tok].
 * output rows = [first_half | second_half] == llama.cpp convention */
static struct ggml_tensor* rope_apply(struct ggml_context* c, struct ggml_tensor* x,
                                      struct ggml_tensor* cos_t, struct ggml_tensor* sin_t) {
    const int64_t hd = x->ne[0], nt = x->ne[1];
    const size_t rb = (size_t)x->nb[1];              /* stride between token columns */
    const size_t hb = (size_t)(hd/2) * ggml_element_size(x);
    struct ggml_tensor* a = ggml_reshape_2d(c, ggml_view_3d(c, x, hd/2, 1, nt, hb, rb, 0),   hd/2, nt);
    struct ggml_tensor* b = ggml_reshape_2d(c, ggml_view_3d(c, x, hd/2, 1, nt, hb, rb, hb),   hd/2, nt);
    struct ggml_tensor* o1 = ggml_sub(c, ggml_mul(c, a, cos_t), ggml_mul(c, b, sin_t));
    struct ggml_tensor* o2 = ggml_add(c, ggml_mul(c, b, cos_t), ggml_mul(c, a, sin_t));
    return ggml_concat(c, o1, o2, 0);                /* [hd, n] */
}

struct ggml_cgraph* ph_build_graph(struct ggml_context* ctx, const struct ph_hparams* hp,
                                   const struct ph_weights* w, struct ph_kv_cache* kv,
                                   struct ggml_tensor* tokens_in /* [n] i32 */,
                                   int n_tokens, int pos_begin, int logits_last_only) {
    const int E   = hp->n_embd;
    const int NH  = hp->n_head;
    const int HD  = E / NH;
    const int NC  = kv->n_ctx;
    const int nkv_max = pos_begin + n_tokens;   /* keys visible to this step */

    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, GRAPH_SIZE_BUILD, false);

    /* rope tables + causal mask are INPUT tensors: the builder only declares
     * shapes; runtime.c fills them after sched alloc. */
    struct ggml_tensor* cos_t = ggml_new_tensor_2d(ctx, GGML_F32, HD/2, n_tokens);
    struct ggml_tensor* sin_t = ggml_new_tensor_2d(ctx, GGML_F32, HD/2, n_tokens);
    ggml_set_name(cos_t, "rope.cos");  ggml_set_input(cos_t);
    ggml_set_name(sin_t, "rope.sin");  ggml_set_input(sin_t);

    struct ggml_tensor* x = ggml_get_rows(ctx, w->tok_emb, tokens_in);   /* [E, n] */

    /* causal mask [nkv, n]: additive 0 / -1e30 (keys along dim0 to match scores layout) */
    struct ggml_tensor* mask = ggml_new_tensor_2d(ctx, GGML_F32, nkv_max, n_tokens);
    ggml_set_name(mask, "causal.mask"); ggml_set_input(mask);

    for (int L = 0; L < hp->n_layers; L++) {
        struct ggml_tensor* h = norm_affine(ctx, x, w->attn_norm_w[L], w->attn_norm_b[L], hp->eps_norm);

        struct ggml_tensor* q = ggml_mul_mat(ctx, w->wq[L], h);          /* [E, n] == [HD,n,NH] */
        struct ggml_tensor* k = ggml_mul_mat(ctx, w->wk[L], h);          /* [HD, n] */
        struct ggml_tensor* v = ggml_mul_mat(ctx, w->wv[L], h);          /* [HD, n] */

        struct ggml_tensor* k_rope = rope_apply(ctx, k, cos_t, sin_t);   /* [HD, n] */

        /* append K,V into cache at [pos_begin .. ) — cache layout [NC, HD]:
         * slice = rows pos_begin.., all cols -> view_2d(ne0=n, ne1=HD) */
        struct ggml_tensor* kcache = kv->k[L];                            /* [NC, HD] */
        struct ggml_tensor* vcache = kv->v[L];
        struct ggml_tensor* kslice = ggml_view_2d(ctx, kcache, n_tokens, HD, kcache->nb[1],
                                                  (size_t)pos_begin * kcache->nb[0]);
        struct ggml_tensor* vslice = ggml_view_2d(ctx, vcache, n_tokens, HD, vcache->nb[1],
                                                  (size_t)pos_begin * vcache->nb[0]);
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_transpose(ctx, k_rope), kslice));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_transpose(ctx, v),       vslice));

        /* attention per query head (MQA: all heads read same K/V window).
         * kc/vc views: [nkv, HD] starting at col 0 of the cache rows. */
        struct ggml_tensor* attn_outs[8];
        for (int hh = 0; hh < NH && hh < 8; hh++) {
            struct ggml_tensor* qh  = ggml_view_2d(ctx, q, HD, n_tokens, q->nb[1],
                                                   (size_t)(hh*HD)*ggml_element_size(q));
            struct ggml_tensor* qhr = rope_apply(ctx, qh, cos_t, sin_t);  /* [HD, n] */
            struct ggml_tensor* kc  = ggml_view_2d(ctx, kcache, nkv_max, HD, kcache->nb[1], 0); /* [nkv, HD] */
            struct ggml_tensor* vc  = ggml_view_2d(ctx, vcache, nkv_max, HD, vcache->nb[1], 0);

            struct ggml_tensor* s = ggml_mul_mat(ctx, kc, qhr);           /* [nkv, n] */
            s = ggml_scale(ctx, s, 1.0f/sqrtf((float)HD));
            s = ggml_add(ctx, s, mask);
            struct ggml_tensor* p = ggml_soft_max_ext(ctx, s, NULL, 1.0f, 0.0f); /* over dim0 = keys ✓ */
            attn_outs[hh] = ggml_mul_mat(ctx, ggml_permute(ctx, vc, 1, 0, 2, 3), p); /* [HD, n] */
        }

        /* merge heads into contiguous [E, n] scratch via copies (correct strides) */
        struct ggml_tensor* obuf = ggml_new_tensor_2d(ctx, GGML_F32, E, n_tokens);
        for (int hh = 0; hh < NH && hh < 8; hh++) {
            struct ggml_tensor* dst = ggml_view_2d(ctx, obuf, HD, n_tokens, obuf->nb[1],
                                                   (size_t)(hh*HD)*ggml_element_size(obuf));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, attn_outs[hh], dst));
        }
        struct ggml_tensor* attn = ggml_mul_mat(ctx, w->wo[L], obuf);     /* [E, n] */
        x = ggml_add(ctx, x, attn);

        /* FFN: up -> gelu -> down (SwiGLU-lite; gate split lands with real ckpt) */
        struct ggml_tensor* h2 = norm_affine(ctx, x, w->ffn_norm_w[L], w->ffn_norm_b[L], hp->eps_norm);
        struct ggml_tensor* u  = ggml_mul_mat(ctx, w->w1[L], h2);         /* [FFN, n] */
        struct ggml_tensor* g  = ggml_gelu(ctx, u);
        struct ggml_tensor* d  = ggml_mul_mat(ctx, w->w2[L], g);          /* [E, n] */
        x = ggml_add(ctx, x, d);
    }

    struct ggml_tensor* xn = norm_affine(ctx, x, w->out_norm_w, w->out_norm_b, hp->eps_norm);
    struct ggml_tensor* logits;
    if (logits_last_only && n_tokens > 1) {
        struct ggml_tensor* last = ggml_view_2d(ctx, xn, E, 1, xn->nb[1],
                                                (size_t)(n_tokens-1) * xn->nb[1]);
        logits = ggml_mul_mat(ctx, w->lm_head, last);                     /* [V, 1] */
    } else {
        logits = ggml_mul_mat(ctx, w->lm_head, xn);                       /* [V, n] */
    }
    ggml_build_forward_expand(gf, logits);
    return gf;
}

/* ---- KV cache ------------------------------------------------------------
 * ONE persistent host buffer holds every K/V tensor payload; tensor structs
 * live in a no_alloc ctx. Cache tensors are [n_ctx, HD] (dim0 = keys) so the
 * attention softmax normalizes over dim0 directly — no transposes at read time.
 * Allocated once at load, never reallocated as the sequence grows. */
int ph_kv_alloc(struct ph_kv_cache* kv, const struct ph_hparams* hp, int n_ctx) {
    memset(kv, 0, sizeof(*kv));
    const int HD = hp->n_embd / hp->n_head;
    const size_t per   = (size_t)n_ctx * HD * sizeof(float);
    const size_t total = per * 2 * (size_t)hp->n_layers;

    ggml_backend_dev_t cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu_dev) { phonos_log(PHONOS_LOG_ERROR, "kv: no CPU device registered"); return -1; }
    kv->buf = ggml_backend_buft_alloc_buffer(ggml_backend_dev_buffer_type(cpu_dev), total);
    if (!kv->buf) { phonos_log(PHONOS_LOG_ERROR, "kv: buffer alloc (%zu B) failed", total); return -1; }

    char* base = (char*)ggml_backend_buffer_get_base(kv->buf);
    ggml_backend_buffer_clear(kv->buf, 0, total);      /* zero-fill once */

    struct ggml_init_params p = { .mem_size = (size_t)hp->n_layers*96 + 4096, .no_alloc = true };
    kv->ctx = ggml_init(p);
    if (!kv->ctx) { phonos_log(PHONOS_LOG_ERROR, "kv: ggml_init failed"); goto bad; }
    kv->k = calloc(hp->n_layers, sizeof(struct ggml_tensor*));
    kv->v = calloc(hp->n_layers, sizeof(struct ggml_tensor*));
    if (!kv->k || !kv->v) goto bad;

    for (int L = 0; L < hp->n_layers; L++) {
        char nm[32];
        snprintf(nm, sizeof nm, "kvcache.k%d", L);
        kv->k[L] = ggml_new_tensor_2d(kv->ctx, GGML_F32, n_ctx, HD);     /* [NC, HD] */
        ggml_set_name(kv->k[L], nm);
        if (ggml_backend_tensor_alloc(kv->buf, kv->k[L], base + (size_t)(2*L)*per) != GGML_STATUS_SUCCESS) goto bad;

        snprintf(nm, sizeof nm, "kvcache.v%d", L);
        kv->v[L] = ggml_new_tensor_2d(kv->ctx, GGML_F32, n_ctx, HD);
        ggml_set_name(kv->v[L], nm);
        if (ggml_backend_tensor_alloc(kv->buf, kv->v[L], base + (size_t)(2*L+1)*per) != GGML_STATUS_SUCCESS) goto bad;
    }
    kv->n_ctx = n_ctx;
    kv->cur_len = 0;
    return 0;
bad:
    ph_kv_free(kv);
    return -1;
}

void ph_kv_free(struct ph_kv_cache* kv) {
    if (!kv) return;
    if (kv->ctx) ggml_free(kv->ctx);
    if (kv->buf) ggml_backend_buffer_free(kv->buf);
    free(kv->k); free(kv->v);
    memset(kv, 0, sizeof(*kv));
}

void ph_kv_rewind(struct ph_kv_cache* kv, int new_len, int n_layers) {
    if (new_len < 0) new_len = 0;
    if (new_len >= kv->cur_len) { if (new_len > kv->cur_len) kv->cur_len = new_len; return; }
    const int HD = (int)kv->k[0]->ne[1];
    const size_t row = kv->k[0]->nb[1];                 /* bytes per key row */
    for (int L = 0; L < n_layers; L++) {
        memset((char*)kv->k[L]->data + (size_t)new_len*row, 0,
               (size_t)(kv->cur_len-new_len)*row);
        memset((char*)kv->v[L]->data + (size_t)new_len*row, 0,
               (size_t)(kv->cur_len-new_len)*row);
    }
    kv->cur_len = new_len;
}
