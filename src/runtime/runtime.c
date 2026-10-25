/* runtime.c — engine object: load, prefill/decode loop, session state.
 *
 * Execution model (verified against vendored ggml headers):
 *   - ONE persistent ggml_backend_sched over {accel?, cpu}; per step we
 *     build the graph in a resettable arena ctx, sched_reset + sched_alloc_graph
 *     + fill declared INPUT tensors (tokens / rope cos/sin / causal mask) and
 *     sched_graph_compute. Steady-state decode => zero malloc.
 *   - KV cache lives in ONE host buffer allocated at load (see ph_kv_alloc);
 *     never reallocated as the sequence grows.
 *   - Prefill runs in chunks with logits_last_only so lm_head is paid once
 *     per chunk; decode computes full-row logits only for the final token.
 *   - Timing uses monotonic ms for tokens/s reporting (Phase 5 benchmarks).
 */
#include "engine.h"
#include "../model/model.h"
#include "../io/gguf_loader.h"
#include "../core/phonos_log.h"

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define DEFAULT_CHUNK 512
#define GRAPH_SIZE    8192

struct engine {
    struct gguf_file*   g;
    struct ph_hparams   hp;
    struct ph_weights   w;
    struct ph_vocab     vocab;

    struct ph_kv_cache  kv;

    /* execution: persistent scheduler + reusable graph contexts */
    ggml_backend_t             be_cpu;
    ggml_backend_t             be_accel;      /* NULL => CPU-only            */
    ggml_backend_sched_t       sched;
    struct ggml_context*       c_build;       /* scratch: graph construction */
    struct ggml_init_params    build_params;
    struct ggml_context*       c_scratch;     /* persistent merge-scratch tensor struct */
    ggml_backend_buffer_t      merge_buf;     /* [E, chunk] F32 attention merge scratch */

    int   n_ctx;              /* effective window (min(req, train))          */
    int   n_threads;
    int   chunk;              /* prefill chunk size                           */

    /* reusable input staging buffers (no alloc in hot loop) */
    int32_t* tok_buf;         /* [chunk]                                      */
    float*   cos_buf;         /* [chunk * HD/2]                               */
    float*   sin_buf;
    float*   mask_buf;        /* [nkv_max * chunk]                            */

    /* perf counters */
    double ms_prefill;
    double ms_decode_total;
    long   n_generated;
};

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1000.0 + ts.tv_nsec/1e6;
}

/* ---- sizing helpers ------------------------------------------------------ */
static size_t estimate_step_bytes(const struct ph_hparams* hp, int chunk, int n_ctx_eff) {
    /* conservative upper bound for one prefill-chunk graph: activations
     * ~ chunk*(E + FFN)*4B per layer *2 + scores chunk*n_ctx*4 + misc */
    const size_t E = (size_t)hp->n_embd, F = (size_t)hp->n_ffn;
    size_t per_tok = (E*8 + F*2) * sizeof(float);          /* x,h,q,k,v,u,g,d + head bufs */
    size_t layers  = per_tok * (size_t)chunk * (size_t)hp->n_layers;
    size_t attn    = (size_t)chunk * (size_t)n_ctx_eff * sizeof(float) * (size_t)hp->n_layers;
    size_t logits  = (size_t)hp->n_vocab * chunk * sizeof(float);
    size_t total   = layers + attn + logits + (1u<<20);   /* +1 MiB slack */
    if (total < (8u<<20)) total = 8u<<20;                  /* floor 8 MiB */
    return total;
}

/* ---- init/shutdown ------------------------------------------------------- */
phonos_status engine_init(int n_threads_hint) {
    extern phonos_status be_init(int);
    return be_init(n_threads_hint);
}
void engine_shutdown(void) {
    extern void be_shutdown(void);
    be_shutdown();
}
int engine_backend_count(void)        { extern int be_backend_count(void); return be_backend_count(); }
const char* engine_backend_name(int i){ extern const char* be_backend_name(int); return be_backend_name(i); }
int engine_backend_is_cpu(int i)      { extern int be_backend_is_cpu(int); return be_backend_is_cpu(i); }

/* ---- load ---------------------------------------------------------------- */
struct engine* engine_load(const struct engine_params* p, phonos_status* err) {
    phonos_status st = PHONOS_OK;
    struct engine* e = calloc(1, sizeof(*e));
    if (!e) { st = PHONOS_ERR_NOMEM; goto fail; }

    e->g = gguf_file_load(p->path, p->use_mmap);
    if (!e->g) { st = PHONOS_ERR_IO; goto fail; }

    if (ph_hparams_read(e->g, &e->hp) != 0) { st = PHONOS_ERR_FORMAT; goto fail; }
    if (ph_weights_load(e->g, &e->hp, &e->w) != 0) { st = PHONOS_ERR_NOMEM; goto fail; }

    ph_vocab_load(e->g, &e->hp, &e->vocab);
    if (e->vocab.n <= 0) { st = PHONOS_ERR_FORMAT; goto fail; }

    /* effective context: requested or trained, clamped to something sane */
    int want = p->n_ctx > 0 ? p->n_ctx : e->hp.n_ctx_train;
    e->n_ctx = want;
    if (e->n_ctx > e->hp.n_ctx_train) e->n_ctx = e->hp.n_ctx_train;
    if (e->n_ctx < 64) e->n_ctx = 64;

    e->chunk = (p->chunk > 0 && p->chunk <= DEFAULT_CHUNK) ? p->chunk : DEFAULT_CHUNK;

    /* ---- backends + scheduler ------------------------------------------- */
    extern phonos_status be_init(int);
    be_init(p->n_threads_hint);

    e->be_cpu = ggml_backend_cpu_init();
    if (!e->be_cpu) { st = PHONOS_ERR_GENERIC; goto fail; }

    ggml_backend_dev_t dev = engine_pick_device(p->n_gpu_layers);
    if (dev) e->be_accel = ggml_backend_dev_init(dev, NULL);

    ggml_backend_t bebs[2]; int nb = 0;
    if (e->be_accel) bebs[nb++] = e->be_accel;
    bebs[nb++] = e->be_cpu;
    e->sched = ggml_backend_sched_new(bebs, NULL, nb, GRAPH_SIZE, false, true);
    if (!e->sched) { st = PHONOS_ERR_NOMEM; goto fail; }

    /* KV cache lives on the accelerator when available (device memory =>
     * scheduler treats it as leaf data: no per-step H2D copies). CPU path uses
     * one persistent host buffer. */
    if (ph_kv_alloc(&e->kv, &e->hp, e->n_ctx, dev ? dev : NULL) != 0) {
        st = PHONOS_ERR_NOMEM; goto fail;
    }

    /* graph-construction arena (reset every forward pass) */
    size_t bytes = estimate_step_bytes(&e->hp, e->chunk, e->n_ctx);
    e->build_params = (struct ggml_init_params){ .mem_size = bytes, .no_alloc = false };
    e->c_build = ggml_init(e->build_params);
    if (!e->c_build) { st = PHONOS_ERR_NOMEM; goto fail; }

    const int HD = e->hp.n_embd / e->hp.n_head;
    e->tok_buf  = malloc(sizeof(int32_t) * (size_t)e->chunk);
    e->cos_buf  = malloc(sizeof(float) * (size_t)e->chunk * (HD/2));
    e->sin_buf  = malloc(sizeof(float) * (size_t)e->chunk * (HD/2));
    e->mask_buf = malloc(sizeof(float) * (size_t)(e->n_ctx + 1) * (size_t)e->chunk);
    if (!e->tok_buf || !e->cos_buf || !e->sin_buf || !e->mask_buf) { st = PHONOS_ERR_NOMEM; goto fail; }

    /* persistent attention-merge scratch [E, chunk] F32 in a dedicated ctx so
     * its struct survives ggml_reset of the build arena (zero-alloc decode) */
    {
        struct ggml_init_params sp = { .mem_size = 4096, .no_alloc = false };
        e->c_scratch = ggml_init(sp);
        if (!e->c_scratch) { st = PHONOS_ERR_NOMEM; goto fail; }
        struct ggml_tensor* m = ggml_new_tensor_2d(e->c_scratch, GGML_TYPE_F32, e->hp.n_embd, e->chunk);
        ggml_set_name(m, "attn.merge");
        ggml_backend_buffer_t mb = ggml_backend_alloc_buffer(e->be_cpu, ggml_nbytes(m));
        if (!mb) { st = PHONOS_ERR_NOMEM; goto fail; }
        if (ggml_backend_tensor_alloc(mb, m, ggml_backend_buffer_get_base(mb)) != GGML_STATUS_SUCCESS) {
            ggml_backend_buffer_free(mb); st = PHONOS_ERR_NOMEM; goto fail;
        }
        e->merge_buf = mb;
        ph_set_merge_scratch(m, e->chunk);
    }

    e->n_threads = engine_n_threads_impl();
    be_set_cpu_threads(e->be_cpu, e->n_threads);

    phonos_log(PHONOS_LOG_INFO, "engine: loaded '%s' arch=%s n_ctx=%d weights=%.2f MiB",
               p->path, PH_ARCH_NAME, e->n_ctx, (double)e->w.total_bytes/1048576.0);
    if (err) *err = PHONOS_OK;
    return e;

fail:
    if (err) *err = st;
    engine_free(e);
    return NULL;
}

void engine_free(struct engine* e) {
    if (!e) return;
    ph_set_merge_scratch(NULL, 0);
    free(e->tok_buf); free(e->cos_buf); free(e->sin_buf); free(e->mask_buf);
    if (e->merge_buf) ggml_backend_buffer_free(e->merge_buf);
    if (e->c_scratch) ggml_free(e->c_scratch);
    if (e->sched) ggml_backend_sched_free(e->sched);
    if (e->c_build) ggml_free(e->c_build);
    if (e->be_accel) ggml_backend_free(e->be_accel);
    if (e->be_cpu) ggml_backend_free(e->be_cpu);
    ph_kv_free(&e->kv);
    ph_vocab_free(&e->vocab);
    ph_weights_free(&e->w);
    if (e->g) gguf_file_free(e->g);
    free(e);
}

void engine_reset_session(struct engine* e) {
    if (!e) return;
    ph_kv_rewind(&e->kv, 0, e->hp.n_layers);
    e->kv.cur_len = 0;
}

/* ---- single forward step --------------------------------------------------
 * tokens: ids[n]; pos_begin: absolute position of first token.
 * Returns pointer into logits data ([V, n] or [V, 1] when last-only),
 * or NULL on error. All memory comes from the two persistent contexts. */
static const float* run_step(struct engine* e, const int* tokens, int n,
                             int pos_begin, int logits_last_only) {
    const int HD = e->hp.n_embd / e->hp.n_head;

    /* 1. build fresh graph in reset arena (zero malloc steady-state: ggml_reset
     *    reuses the same block; no tensors are created between steps outside it) */
    ggml_reset(e->c_build);

    struct ggml_tensor* tin = ggml_new_tensor_1d(e->c_build, GGML_TYPE_I32, n);
    ggml_set_name(tin, "input.tokens");
    ggml_set_input(tin);

    struct ggml_cgraph* gf_built = ph_build_graph(e->c_build, &e->hp, &e->w, &e->kv,
                                                  tin, n, pos_begin, logits_last_only);
    if (!gf_built) { phonos_log(PHONOS_LOG_ERROR, "runtime: graph build failed"); return NULL; }

    /* 2. let the scheduler place ops and allocate the graph tensors.
     *    NOTE: ggml_backend_sched_alloc_graph returns bool; the expanded,
     *    backend-allocated graph is retrieved via ggml_backend_sched_get_compute_graph. */
    ggml_backend_sched_reset(e->sched);
    if (!ggml_backend_sched_alloc_graph(e->sched, gf_built)) {
        phonos_log(PHONOS_LOG_ERROR, "runtime: sched alloc failed"); return NULL;
    }
    struct ggml_cgraph* gf = gf_built;   /* alloc_graph fills buffers in-place on this graph */

    /* 3. fill declared inputs — ggml_backend_tensor_set is the universal path:
     *    plain memcpy for host buffers, async H2D copy for device buffers. */
    struct ggml_tensor* tok_in  = ggml_graph_get_tensor(gf, "input.tokens");
    struct ggml_tensor* cos_in  = ggml_graph_get_tensor(gf, "rope.cos");
    struct ggml_tensor* sin_in  = ggml_graph_get_tensor(gf, "rope.sin");
    struct ggml_tensor* mask_in = ggml_graph_get_tensor(gf, "causal.mask");
    if (!tok_in || !cos_in || !sin_in || !mask_in) {
        phonos_log(PHONOS_LOG_ERROR, "runtime: input tensors missing from graph");
        return NULL;
    }
    #define FILL(t, src, nbytes) ggml_backend_tensor_set((t), (src), 0, (nbytes))

    FILL(tok_in, tokens, sizeof(int32_t)*(size_t)n);

    ph_rope_tables(e->cos_buf, e->sin_buf, n, HD, pos_begin, e->hp.rope_theta);
    FILL(cos_in, e->cos_buf, sizeof(float)*(size_t)n*(HD/2));
    FILL(sin_in, e->sin_buf, sizeof(float)*(size_t)n*(HD/2));

    /* causal mask [nkv, n]: additive 0 / -1e30 (keys along dim0) */
    {
        const int nkv = pos_begin + n;
        for (int t = 0; t < n; t++) {
            const int visible = pos_begin + t + 1;
            for (int j = 0; j < nkv; j++)
                e->mask_buf[(size_t)t*nkv + j] = (j < visible) ? 0.0f : -1e30f;
        }
        FILL(mask_in, e->mask_buf, sizeof(float)*(size_t)n*(size_t)nkv);
    }
    #undef FILL

    /* 4. execute */
    enum ggml_status rc = ggml_backend_sched_graph_compute(e->sched, gf);
    if (rc != GGML_STATUS_SUCCESS) {
        phonos_log(PHONOS_LOG_ERROR, "runtime: sched compute=%d", (int)rc);
        return NULL;
    }

    /* 5. read logits: last node of the expanded graph is the lm_head mul_mat */
    struct ggml_tensor* logits = ggml_graph_node(gf, ggml_graph_n_nodes(gf)-1);
    if (!logits || ggml_op(logits) != GGML_OP_MUL_MAT) {
        phonos_log(PHONOS_LOG_ERROR, "runtime: logits node missing");
        return NULL;
    }
    if (ggml_backend_buffer_is_host(logits->buffer)) {
        return (const float*)logits->data;   /* CPU path: zero copy */
    }
    {
        /* device logits -> copy out once per step (Phase 4 will keep sampling on-device) */
        static float* tmp = NULL; static size_t tmp_sz = 0;
        size_t need = ggml_nbytes(logits);
        if (need > tmp_sz) { free(tmp); tmp = malloc(need); tmp_sz = need; }
        ggml_backend_tensor_get(logits, tmp, 0, need);
        return tmp;
    }
}

/* ---- generation loop ------------------------------------------------------ */
phonos_status engine_generate(struct engine* e, const char* prompt_utf8,
                              const struct gen_config* cfg,
                              phonos_token_cb cb, void* user) {
    if (!e || !prompt_utf8 || !cb) return PHONOS_ERR_INVALID_ARG;

    int cap = (int)strlen(prompt_utf8) + 8;
    int* ids = malloc(sizeof(int)*(size_t)cap);
    if (!ids) return PHONOS_ERR_NOMEM;
    int n = ph_tokenize(&e->vocab, prompt_utf8, ids, cap);
    if (n < 0) { free(ids); return PHONOS_ERR_INVALID_ARG; }

    /* insert BOS at front (shift) */
    memmove(ids+1, ids, sizeof(int)*(size_t)n);
    ids[0] = e->vocab.bos;
    n++;

    struct ph_sampler_cfg sc = {
        .seed = cfg->seed, .temperature = cfg->temperature,
        .top_k = cfg->top_k, .top_p = cfg->top_p,
        .repeat_penalty = cfg->repeat_penalty };
    uint64_t rng = (cfg->seed >= 0) ? (uint64_t)cfg->seed : 1234567ull;

    engine_reset_session(e);

    /* ---- prefill in chunks ---- */
    double t0 = now_ms();
    int pos = 0;
    const float* last_logits = NULL;
    while (pos < n) {
        int take = n - pos; if (take > e->chunk) take = e->chunk;
        last_logits = run_step(e, ids + pos, take, pos, /*last_only*/1);
        if (!last_logits) { free(ids); return PHONOS_ERR_GENERIC; }
        pos += take;
        if (pos > e->n_ctx - 8) break;   /* window full: drop rest of prompt */
    }
    e->ms_prefill = now_ms() - t0;
    e->kv.cur_len = pos;                 /* graph wrote KV up to pos */

    /* ---- first sample ---- */
    int next = -1;
    {
        int tmp_id;
        if (ph_sample_next(last_logits, e->hp.n_vocab, &sc, rng, NULL, 0, &tmp_id) != PHONOS_OK)
            { free(ids); return PHONOS_ERR_NOMEM; }
        next = tmp_id;
    }

    /* penalty history: fixed ring (no alloc in loop) */
    int hist[512]; int nhist = 0;
    #define PUSH_HIST(t) do { if (nhist < 512) hist[nhist++] = (t); } while (0)
    for (int i = 0; i < pos && i < 512; i++) PUSH_HIST(ids[i]);

    long budget = cfg->n_predict < 0 ? 100000 : cfg->n_predict;
    char piece[128];
    int stop_id = -1;
    if (cfg->stop_word) {
        for (int t = 0; t < e->vocab.n; t++)
            if (strcmp(e->vocab.text[t], cfg->stop_word) == 0) { stop_id = t; break; }
    }

    /* ---- autoregressive decode ---- */
    double td0 = now_ms();
    long generated = 0;
    while (generated < budget) {
        if (next == e->vocab.eos) break;
        if (stop_id >= 0 && next == stop_id) break;

        ph_detokenize_piece(&e->vocab, next, piece, sizeof piece);
        if (cb(piece, next, user) != 0) { free(ids); return PHONOS_ERR_CANCELLED; }

        PUSH_HIST(next);
        generated++;

        if (e->kv.cur_len >= e->n_ctx) {
            phonos_log(PHONOS_LOG_WARN, "runtime: context full at %d tokens", e->kv.cur_len);
            break;
        }

        last_logits = run_step(e, &next, 1, e->kv.cur_len, 0);
        if (!last_logits) { free(ids); return PHONOS_ERR_GENERIC; }
        e->kv.cur_len++;

        int tmp2;
        if (ph_sample_next(last_logits, e->hp.n_vocab, &sc, rng,
                           hist, nhist, &tmp2) != PHONOS_OK) { free(ids); return PHONOS_ERR_NOMEM; }
        next = tmp2;
        rng = ph_rng_advance(rng);
    }
    e->ms_decode_total = now_ms() - td0;
    e->n_generated = generated;

    free(ids);
    return PHONOS_OK;
}

phonos_status engine_transcribe(struct engine* e, const struct audio_view* av,
                                const char* text_prompt,
                                const struct gen_config* cfg,
                                phonos_token_cb cb, void* user) {
    if (!e) return PHONOS_ERR_INVALID_ARG;
    if (!av && !text_prompt) return PHONOS_ERR_INVALID_ARG;
    if (av && !e->hp.has_speech) {
        phonos_log(PHONOS_LOG_ERROR, "runtime: model has no speech encoder");
        return PHONOS_ERR_UNSUPPORTED;
    }
    if (av) {
        phonos_log(PHONOS_LOG_WARN, "runtime: speech path lands in Phase 2 (P2)");
        return PHONOS_ERR_UNSUPPORTED;
    }
    return engine_generate(e, text_prompt, cfg, cb, user);
}

/* ---- introspection --------------------------------------------------------- */
int engine_vocab_size(struct engine* e)   { return e ? e->hp.n_vocab : 0; }
int engine_n_layers(struct engine* e)     { return e ? e->hp.n_layers : 0; }
int engine_n_ctx_train(struct engine* e)  { return e ? e->n_ctx : 0; }
int engine_has_speech(struct engine* e)   { return e ? e->hp.has_speech : 0; }
const char* engine_arch_name(struct engine* e) { (void)e; return PH_ARCH_NAME; }
size_t engine_weight_bytes(struct engine* e){ return e ? e->w.total_bytes : 0; }

void engine_last_perf(struct engine* e, double* ms_prefill, double* ms_per_tok, long* n_tokens) {
    if (ms_prefill)  *ms_prefill  = e ? e->ms_prefill : 0;
    if (ms_per_tok)  *ms_per_tok  = (e && e->n_generated > 0) ? e->ms_decode_total/(double)e->n_generated : 0;
    if (n_tokens)    *n_tokens    = e ? e->n_generated : 0;
}
