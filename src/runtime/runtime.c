/* runtime.c — engine object: load, prefill/decode loop, session state.
 *
 * Performance contract (PLAN.md):
 *   - ONE ggml context holds all weights (loaded once); per-step compute uses
 *     a dedicated scratch ctx that is reset every forward pass => zero malloc
 *     in the steady-state decode loop;
 *   - KV cache buffers are allocated once at load and never reallocated;
 *   - prompt prefill runs in chunks (default 512) with logits_last_only so we
 *     pay lm_head cost once per chunk, not once per token;
 *   - timing uses monotonic ms for tokens/s reporting (Phase 5 benchmarks).
 */
#include "engine.h"
#include "../model/model.h"
#include "../io/gguf_loader.h"
#include "../core/phonos_log.h"

#include <ggml.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_CHUNK 512

struct engine {
    struct gguf_file*   g;
    struct ph_hparams   hp;
    struct ph_weights   w;
    struct ph_vocab     vocab;

    struct ph_kv_cache  kv;

    /* per-step compute arena: reset before every graph build+run */
    struct ggml_context* c_step;
    struct ggml_init_params step_params;

    int   n_ctx;              /* effective window (min(req, train)) */
    int   n_threads;

    /* reusable input staging buffers (no alloc in hot loop) */
    int32_t* tok_buf;         /* [chunk] */
    float*   cos_buf;         /* [chunk * HD/2] */
    float*   sin_buf;

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
    size_t total   = layers + attn + logits + 1u<<20;      /* +1 MiB slack */
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
int engine_backend_count(void)      { extern int be_backend_count(void); return be_backend_count(); }
const char* engine_backend_name(int i){ extern const char* be_backend_name(int); return be_backend_name(i); }
int engine_backend_is_cpu(int i)    { extern int be_backend_is_cpu(int); return be_backend_is_cpu(i); }

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

    ph_kv_alloc(&e->kv, &e->hp, e->n_ctx);
    if (!e->kv.ctx || !e->kv.k) { st = PHONOS_ERR_NOMEM; goto fail; }

    /* step arena sized from chunk=DEFAULT_CHUNK */
    size_t bytes = estimate_step_bytes(&e->hp, DEFAULT_CHUNK, e->n_ctx);
    e->step_params = (struct ggml_init_params){ .mem_size = bytes, .no_alloc = false };
    e->c_step = ggml_init(e->step_params);
    if (!e->c_step) { st = PHONOS_ERR_NOMEM; goto fail; }

    e->tok_buf = malloc(sizeof(int32_t) * DEFAULT_CHUNK);
    e->cos_buf = malloc(sizeof(float) * DEFAULT_CHUNK * (PH_EMBD/PH_N_HEADS/2));
    e->sin_buf = malloc(sizeof(float) * DEFAULT_CHUNK * (PH_EMBD/PH_N_HEADS/2));
    if (!e->tok_buf || !e->cos_buf || !e->sin_buf) { st = PHONOS_ERR_NOMEM; goto fail; }

    extern int engine_n_threads_impl(void);
    e->n_threads = engine_n_threads_impl();

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
    free(e->tok_buf); free(e->cos_buf); free(e->sin_buf);
    if (e->c_step) ggml_free(e->c_step);
    ph_kv_free(&e->kv);
    ph_vocab_free(&e->vocab);
    ph_weights_free(&e->w);
    if (e->g) gguf_file_free(e->g);
    free(e);
}

void engine_reset_session(struct engine* e) {
    if (!e) return;
    ph_kv_rewind(&e->kv, 0, e->hp.n_layers);
}

/* ---- single forward step --------------------------------------------------
 * tokens: ids[n]; pos_begin: absolute position of first token.
 * Returns pointer into logits tensor data ([V, n] or [V, 1] when last-only),
 * or NULL on error. Caller must NOT free anything; step ctx was reset. */
static const float* run_step(struct engine* e, const int* tokens, int n,
                             int pos_begin, int logits_last_only, int fill_rope) {
    ggml_reset(e->c_step);

    struct ggml_tensor* tin = ggml_new_tensor_1d(e->c_step, GGML_I32, n);
    memcpy(tin->data, tokens, sizeof(int32_t)*(size_t)n);

    struct ggml_cgraph* gf = ph_build_graph(e->c_step, &e->hp, &e->w, &e->kv,
                                            tin, n, pos_begin, logits_last_only);
    if (!gf) { phonos_log(PHONOS_LOG_ERROR, "runtime: graph build failed"); return NULL; }

    if (fill_rope) {
        const int HD = e->hp.n_embd / e->hp.n_head;
        ph_rope_tables(e->cos_buf, e->sin_buf, n, HD, pos_begin, e->hp.rope_theta);
        /* find the input tensors by name? cheaper: they are the only 2-D f32
         * inputs -> locate via graph nodes is fragile; instead re-derive: */
    }

    struct ggml_cplan plan = ggml_graph_compute_prepare(gf, 0);
    plan.n_threads = e->n_threads;
    enum ggml_status rc = ggml_graph_compute_with_cplan(gf, &plan);
    if (rc != GGML_STATUS_SUCCESS) {
        phonos_log(PHONOS_LOG_ERROR, "runtime: ggml_graph_compute=%d", (int)rc);
        return NULL;
    }

    struct ggml_tensor* out = ggml_graph_get_output(gf, ggml_graph_n_nodes(gf)-1);
    (void)out;
    /* last expanded node is logits */
    struct ggml_tensor* logits = gf->nodes[gf->n_nodes-1];
    while (logits && ggml_op(logits) != GGML_OP_MUL_MAT) logits = NULL;
    if (!logits) { phonos_log(PHONOS_LOG_ERROR, "runtime: logits node missing"); return NULL; }
    return (const float*)logits->data;
}

/* NOTE on rope inputs: ph_build_graph marks cos_t/sin_t as ggml_set_input but
 * fills them itself in older revision; to keep the runtime simple and robust
 * we rebuild tables inside the graph builder path via set_input tensors.
 * See fix below: we actually let graph builder own table memory and runtime
 * only passes pos_begin (already implemented). The fill_rope branch above is
 * therefore unused; kept for future external-rope optimization. */

/* ---- generation loop ------------------------------------------------------ */
phonos_status engine_generate(struct engine* e, const char* prompt_utf8,
                              const struct gen_config* cfg,
                              phonos_token_cb cb, void* user) {
    if (!e || !prompt_utf8 || !cb) return PHONOS_ERR_INVALID_ARG;

    int max_ids = DEFAULT_CHUNK;
    int* ids = malloc(sizeof(int)*(size_t)(strlen(prompt_utf8)+8));
    int n = ph_tokenize(&e->vocab, prompt_utf8, ids, max_ids);
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
        int take = n - pos; if (take > DEFAULT_CHUNK) take = DEFAULT_CHUNK;
        last_logits = run_step(e, ids + pos, take, pos, /*last_only*/1, 0);
        if (!last_logits) { free(ids); return PHONOS_ERR_GENERIC; }
        pos += take;
        if (pos > e->n_ctx - 8) break;   /* window full: drop rest of prompt */
    }
    e->ms_prefill = now_ms() - t0;
    e->kv.cur_len = pos;                 /* graph wrote KV up to pos */

    /* ---- autoregressive decode ---- */
    int next = -1;
    {
        int tmp;
        if (ph_sample_next((float*)last_logits, e->hp.n_vocab, &sc, rng, NULL, 0, &tmp) != PHONOS_OK)
            { free(ids); return PHONOS_ERR_NOMEM; }
        next = tmp;
    }

    /* penalty history: reuse ids buffer tail region? simpler: small fixed ring */
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

        last_logits = run_step(e, &next, 1, e->kv.cur_len, 0, 0);
        if (!last_logits) { free(ids); return PHONOS_ERR_GENERIC; }
        e->kv.cur_len++;

        int tmp2;
        if (ph_sample_next((float*)last_logits, e->hp.n_vocab, &sc, rng,
                           hist, nhist, &tmp2) != PHONOS_OK) { free(ids); return PHONOS_ERR_NOMEM; }
        next = tmp2;
        rng = ph_rng_advance(rng);
    }
    e->ms_decode_total = now_ms() - td0;
    e->n_generated = generated;

    free(ids);
    (void)max_ids;
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
