/* phonos.c — public API implementation: thin facade over the engine internals.
 * Keep this file boring: arg validation + dispatch only. No model logic here. */
#include "phonos.h"
#include "phonos_log.h"
#include "engine.h"

#include <stdlib.h>
#include <string.h>

phonos_status phonos_init(int n_threads_hint) {
    phonos_install_ggml_logger();
    return engine_init(n_threads_hint);
}

void phonos_shutdown(void) {
    engine_shutdown();
}

int phonos_backend_count(void)      { return engine_backend_count(); }
const char* phonos_backend_name(int i) { return engine_backend_name(i); }
int phonos_backend_is_cpu(int i)    { return engine_backend_is_cpu(i); }

phonos_context* phonos_model_load(const phonos_model_params* mp, phonos_status* out_err) {
    if (!mp || !mp->path) {
        if (out_err) *out_err = PHONOS_ERR_INVALID_ARG;
        return NULL;
    }
    struct engine_params ep = {0};
    ep.path         = mp->path;
    ep.n_gpu_layers = mp->n_gpu_layers;
    ep.n_ctx        = mp->n_ctx;
    ep.use_mmap     = mp->use_mmap;
    ep.chunk        = mp->n_ctx_chunk;
    ep.n_threads_hint = mp->n_threads;

    struct engine* e = engine_load(&ep, out_err);
    if (!e) return NULL;
    return (phonos_context*)e;
}

void phonos_model_free(phonos_context* ctx) {
    if (ctx) engine_free((struct engine*)ctx);
}

int         phonos_vocab_size(const phonos_context* c)    { return engine_vocab_size((struct engine*)c); }
int         phonos_n_layers(const phonos_context* c)      { return engine_n_layers((struct engine*)c); }
int         phonos_n_ctx_train(const phonos_context* c)   { return engine_n_ctx_train((struct engine*)c); }
int         phonos_has_speech(const phonos_context* c)    { return engine_has_speech((struct engine*)c); }
const char* phonos_arch_name(const phonos_context* c)     { return engine_arch_name((struct engine*)c); }
size_t      phonos_weight_bytes(const phonos_context* c)  { return engine_weight_bytes((struct engine*)c); }

void phonos_reset_session(phonos_context* ctx) {
    if (ctx) engine_reset_session((struct engine*)ctx);
}

static void gp_to_internal(const phonos_gen_params* gp, struct gen_config* out) {
    memset(out, 0, sizeof(*out));
    out->n_predict      = gp ? gp->n_predict      : 32;
    out->seed           = gp ? gp->seed           : -1;
    out->temperature    = gp ? gp->temperature    : 0.0f;
    out->top_k          = gp ? gp->top_k          : 0;
    out->top_p          = gp ? gp->top_p          : 1.0f;
    out->repeat_penalty = gp ? gp->repeat_penalty : 1.0f;
    out->chunk          = gp ? gp->n_ctx_chunk    : 0;
    out->stop_word      = gp ? gp->stop_word      : NULL;
}

phonos_status phonos_generate(phonos_context* ctx, const char* prompt_utf8,
                              const phonos_gen_params* gp,
                              phonos_token_cb on_token, void* user) {
    if (!ctx || !prompt_utf8 || !on_token) return PHONOS_ERR_INVALID_ARG;
    struct gen_config cfg;
    gp_to_internal(gp, &cfg);
    /* public API contract: the engine prepends BOS itself (runtime.c) */
    return engine_generate((struct engine*)ctx, prompt_utf8, &cfg, on_token, user);
}

phonos_status phonos_transcribe(phonos_context* ctx, const phonos_audio* audio,
                                const char* text_prompt, const phonos_gen_params* gp,
                                phonos_token_cb on_token, void* user) {
    if (!ctx || !on_token) return PHONOS_ERR_INVALID_ARG;
    if (!audio && !text_prompt) return PHONOS_ERR_INVALID_ARG;
    struct gen_config cfg;
    gp_to_internal(gp, &cfg);
    struct audio_view av = {0};
    if (audio) {
        av.pcm = audio->pcm_f32;
        av.n_samples = audio->n_samples;
        av.sample_rate = audio->sample_rate;
    }
    return engine_transcribe((struct engine*)ctx, audio ? &av : NULL,
                             text_prompt, &cfg, on_token, user);
}

int phonos_has_synth(const phonos_context* c)   { return engine_has_synth((struct engine*)c); }
int phonos_synth_sample_rate(const phonos_context* c) { return engine_synth_sample_rate((struct engine*)c); }

phonos_status phonos_synth(phonos_context* ctx, const char* text_utf8,
                           const phonos_gen_params* gp,
                           phonos_audio_cb on_audio, void* user) {
    if (!ctx || !text_utf8 || !on_audio) return PHONOS_ERR_INVALID_ARG;
    struct gen_config cfg;
    gp_to_internal(gp, &cfg);
    return engine_synth((struct engine*)ctx, text_utf8, &cfg, on_audio, user);
}

void phonos_last_perf(const phonos_context* ctx, double* ms_prefill,
                      double* ms_per_tok, long* n_tokens) {
    if (!ctx) {
        if (ms_prefill)  *ms_prefill = 0;
        if (ms_per_tok)  *ms_per_tok = 0;
        if (n_tokens)    *n_tokens = 0;
        return;
    }
    engine_last_perf((struct engine*)ctx, ms_prefill, ms_per_tok, n_tokens);
}
