/* engine.h — internal contract between the public facade (src/core),
 * the runtime layer (src/runtime) and model/audio builders (src/model, src/audio).
 * NOT part of the public API. */
#ifndef PHONOS_ENGINE_H
#define PHONOS_ENGINE_H

#include "phonos.h"

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

struct gguf_context;

/* ---- runtime selection --------------------------------------------------- */
enum phonos_device {
    PHONOS_DEV_AUTO = 0,   /* pick best available at load time             */
    PHONOS_DEV_CPU  = 1,   /* force CPU                                    */
    PHONOS_DEV_GPU  = 2,   /* require a GPU backend or fail                */
};

/* global one-time init: backend registry + thread pool sizing */
phonos_status engine_init(int n_threads_hint);
void          engine_shutdown(void);
int           engine_backend_count(void);
const char*   engine_backend_name(int i);
int           engine_backend_is_cpu(int i);
int           engine_n_threads(void);

/* returns accelerator device or NULL (=CPU). Implemented in backend_dispatch.c */
ggml_backend_dev_t engine_pick_device(int n_gpu_layers);

/* ---- loaded model / session --------------------------------------------- */
struct engine_params {
    const char* path;
    int         n_gpu_layers;
    int         n_ctx;        /* 0 => from metadata */
    int         use_mmap;
};

/* sampler knobs shared by text generation and the (Phase 2) speech decode loop */
struct ph_sampler_cfg {
    int   seed;             /* <0 or temperature==0 => greedy */
    float temperature;
    int   top_k;            /* <=0 disables */
    float top_p;            /* >=1 disables */
    float repeat_penalty;   /* <=1 disables */
};

phonos_status ph_sample_next(const float* logits, int n_vocab,
                             const struct ph_sampler_cfg* sc, uint64_t rng_state,
                             const int* penalty_tokens, int n_penalty,
                             int* out_id);
uint64_t      ph_rng_advance(uint64_t state);

struct gen_config {
    int   n_predict;
    int   seed;             /* <0 => greedy regardless of temperature */
    float temperature;
    int   top_k;
    float top_p;
    float repeat_penalty;
    int   chunk;            /* prefill chunk tokens; 0 => 512 */
    const char* stop_word;
};

struct audio_view {
    const float* pcm;
    uint64_t     n_samples;
    uint32_t     sample_rate;
};

/* opaque engine object (cast to/from phonos_context*) */
struct engine;

struct engine* engine_load(const struct engine_params* p, phonos_status* err);
void           engine_free(struct engine* e);
void           engine_reset_session(struct engine* e);

phonos_status engine_generate(struct engine* e, const char* prompt_utf8,
                              const struct gen_config* cfg,
                              phonos_token_cb cb, void* user);

phonos_status engine_transcribe(struct engine* e, const struct audio_view* av,
                                const char* text_prompt,
                                const struct gen_config* cfg,
                                phonos_token_cb cb, void* user);

int         engine_vocab_size(struct engine* e);
int         engine_n_layers(struct engine* e);
int         engine_n_ctx_train(struct engine* e);
int         engine_has_speech(struct engine* e);
const char* engine_arch_name(struct engine* e);
size_t      engine_weight_bytes(struct engine* e);
void        engine_last_perf(struct engine* e, double* ms_prefill,
                             double* ms_per_tok, long* n_tokens);

#endif /* PHONOS_ENGINE_H */
