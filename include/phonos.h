/* phonos.h — single-header public C API for the PHONOS speech+text engine.
 *
 * This header is the ONLY thing embedders need (Android JNI, Swift bridge,
 * desktop apps). The implementation stays modular in src/ — the "single
 * header" property belongs to the interface, never to the engine itself.
 *
 * Threading model: a phonos_context is NOT thread-safe; serialize calls or
 * own one context per thread. Callbacks run on the calling thread.
 */
#ifndef PHONOS_H
#define PHONOS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PHONOS_VERSION 1

/* ---- visibility --------------------------------------------------------- */
#if defined(_WIN32)
#  ifndef PHONOS_API
#    define PHONOS_API __declspec(dllexport)
#  endif
#else
#  ifndef PHONOS_API
#    define PHONOS_API __attribute__((visibility("default")))
#  endif
#endif

/* ---- error codes -------------------------------------------------------- */
typedef enum phonos_status {
    PHONOS_OK              =  0,
    PHONOS_ERR_GENERIC     = -1,
    PHONOS_ERR_IO          = -2,   /* file not found / truncated */
    PHONOS_ERR_FORMAT      = -3,   /* bad GGUF magic/kv/arch version */
    PHONOS_ERR_NOMEM       = -4,   /* arena / buffer allocation failed */
    PHONOS_ERR_BACKEND     = -5,   /* no usable backend */
    PHONOS_ERR_INVALID_ARG = -6,
    PHONOS_ERR_UNSUPPORTED = -7,   /* op/type not available on chosen backend */
    PHONOS_ERR_CANCELLED   = -8,
} phonos_status;

/* ---- logging ------------------------------------------------------------ */
typedef enum phonos_log_level { PHONOS_LOG_DEBUG=0, PHONOS_LOG_INFO=1, PHONOS_LOG_WARN=2, PHONOS_LOG_ERROR=3 } phonos_log_level;
typedef void (*phonos_log_cb)(phonos_log_level lvl, const char* msg, void* user);
PHONOS_API void phonos_set_log(phonos_log_level min_lvl, phonos_log_cb cb, void* user);

/* ---- library init / backends -------------------------------------------- */
/* Idempotent. Initializes ggml backend registry. n_threads_hint<=0 => auto. */
PHONOS_API phonos_status phonos_init(int n_threads_hint);
PHONOS_API void          phonos_shutdown(void);
PHONOS_API int           phonos_backend_count(void);
PHONOS_API const char*   phonos_backend_name(int i);       /* "CPU", "CUDA0", ... */
PHONOS_API int           phonos_backend_is_cpu(int i);

/* ---- audio input description (for speech path) -------------------------- */
typedef struct phonos_audio {
    const float* pcm_f32;   /* mono, normalized [-1,1] */
    uint64_t     n_samples;
    uint32_t     sample_rate;   /* e.g. 16000; resampled internally if needed */
} phonos_audio;

/* ---- generation params -------------------------------------------------- */
typedef struct phonos_gen_params {
    int      n_predict;        /* max new tokens; -1 = until EOS            */
    int      seed;             /* <0 => deterministic greedy path regardless */
    float    temperature;      /* 0.0 => greedy                            */
    int      top_k;            /* <=0 disables                             */
    float    top_p;            /* >=1.0 disables                           */
    float    repeat_penalty;   /* 1.0 disables                             */
    int      n_ctx_chunk;      /* prefill chunk size; 0 => default (512)   */
    const char* stop_word;     /* optional: stop after emitting this token */
} phonos_gen_params;

/* streaming callback: called per generated token.
 * text = decoded piece (UTF-8), id = token id.
 * Return non-zero to cancel generation early. */
typedef int (*phonos_token_cb)(const char* text, int id, void* user);

/* ---- opaque context ----------------------------------------------------- */
typedef struct phonos_context phonos_context;

typedef struct phonos_model_params {
    const char* path;          /* .gguf model file                          */
    int         n_gpu_layers;  /* 0 => pure CPU; >0 => offload if backend present */
    int         n_ctx;         /* prompt/context window; 0 => from metadata */
    int         use_mmap;      /* mmap weights when available               */
    int         n_threads;     /* 0 => auto (honors phonos_init hint)       */
    int         n_ctx_chunk;   /* prefill chunk size; 0 => default (512)  */
} phonos_model_params;

PHONOS_API phonos_context* phonos_model_load(const phonos_model_params* mp, phonos_status* out_err);
PHONOS_API void            phonos_model_free(phonos_context* ctx);

/* introspection (for `phonos-cli info`) */
PHONOS_API int         phonos_vocab_size(const phonos_context* ctx);
PHONOS_API int         phonos_n_layers(const phonos_context* ctx);
PHONOS_API int         phonos_n_ctx_train(const phonos_context* ctx);
PHONOS_API int         phonos_has_speech(const phonos_context* ctx); /* encoder present? */
PHONOS_API const char* phonos_arch_name(const phonos_context* ctx);
PHONOS_API size_t      phonos_weight_bytes(const phonos_context* ctx);

/* ---- KV/session state --------------------------------------------------- */
PHONOS_API void phonos_reset_session(phonos_context* ctx);

/* ---- text-only generate ------------------------------------------------- */
PHONOS_API phonos_status phonos_generate(phonos_context* ctx,
                                         const char* prompt_utf8,
                                         const phonos_gen_params* gp,
                                         phonos_token_cb on_token, void* user);

/* ---- speech (or speech+text-prompt) ------------------------------------ */
/* Encodes audio through the speech front-end, conditions the LM, then runs
 * generation. audio may be NULL (pure text). Streaming re-call with growing
 * buffers is supported: internal frame cache avoids re-processing old audio. */
PHONOS_API phonos_status phonos_transcribe(phonos_context* ctx,
                                           const phonos_audio* audio,
                                           const char* text_prompt,   /* may be NULL */
                                           const phonos_gen_params* gp,
                                           phonos_token_cb on_token, void* user);

/* ---- perf readout (engine internals/tests/benchmarks) ------------------- */
PHONOS_API void phonos_last_perf(const phonos_context* ctx,
                                 double* ms_prefill, double* ms_per_tok,
                                 long* n_tokens);

#ifdef __cplusplus
}
#endif

#endif /* PHONOS_H */
