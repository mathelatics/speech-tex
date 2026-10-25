/* model.h — hyper-parameters + weight tensor registry for the "phonon2" arch.
 * Internal header shared by loader, graph builders and runtime. */
#ifndef PHONOS_MODEL_H
#define PHONOS_MODEL_H

#include <ggml.h>
#include <ggml-backend.h>
#include <stdint.h>

/* ---- architecture constants (fixed for phonon2; documented in PLAN) ------ */
#define PH_ARCH_NAME      "phonon2"
#define PH_EMBD           256
#define PH_N_HEADS        4
#define PH_NKV            1          /* MQA: single K/V head -> tiny KV cache on mobile */
#define PH_KV_DIM         (PH_EMBD / PH_N_HEADS)   /* = 64; K/V width per token */
#define PH_HEAD_DIM       (PH_EMBD / PH_N_HEADS)
#define PH_FFN_MULT       8          /* ffn_hidden = 8*embd = 2048 */
#define PH_FFN_H          (PH_EMBD * PH_FFN_MULT)
#define PH_ROPE_THETA     10000.0f

/* speech front-end (log-mel) defaults; overridable via GGUF kv */
#define PH_MEL_BINS       80
#define PH_MEL_FRAMES_MS  10.0f
#define PH_MEL_WIN_MS     25.0f
#define PH_MEL_HOP_MS     10.0f
#define PH_MEL_SR         16000u
#define PH_ENC_OUT_PER_100MS 4       /* encoder temporal downsample: 10 frames -> 4 vectors */

/* special token ids (also stored in GGUF metadata for real models) */
enum ph_special {
    PH_TOK_BOS = 0,
    PH_TOK_EOS = 1,
    PH_TOK_PAD = 2,
    PH_TOK_AUDIO = 3,   /* audio conditioning placeholder (unused in v0 graphs) */
};

/* ---- hparams ------------------------------------------------------------- */
struct ph_hparams {
    int n_vocab;
    int n_layers;
    int n_embd;        /* == PH_EMBD validated at load */
    int n_head;
    int n_ffn;
    int n_ctx_train;
    float rope_theta;
    float eps_norm;
    /* speech */
    int   has_speech;
    int   mel_bins;
    int   enc_out_dim;   /* per-frame output dim of speech encoder */
    int   enc_frames_per_chunk; /* decoder tokens consumed per 100ms window */
    uint32_t sample_rate;
};

/* ---- weight registry: named ggml tensors built from GGUF data ------------ */
struct ph_weights {
    struct ggml_context* ctx_data;   /* owns tensor structs (no_alloc arena)  */
    ggml_backend_buffer_t buf;       /* persistent host/device weight buffer  */
    char*                base;       /* host-visible base of buf (memcpy dst) */
    size_t               buf_cursor; /* bump allocator into buf               */

    struct ggml_tensor* tok_emb;     /* [n_embd, n_vocab]  */
    struct ggml_tensor* out_norm_w;  /* [n_embd]           */
    struct ggml_tensor* out_norm_b;  /* [n_embd]           */
    struct ggml_tensor* lm_head;     /* [n_embd, n_vocab]  (untied option) */

    /* per layer, arrays of length n_layers */
    struct ggml_tensor** attn_norm_w;
    struct ggml_tensor** attn_norm_b;
    struct ggml_tensor** wq;   /* [n_embd, n_embd] */
    struct ggml_tensor** wk;   /* [n_head*kv_dim? keep full n_embd for simplicity] */
    struct ggml_tensor** wv;
    struct ggml_tensor** wo;
    struct ggml_tensor** ffn_norm_w;
    struct ggml_tensor** ffn_norm_b;
    struct ggml_tensor** w1;   /* [n_ffn, n_embd] */
    struct ggml_tensor** w2;   /* [n_embd, n_ffn] */

    /* speech encoder (has_speech only) */
    struct ggml_tensor* enc_proj_w;   /* [enc_out_dim, mel_bins] */
    struct ggml_tensor* enc_proj_b;   /* [enc_out_dim] */
    struct ggml_tensor* cond_w;       /* [n_embd, enc_out_dim] */
    struct ggml_tensor* cond_b;       /* [n_embd] */

    size_t total_bytes;
};

struct gguf_file; /* fwd from io/gguf_loader.h */

int  ph_hparams_read(struct gguf_file* g, struct ph_hparams* hp);
int  ph_weights_load(struct gguf_file* g, const struct ph_hparams* hp, struct ph_weights* w);
void ph_weights_free(struct ph_weights* w);

/* ---- graph builders (graph_text.c) --------------------------------------- */
struct ggml_cgraph;
/* Persistent KV cache: ONE host buffer (F32) sized 2*n_layers*n_ctx*HD floats.
 * Tensor structs live in a no_alloc context and their ->data points into the
 * buffer via ggml_backend_tensor_alloc + ggml_backend_view_init — allocated
 * once at load, NEVER reallocated as the sequence grows (PLAN.md rule). */
struct ph_kv_cache {
    struct ggml_context*      ctx;      /* tensor structs only (no_alloc)   */
    ggml_backend_buffer_t     buf;      /* persistent host data buffer      */
    struct ggml_tensor** k;             /* [n_layers] each [HD, n_ctx]      */
    struct ggml_tensor** v;
    int n_ctx;
    int cur_len;                        /* tokens currently stored          */
};

/* tokens_in must be a declared INPUT tensor (ggml_set_input) of shape [n], i32 */
struct ggml_cgraph* ph_build_graph(struct ggml_context* ctx, const struct ph_hparams* hp,
                                   const struct ph_weights* w, struct ph_kv_cache* kv,
                                   struct ggml_tensor* tokens_in, int n_tokens, int pos_begin,
                                   int logits_last_only);
int  ph_kv_alloc(struct ph_kv_cache* kv, const struct ph_hparams* hp, int n_ctx, ggml_backend_dev_t dev);
void ph_kv_free(struct ph_kv_cache* kv);
void ph_kv_rewind(struct ph_kv_cache* kv, int new_len, int n_layers);

/* persistent attention-merge scratch [E, max_n] F32; NULL reverts to arena alloc */
struct ggml_tensor;
void ph_set_merge_scratch(struct ggml_tensor* t, int max_n);

/* ---- tokenizer (tokenizer.c) --------------------------------------------- */
struct ph_vocab {
    char**   text;      /* n_vocab entries, raw pieces (no special leading space marker stripping) */
    uint8_t* is_special;
    int      n;
    int      bos, eos, pad, audio_tok;
};
void ph_vocab_load(struct gguf_file* g, struct ph_hparams* hp, struct ph_vocab* v);
void ph_vocab_free(struct ph_vocab* v);
/* greedy longest-match over the byte-level vocab; returns token count or -1 */
int  ph_tokenize(const struct ph_vocab* v, const char* text, int* ids, int max_n);
/* copy the piece for id into buf (NUL-terminated); returns buf */
const char* ph_detokenize_piece(const struct ph_vocab* v, int id, char* buf, size_t bufsz);

/* rope cos/sin tables [n_tok, hd/2] row-major; shared with runtime input fill */
void ph_rope_tables(float* cos_t, float* sin_t, int n_tok, int hd, int pos_begin, float theta);

/* ---- speech (graph_speech.c + audio front-end) --------------------------- */
struct ggml_cgraph* ph_build_speech_graph(struct ggml_context* ctx, const struct ph_hparams* hp,
                                          const struct ph_weights* w,
                                          struct ggml_tensor* mel_in /* [mel_bins, n_frames] */,
                                          int n_frames);

#endif /* PHONOS_MODEL_H */
