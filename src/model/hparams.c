/* hparams.c — read phonon2 / gradium_phonon_tts hyper-parameters from GGUF. */
#include "model.h"
#include "../io/gguf_loader.h"
#include "../core/phonos_log.h"

#include <stdio.h>
#include <string.h>

int ph_hparams_read(struct gguf_file* g, struct ph_hparams* hp) {
    memset(hp, 0, sizeof(*hp));

    const char* arch = gguf_get_str(g, "general.architecture");
    if (!arch) {
        phonos_log(PHONOS_LOG_ERROR, "hparams: missing general.architecture");
        return -1;
    }
    int is_tts = (strcmp(arch, PH_ARCH_TTS_NAME) == 0);
    if (!is_tts && strcmp(arch, PH_ARCH_NAME) != 0) {
        phonos_log(PHONOS_LOG_ERROR, "hparams: architecture='%s' (expected '%s' or '%s')",
                   arch, PH_ARCH_NAME, PH_ARCH_TTS_NAME);
        return -1;
    }
    hp->kind = is_tts ? PH_KIND_TTS : PH_KIND_STT;

    /* TTS checkpoints use llama.cpp-style keys ("general.*"); our reference
     * arch uses "phonon2.*". Read both with sensible defaults. */
    const char* pre = is_tts ? "general" : "phonon2";
    char key[96];
#define GET_I(def_, path_) (snprintf(key,sizeof(key),"%s." path_,pre), gguf_get_int(g,key,def_))
#define GET_F(def_, path_) (snprintf(key,sizeof(key),"%s." path_,pre), gguf_get_f32(g,key,def_))

    hp->n_embd       = is_tts ? (int)GET_I(512, "embedding_length") : PH_EMBD;
    hp->n_vocab      = (int)(is_tts ? GET_I(0, "vocab_size") : GET_I(0, "vocab_size"));
    if (hp->n_vocab <= 0) { /* llama convention */
        snprintf(key, sizeof(key), "%s.vocab_size", pre);
        hp->n_vocab = (int)gguf_get_int(g, key, 0);
    }
    hp->n_layers     = (int)GET_I(0, "block_count");
    hp->n_head       = (int)GET_I(PH_N_HEADS, "attention.head_count");
    hp->n_ffn        = (int)GET_I(hp->n_embd * PH_FFN_MULT, "feed_forward_length");
    hp->n_ctx_train  = (int)GET_I(1024, "context_length");
    hp->rope_theta   = GET_F(PH_ROPE_THETA, "rope.freq_base");
    if (hp->rope_theta <= 0.f) hp->rope_theta = PH_ROPE_THETA;
    hp->eps_norm     = GET_F(1e-5f, "attention.layer_norm_rms_epsilon");

    if (is_tts) {
        /* audio codebook parameters (gradium_phonon_tts-specific kv, optional) */
        hp->n_codebooks    = (int)gguf_get_int(g, "gradium.audio.codebooks", 8);
        hp->codebook_size  = (int)gguf_get_int(g, "gradium.audio.codebook_size", 1024);
        hp->frames_per_token = (int)gguf_get_int(g, "gradium.audio.frames_per_token", 2);
        hp->sample_rate    = (uint32_t)gguf_get_int(g, "gradium.audio.sample_rate", 24000);
        hp->n_text_tokens  = (int)gguf_get_int(g, "gradium.text.vocab_size", hp->n_vocab);
        hp->has_speech     = 0;
    } else {
        /* speech side is optional; presence of enc_proj.weight enables the path */
        hp->has_speech   = (int)GET_I(0, "has_speech");
        hp->mel_bins     = (int)GET_I(PH_MEL_BINS, "mel.bins");
        hp->enc_out_dim  = (int)GET_I(0, "encoder.output_length");
        hp->enc_frames_per_chunk = (int)GET_I(PH_ENC_OUT_PER_100MS, "encoder.frames_per_chunk");
        hp->sample_rate  = (uint32_t)GET_I(PH_MEL_SR, "audio.sample_rate");
    }
#undef GET_I
#undef GET_F

    if (hp->n_vocab <= 0 || hp->n_layers <= 0 || hp->n_head <= 0 || hp->n_ffn <= 0) {
        phonos_log(PHONOS_LOG_ERROR, "hparams: invalid core dims (vocab=%d layers=%d heads=%d ffn=%d)",
                   hp->n_vocab, hp->n_layers, hp->n_head, hp->n_ffn);
        return -1;
    }
    if (hp->n_embd % hp->n_head != 0) {
        phonos_log(PHONOS_LOG_ERROR, "hparams: n_embd %% n_head != 0");
        return -1;
    }
    if (hp->kind == PH_KIND_STT && !is_tts && hp->has_speech && (hp->enc_out_dim <= 0 || hp->mel_bins <= 0)) {
        phonos_log(PHONOS_LOG_ERROR, "hparams: has_speech but encoder dims missing");
        return -1;
    }

    phonos_log(PHONOS_LOG_INFO,
               "hparams: arch=%s vocab=%d layers=%d embd=%d heads=%d ffn=%d ctx=%d speech=%d",
               arch, hp->n_vocab, hp->n_layers, hp->n_embd, hp->n_head, hp->n_ffn,
               hp->n_ctx_train, hp->has_speech);
    return 0;
}
