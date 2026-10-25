/* hparams.c — read phonon2 hyper-parameters + validate against fixed arch constants.
 * Metadata keys follow the GGUF convention used by llama.cpp ("general.*",
 * "phonon2.*") so real checkpoints and our synthetic validator share one format. */
#include "model.h"
#include "../io/gguf_loader.h"
#include "../core/phonos_log.h"

#include <stdio.h>
#include <string.h>

int ph_hparams_read(struct gguf_file* g, struct ph_hparams* hp) {
    memset(hp, 0, sizeof(*hp));

    const char* arch = gguf_get_str(g, "general.architecture");
    if (!arch || strcmp(arch, PH_ARCH_NAME) != 0) {
        phonos_log(PHONOS_LOG_ERROR, "hparams: architecture='%s' (expected '%s')",
                   arch ? arch : "(null)", PH_ARCH_NAME);
        return -1;
    }

    const int64_t P = gguf_get_int(g, "phonon2.embedding_length", PH_EMBD);

    hp->n_embd       = PH_EMBD;
    hp->n_vocab      = (int)gguf_get_int(g, "phonon2.vocab_size", 0);
    hp->n_layers     = (int)gguf_get_int(g, "phonon2.block_count", 0);
    hp->n_head       = (int)gguf_get_int(g, "phonon2.attention.head_count", PH_N_HEADS);
    hp->n_ffn        = (int)gguf_get_int(g, "phonon2.feed_forward_length", PH_FFN_H);
    hp->n_ctx_train  = (int)gguf_get_int(g, "phonon2.context_length", 1024);
    hp->rope_theta   = gguf_get_f32(g, "phonon2.rope.theta", PH_ROPE_THETA);
    hp->eps_norm     = gguf_get_f32(g, "phonon2.attention.layer_norm_rms_epsilon", 1e-5f);

    /* speech side is optional; presence of enc_proj.weight enables the path */
    hp->has_speech   = (int)gguf_get_int(g, "phonon2.has_speech", 0);
    hp->mel_bins     = (int)gguf_get_int(g, "phonon2.mel.bins", PH_MEL_BINS);
    hp->enc_out_dim  = (int)gguf_get_int(g, "phonon2.encoder.output_length", 0);
    hp->enc_frames_per_chunk = (int)gguf_get_int(g, "phonon2.encoder.frames_per_chunk", PH_ENC_OUT_PER_100MS);
    hp->sample_rate  = (uint32_t)gguf_get_int(g, "phonon2.audio.sample_rate", PH_MEL_SR);

    if (hp->n_vocab <= 0 || hp->n_layers <= 0 || hp->n_head <= 0 || hp->n_ffn <= 0) {
        phonos_log(PHONOS_LOG_ERROR, "hparams: invalid core dims (vocab=%d layers=%d heads=%d ffn=%d)",
                   hp->n_vocab, hp->n_layers, hp->n_head, hp->n_ffn);
        return -1;
    }
    if (PH_EMBD % hp->n_head != 0) {
        phonos_log(PHONOS_LOG_ERROR, "hparams: n_embd %% n_head != 0");
        return -1;
    }
    if (hp->has_speech && (hp->enc_out_dim <= 0 || hp->mel_bins <= 0)) {
        phonos_log(PHONOS_LOG_ERROR, "hparams: has_speech but encoder dims missing");
        return -1;
    }

    phonos_log(PHONOS_LOG_INFO,
               "hparams: vocab=%d layers=%d embd=%d heads=%d ffn=%d ctx=%d speech=%d",
               hp->n_vocab, hp->n_layers, hp->n_embd, hp->n_head, hp->n_ffn,
               hp->n_ctx_train, hp->has_speech);
    return 0;
}
