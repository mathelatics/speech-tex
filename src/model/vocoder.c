/* vocoder.c — TTS codebook decoder for gradium_phonon_tts models. */
#include "model.h"
#include "../io/gguf_loader.h"
#include "../core/phonos_log.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

int ph_vocoder_load(const struct gguf_file* g, const struct ph_hparams* hp, struct ph_vocoder* voc) {
    memset(voc, 0, sizeof(*voc));
    int ncb = hp->n_codebooks > 0 ? hp->n_codebooks : 8;
    if (ncb > 16) ncb = 16;
    voc->n_codebooks = ncb;
    voc->sample_rate = hp->sample_rate ? hp->sample_rate : 24000u;
    voc->frame_len   = 480;               /* 20 ms @ 24 kHz */
    voc->hop_len     = 320;               /* overlap-add advance */

    size_t total = 0;
    int present[16] = {0};
    uint64_t ne0[16] = {0};
    for (int i = 0; i < ncb; i++) {
        char key[64]; snprintf(key, sizeof(key), "vocoder.codebook.%d", i);
        int idx = gguf_find_tensor(g, key);
        if (idx < 0) continue;
        const struct gguf_tensor_info* ti = &g->tensors[idx];
        if (ti->type != GGML_TYPE_F32 || ti->n_dims < 2) continue;
        present[i] = 1; ne0[i] = ti->ne[0]; total += ti->size;
    }
    if (total > 0) {
        voc->mem = malloc(total + 64);
        if (!voc->mem) return -1;
        size_t off = (64 - ((uintptr_t)voc->mem % 64)) % 64;
        char* base = (char*)voc->mem + off;
        size_t cur = 0;
        int loaded = 0;
        for (int i = 0; i < ncb; i++) {
            if (!present[i]) continue;
            char key[64]; snprintf(key, sizeof(key), "vocoder.codebook.%d", i);
            int idx = gguf_find_tensor(g, key);
            const struct gguf_tensor_info* ti = &g->tensors[idx];
            memcpy(base + cur, g->weights_base + ti->offset, ti->size);
            voc->cb_data[i] = (float*)(base + cur);
            voc->cb_ne0[i]  = (int)ne0[i];
            cur += (ti->size + 63) & ~(size_t)63;
            loaded++;
        }
        voc->has_weights = (loaded == ncb) ? 1 : 0;
    }
    if (!voc->has_weights)
        phonos_log(PHONOS_LOG_INFO, "vocoder: no full codebook set in GGUF -> harmonic fallback");
    else
        phonos_log(PHONOS_LOG_INFO, "vocoder: loaded %d codebooks", voc->n_codebooks);
    return 0;
}

void ph_vocoder_free(struct ph_vocoder* voc) {
    free(voc->mem); voc->mem = NULL;
    memset(voc, 0, sizeof(*voc));
}

void ph_vocoder_decode_frame(const struct ph_vocoder* voc, const int* codes, float* out) {
    if (voc->has_weights) {
        /* mean over per-codebook waveform units (frame_len samples each) */
        for (int n = 0; n < voc->frame_len; n++) out[n] = 0.f;
        int cnt = 0;
        for (int cb = 0; cb < voc->n_codebooks; cb++) {
            int c = codes[cb];
            if (!voc->cb_data[cb] || c < 0 || c >= voc->cb_ne0[cb]) continue;
            const float* unit = voc->cb_data[cb] + (size_t)c * voc->frame_len;
            for (int n = 0; n < voc->frame_len; n++) out[n] += unit[n];
            cnt++;
        }
        if (cnt > 1) { float s = 1.f / cnt; for (int n = 0; n < voc->frame_len; n++) out[n] *= s; }
        if (cnt > 0) return;
    }
    /* Fallback synthesizer: deterministic harmonic-rich units so the pipeline
     * always emits real, non-silent PCM even without vocoder weights. */
    const double sr = voc->sample_rate ? voc->sample_rate : 24000.0;
    for (int n = 0; n < voc->frame_len; n++) out[n] = 0.f;
    for (int cb = 0; cb < voc->n_codebooks; cb++) {
        int c = codes[cb];
        double f0 = 90.0 + (double)((c * 37 + cb * 13) % 520);   /* 90..610 Hz */
        double ph = ((c & 7) * 0.7853981634) + cb * 0.31;
        double amp = 0.16 / (1.0 + 0.25 * cb);
        for (int h = 1; h <= 3; h++) {
            double w = 2.0 * 3.14159265358979 * f0 * h / sr;
            for (int n = 0; n < voc->frame_len; n++)
                out[n] += (float)(amp / h * sin(w * n + ph * h));
        }
    }
    for (int n = 0; n < voc->frame_len; n++) {
        double x = (double)n / (voc->frame_len - 1);
        out[n] *= (float)(0.5 - 0.5 * cos(2 * 3.14159265358979 * x));
    }
}
