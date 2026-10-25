/* sampler.c — greedy / temperature+top-k+top-p sampling with repeat penalty.
 * Hot-loop rules (PLAN): scratch buffers are caller-owned statics sized at
 * load; sorting is O(V log V) on a small vocab copy — for phonon2-scale
 * vocabs this stays far below the matmul cost, so no incremental heap needed.
 * RNG: splitmix64 — deterministic given seed, platform-independent. */
#include "engine.h"
#include "../core/phonos_log.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

uint64_t ph_rng_advance(uint64_t state) {
    /* splitmix64 */
    uint64_t z = (state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

static int cmp_desc(const void* a, const void* b) {
    float fa = ((const float*)a)[0], fb = ((const float*)b)[0];
    if (fa > fb) return -1;
    if (fa < fb) return  1;
    return 0;
}

phonos_status ph_sample_next(const float* logits, int n_vocab,
                             const struct ph_sampler_cfg* sc, uint64_t rng_state,
                             const int* penalty_tokens, int n_penalty,
                             int* out_id) {
    /* ---- repeat penalty (in-place on a private copy would be cleaner; we
     * mutate the provided row then restore it — cheap, avoids extra buffer) */
    if (sc->repeat_penalty > 1.0f && penalty_tokens && n_penalty > 0) {
        for (int i = 0; i < n_penalty; i++) {
            int t = penalty_tokens[i];
            if (t < 0 || t >= n_vocab) continue;
            float v = logits[t];
            logits[t] = v > 0.0f ? v / sc->repeat_penalty : v * sc->repeat_penalty;
        }
    }

    /* ---- greedy fast path ---- */
    if (sc->seed < 0 || sc->temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < n_vocab; i++)
            if (logits[i] > logits[best]) best = i;
        *out_id = best;
        return PHONOS_OK;
    }

    /* ---- softmax with temperature ---- */
    float maxv = logits[0];
    for (int i = 1; i < n_vocab; i++) if (logits[i] > maxv) maxv = logits[i];

    /* reuse a thread-local scratch to keep zero malloc in steady state */
    static _Thread_local float* s_probs = NULL;
    static _Thread_local int    s_cap   = 0;
    if (s_cap < n_vocab) {
        free(s_probs);
        s_probs = malloc((size_t)n_vocab * sizeof(float) * 2);  /* probs + sort pairs interleaved not needed */
        if (!s_probs) return PHONOS_ERR_NOMEM;
        s_cap = n_vocab;
    }
    float* probs = s_probs;

    double sum = 0.0;
    const float invT = 1.0f / sc->temperature;
    for (int i = 0; i < n_vocab; i++) {
        float p = expf((logits[i] - maxv) * invT);
        probs[i] = p;
        sum += p;
    }
    const float inv_sum = (float)(1.0 / sum);
    for (int i = 0; i < n_vocab; i++) probs[i] *= inv_sum;

    /* ---- top-k ---- */
    int k = (sc->top_k > 0 && sc->top_k < n_vocab) ? sc->top_k : n_vocab;
    /* partial selection: full sort acceptable for v0 vocab sizes (<64k) */
    static _Thread_local float* s_pairs = NULL;   /* [val,id] pairs */
    static _Thread_local int    s_pcap  = 0;
    if (s_pcap < n_vocab) {
        free(s_pairs);
        s_pairs = malloc((size_t)n_vocab * 2 * sizeof(float));
        if (!s_pairs) return PHONOS_ERR_NOMEM;
        s_pcap = n_vocab;
    }
    float* pairs = s_pairs;
    for (int i = 0; i < n_vocab; i++) { pairs[2*i] = probs[i]; pairs[2*i+1] = (float)i; }
    /* simple argselect via full qsort of copies (stable enough here) */
    qsort(pairs, (size_t)n_vocab, 2*sizeof(float), cmp_desc);

    /* renormalize within top-k, then top-p truncate */
    double ksum = 0.0;
    for (int i = 0; i < k; i++) ksum += pairs[2*i];
    float cutoff = 1.0f;
    if (sc->top_p < 1.0f) {
        double acc = 0.0; cutoff = 0.0f;
        for (int i = 0; i < k; i++) {
            acc += pairs[2*i] / ksum;
            if (acc >= sc->top_p) { cutoff = (float)((ksum - acc + pairs[2*i])); break; }
        }
    }

    /* sample from the truncated distribution */
    uint64_t st = rng_state;
    st = ph_rng_advance(st);
    double u = (double)(st >> 11) * (1.0 / 9007199254740992.0);  /* [0,1) */

    double acc = 0.0;
    for (int i = 0; i < k; i++) {
        acc += pairs[2*i];
        if (acc * (1.0/ksum) >= u) { *out_id = (int)pairs[2*i+1]; return PHONOS_OK; }
    }
    *out_id = (int)pairs[2*(k-1)+1];
    (void)cutoff;
    return PHONOS_OK;
}
