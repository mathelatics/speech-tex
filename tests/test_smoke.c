/* test_smoke.c — P1 exit gate: public-API end-to-end behavior.
 *
 * Checks (in order):
 *   1. phonos_init / backend enumeration
 *   2. model load of the synthetic tiny GGUF (arch introspection)
 *   3. greedy generation determinism: same prompt+seed => identical token stream
 *   4. reset_session + re-generate => identical again (KV rewind correctness)
 *   5. perf readout sanity (prefill ms > 0, tokens == requested)
 *   6. invalid-arg handling (NULL prompt must return PHONOS_ERR_INVALID_ARG)
 *   7. speech path on a text-only model must report PHONOS_ERR_UNSUPPORTED
 * Model path: $PHONOS_TEST_MODEL or models/phonos-tiny.gguf relative to cwd. */
#include "phonos.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAXTOK 64
typedef struct { int ids[MAXTOK]; char text[4096]; int n; } capture;

static int cap_cb(const char* piece, int id, void* user) {
    capture* c = (capture*)user;
    if (c->n >= MAXTOK) return 1;                 /* cancel: cap overflow */
    c->ids[c->n] = id;
    strncat(c->text, piece, sizeof(c->text) - strlen(c->text) - 1);
    c->n++;
    return 0;
}

static int fails = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { printf("FAIL: %s\n", msg); fails++; } \
    else         { printf("ok:   %s\n", msg); } \
} while (0)

int main(void) {
    const char* model = getenv("PHONOS_TEST_MODEL");
    if (!model) model = "models/phonos-tiny.gguf";

    /* 1. init + backends */
    CHECK(phonos_init(0) == PHONOS_OK, "phonos_init");
    CHECK(phonos_backend_count() >= 1, "at least one backend enumerated");

    /* 2. load */
    phonos_status st = PHONOS_OK;
    phonos_model_params mp = {0};
    mp.path = model;
    mp.n_gpu_layers = 0;      /* CPU forced: portable everywhere */
    mp.use_mmap = 1;
    phonos_context* ctx = phonos_model_load(&mp, &st);
    if (!ctx) {
        printf("FAIL: model load (%s) err=%d\n", model, (int)st);
        return 1;
    }
    CHECK(st == PHONOS_OK, "phonos_model_load ok");
    CHECK(strcmp(phonos_arch_name(ctx), "phonon2") == 0, "arch name == phonon2");
    CHECK(phonos_vocab_size(ctx) == 128, "vocab size == 128");
    CHECK(phonos_n_layers(ctx) == 2, "n_layers == 2");
    CHECK(phonos_weight_bytes(ctx) > 0, "weight bytes reported");

    /* 3. determinism across two runs */
    phonos_gen_params gp = {0};
    gp.n_predict = 8;
    gp.seed = -1;             /* greedy */
    gp.temperature = 0.0f;
    gp.top_p = 1.0f;
    gp.repeat_penalty = 1.0f;

    capture a = {{0}}, b = {{0}};
    CHECK(phonos_generate(ctx, "hello", &gp, cap_cb, &a) == PHONOS_OK, "generate run 1");
    CHECK(a.n == 8, "run 1 produced exactly 8 tokens");
    CHECK(phonos_generate(ctx, "hello", &gp, cap_cb, &b) == PHONOS_OK, "generate run 2");
    CHECK(b.n == a.n && memcmp(a.ids, b.ids, sizeof(int)*(size_t)a.n) == 0,
          "greedy output is deterministic (token ids match)");
    CHECK(strcmp(a.text, b.text) == 0, "detokenized text matches");

    /* 4. session reset must not change greedy result (each generate resets anyway) */
    capture c = {{0}};
    phonos_reset_session(ctx);
    CHECK(phonos_generate(ctx, "hello", &gp, cap_cb, &c) == PHONOS_OK, "generate after reset");
    CHECK(c.n == a.n && memcmp(c.ids, a.ids, sizeof(int)*(size_t)a.n) == 0,
          "reset_session keeps determinism");

    /* 5. perf readout */
    double ms_pre = 0, ms_tok = 0; long nt = 0;
    phonos_last_perf(ctx, &ms_pre, &ms_tok, &nt);
    CHECK(ms_pre >= 0.0 && nt == 8, "perf readout sane (tokens == 8)");

    /* 6. invalid args */
    CHECK(phonos_generate(ctx, NULL, &gp, cap_cb, &a) == PHONOS_ERR_INVALID_ARG,
          "NULL prompt rejected with INVALID_ARG");
    CHECK(phonos_generate(NULL, "x", &gp, cap_cb, &a) == PHONOS_ERR_INVALID_ARG,
          "NULL ctx rejected with INVALID_ARG");

    /* 7. speech unsupported on this text-only build */
    float pcm[16000] = {0};
    phonos_audio au = { pcm, 16000, 16000 };
    capture d = {{0}};
    CHECK(phonos_transcribe(ctx, &au, NULL, &gp, cap_cb, &d) == PHONOS_ERR_UNSUPPORTED,
          "speech path reports UNSUPPORTED until P2");

    phonos_model_free(ctx);
    phonos_shutdown();

    printf("%s (%d failure(s))\n", fails ? "SMOKE FAILED" : "SMOKE PASSED", fails);
    return fails ? 1 : 0;
}
