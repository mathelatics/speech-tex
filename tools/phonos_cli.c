/* phonos_cli.c — reference command-line front-end for the engine.
 * Subcommands:
 *   info  <model.gguf>                 : arch + backend capability report
 *   gen   <model.gguf> --prompt "..."  : text generation (streaming callback)
 *   bench <model.gguf> [--n N]         : prefill/decode throughput readout
 * This file only uses the public API (include/phonos.h) — exactly what JNI /
 * Swift / desktop wrappers will see, so it doubles as an API usability test. */
#include "phonos.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_first = 1;

static int on_token(const char* piece, int id, void* user) {
    (void)id; (void)user;
    fputs(piece, stdout);
    fflush(stdout);
    return 0;   /* 0 => keep generating */
}

static void print_usage(const char* argv0) {
    fprintf(stderr,
        "usage:\n"
        "  %s info  MODEL.gguf\n"
        "  %s gen   MODEL.gguf --prompt TEXT [--n N] [--temp T] [--seed S] [--threads K] [--gpu L]\n"
        "  %s bench MODEL.gguf --prompt TEXT [--n N] [--threads K]\n",
        argv0, argv0, argv0);
}

int main(int argc, char** argv) {
    if (argc < 3) { print_usage(argv[0]); return 2; }
    const char* cmd = argv[1];
    const char* path = argv[2];

    /* ---- arg parsing (tiny hand-rolled; no deps) ---- */
    const char* prompt = NULL;
    int n_predict = 32, seed = -1, threads = 0, gpu_layers = 0;
    float temp = 0.0f;
    for (int i = 3; i < argc; i++) {
        #define NEED_NEXT() do { if (++i >= argc) { fprintf(stderr, "missing value for %s\n", argv[i-1]); return 2; } } while (0)
        if (!strcmp(argv[i], "--prompt"))      { NEED_NEXT(); prompt = argv[i]; }
        else if (!strcmp(argv[i], "--n"))      { NEED_NEXT(); n_predict = atoi(argv[i]); }
        else if (!strcmp(argv[i], "--seed"))   { NEED_NEXT(); seed = atoi(argv[i]); }
        else if (!strcmp(argv[i], "--threads")){ NEED_NEXT(); threads = atoi(argv[i]); }
        else if (!strcmp(argv[i], "--gpu"))    { NEED_NEXT(); gpu_layers = atoi(argv[i]); }
        else if (!strcmp(argv[i], "--temp"))   { NEED_NEXT(); temp = (float)atof(argv[i]); }
        else { fprintf(stderr, "unknown flag: %s\n", argv[i]); return 2; }
        #undef NEED_NEXT
    }

    phonos_status st = phonos_init(threads);
    if (st != PHONOS_OK) { fprintf(stderr, "phonos_init failed (%d)\n", (int)st); return 1; }

    printf("backends:");
    for (int i = 0; i < phonos_backend_count(); i++)
        printf(" %s%s", phonos_backend_name(i), phonos_backend_is_cpu(i) ? "(cpu)" : "");
    printf("\n");

    phonos_model_params mp = {0};
    mp.path = path;
    mp.n_gpu_layers = gpu_layers;
    mp.n_threads = threads;
    phonos_context* ctx = phonos_model_load(&mp, &st);
    if (!ctx) {
        fprintf(stderr, "load failed: %d\n", (int)st);
        phonos_shutdown();
        return 1;
    }

    if (!strcmp(cmd, "info")) {
        printf("arch        : %s\n", phonos_arch_name(ctx));
        printf("vocab_size  : %d\n", phonos_vocab_size(ctx));
        printf("n_layers    : %d\n", phonos_n_layers(ctx));
        printf("n_ctx_train : %d\n", phonos_n_ctx_train(ctx));
        printf("speech      : %s\n", phonos_has_speech(ctx) ? "yes" : "no");
        double mb = (double)phonos_weight_bytes(ctx) / 1048576.0;
        printf("weights     : %.2f MiB\n", mb);
    } else if (!strcmp(cmd, "gen") || !strcmp(cmd, "bench")) {
        if (!prompt) { fprintf(stderr, "%s requires --prompt\n", cmd); return 2; }
        phonos_gen_params gp = {0};
        gp.n_predict = n_predict;
        gp.seed = seed;
        gp.temperature = temp;
        gp.top_k = 0;
        gp.top_p = 1.0f;
        gp.repeat_penalty = 1.0f;
        st = phonos_generate(ctx, prompt, &gp, on_token, NULL);
        printf("\n");
        if (st != PHONOS_OK) { fprintf(stderr, "generate failed: %d\n", (int)st); return 1; }
        double ms_pre = 0, ms_tok = 0; long nt = 0;
        phonos_last_perf(ctx, &ms_pre, &ms_tok, &nt);
        printf("--- perf: prefill %.1f ms | decode %.2f ms/token (%ld tok/s) | generated %ld tokens\n",
               ms_pre, ms_tok, (long)(ms_tok > 0 ? 1000.0/ms_tok : 0), nt);
    } else {
        print_usage(argv[0]);
        return 2;
    }

    phonos_model_free(ctx);
    phonos_shutdown();
    (void)g_first;
    return 0;
}
