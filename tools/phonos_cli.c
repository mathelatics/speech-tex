/* phonos_cli.c — reference command-line front-end for the engine.
 * Subcommands:
 *   info  <model.gguf>                 : arch + backend capability report
 *   gen   <model.gguf> --prompt "..."  : text generation (streaming callback)
 *   bench <model.gguf> [--n N]         : prefill/decode throughput readout
 *   tts   <model.gguf> --text "..." --out out.wav : speech synthesis
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

/* ---- TTS sink: accumulate PCM frames, write WAV at the end ---- */
struct wav_sink {
    FILE*  f;
    long   total_samples;
    int    sample_rate;
};

static int on_pcm(const float* pcm, int n_frames, void* user) {
    struct wav_sink* w = (struct wav_sink*)user;
    /* float32 [-1,1] -> int16 little-endian */
    static short buf[4096];
    int done = 0;
    while (done < n_frames) {
        int take = n_frames - done; if (take > 4096) take = 4096;
        for (int i = 0; i < take; i++) {
            float v = pcm[done+i];
            if (v >  1.f) v =  1.f;
            if (v < -1.f) v = -1.f;
            buf[i] = (short)(v * 32767.f);
        }
        fwrite(buf, sizeof(short), (size_t)take, w->f);
        w->total_samples += take;
        done += take;
    }
    return 0;   /* keep synthesizing */
}

static void wav_write_header(FILE* f, int sample_rate, long samples) {
    unsigned ds = (unsigned)samples * 2u;             /* data size   */
    unsigned hs = 36u + ds;                           /* RIFF size   */
    unsigned sr = (unsigned)sample_rate;
    unsigned char h[44];
    memcpy(h+0,  "RIFF", 4);
    h[4]=(unsigned char)(hs&0xff); h[5]=(unsigned char)((hs>>8)&0xff);
    h[6]=(unsigned char)((hs>>16)&0xff); h[7]=(unsigned char)((hs>>24)&0xff);
    memcpy(h+8,  "WAVEfmt ", 8);
    h[16]=16; h[17]=0; h[18]=0; h[19]=0;              /* PCM fmt chunk */
    h[20]=1; h[21]=0;                                 /* format = PCM  */
    h[22]=1; h[23]=0;                                 /* channels = 1  */
    h[24]=(unsigned char)(sr&0xff); h[25]=(unsigned char)((sr>>8)&0xff);
    h[26]=(unsigned char)((sr>>16)&0xff); h[27]=(unsigned char)((sr>>24)&0xff);
    unsigned br = sr * 2u;                            /* byte rate (mono s16) */
    h[28]=(unsigned char)(br&0xff); h[29]=(unsigned char)((br>>8)&0xff);
    h[30]=(unsigned char)((br>>16)&0xff); h[31]=(unsigned char)((br>>24)&0xff);
    h[32]=2; h[33]=0;                                 /* block align */
    h[34]=16; h[35]=0;                                /* bits/sample */
    h[36]='d'; h[37]='a'; h[38]='t'; h[39]='a';
    h[40]=(unsigned char)(ds&0xff); h[41]=(unsigned char)((ds>>8)&0xff);
    h[42]=(unsigned char)((ds>>16)&0xff); h[43]=(unsigned char)((ds>>24)&0xff);
    rewind(f);
    fwrite(h, 1, 44, f);
    fclose(f);
}

static void print_usage(const char* argv0) {
    fprintf(stderr,
        "usage:\n"
        "  %s info  MODEL.gguf\n"
        "  %s gen   MODEL.gguf --prompt TEXT [--n N] [--temp T] [--seed S] [--threads K] [--gpu L]\n"
        "  %s bench MODEL.gguf --prompt TEXT [--n N] [--threads K]\n"
        "  %s tts   MODEL.gguf --text TEXT --out OUT.wav [--n N] [--seed S] [--threads K]\n",
        argv0, argv0, argv0, argv0);
}

int main(int argc, char** argv) {
    if (argc < 3) { print_usage(argv[0]); return 2; }
    const char* cmd = argv[1];
    const char* path = argv[2];

    /* ---- arg parsing (tiny hand-rolled; no deps) ---- */
    const char* prompt = NULL;
    const char* out_wav = NULL;
    int n_predict = 32, seed = -1, threads = 0, gpu_layers = 0;
    float temp = 0.0f;
    for (int i = 3; i < argc; i++) {
        #define NEED_NEXT() do { if (++i >= argc) { fprintf(stderr, "missing value for %s\n", argv[i-1]); return 2; } } while (0)
        if (!strcmp(argv[i], "--prompt") || !strcmp(argv[i], "--text")) { NEED_NEXT(); prompt = argv[i]; }
        else if (!strcmp(argv[i], "--out"))     { NEED_NEXT(); out_wav = argv[i]; }
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
        printf("speech-in   : %s\n", phonos_has_speech(ctx) ? "yes" : "no");
        printf("speech-out  : %s\n", phonos_has_synth(ctx) ? "yes (TTS)" : "no");
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
    } else if (!strcmp(cmd, "tts")) {
        if (!prompt) { fprintf(stderr, "tts requires --text\n"); return 2; }
        if (!out_wav) { fprintf(stderr, "tts requires --out FILE.wav\n"); return 2; }
        if (!phonos_has_synth(ctx)) {
            fprintf(stderr, "model is not TTS-capable (arch=%s)\n", phonos_arch_name(ctx));
            return 1;
        }
        int sr = phonos_synth_sample_rate(ctx);
        if (sr <= 0) sr = 24000;
        struct wav_sink sink = { NULL, 0, sr };
        sink.f = fopen(out_wav, "wb");
        if (!sink.f) { fprintf(stderr, "cannot open %s\n", out_wav); return 1; }
        fwrite("XXXX", 1, 44, sink.f);   /* placeholder header, patched at end */
        phonos_gen_params gp = {0};
        gp.n_predict = n_predict < 32 ? 512 : n_predict;   /* budget = LM tokens */
        gp.seed = seed;
        gp.temperature = temp;
        st = phonos_synth(ctx, prompt, &gp, on_pcm, &sink);
        wav_write_header(sink.f, sr, sink.total_samples);
        if (st != PHONOS_OK && st != PHONOS_ERR_CANCELLED) {
            fprintf(stderr, "synth failed: %d\n", (int)st); return 1;
        }
        printf("wrote %s: %ld samples @ %d Hz (%.2f s)\n",
               out_wav, sink.total_samples, sr, (double)sink.total_samples / sr);
    } else {
        print_usage(argv[0]);
        return 2;
    }

    phonos_model_free(ctx);
    phonos_shutdown();
    (void)g_first;
    return 0;
}
