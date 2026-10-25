/* backend_dispatch.c — the ONLY file allowed to talk to ggml devices.
 * Model/audio code never references backends directly (layering rule).
 * CPU is the universal fallback; GPU chosen only if usable + requested. */
#include "engine.h"
#include "../core/phonos_log.h"

#include <ggml.h>
#include <ggml-backend.h>
#ifdef GGML_USE_CPU
#include <ggml-cpu.h>
#endif

#include <omp.h>
#include <stdio.h>
#include <string.h>

static int g_n_threads = 0;
static int g_initialized = 0;

static void fill_backend_names(void);

int engine_n_threads(void) { return g_n_threads > 0 ? g_n_threads : 1; }

/* how many layers to hand to the accelerator (0 => pure CPU path) */
static int g_n_gpu_layers = 0;
int be_n_gpu_layers(void) { return g_n_gpu_layers; }

/* per-graph CPU thread override (ggml-cpu.h), applied before sched compute */
void be_set_cpu_threads(ggml_backend_t cpu_be, int n_threads) {
#ifdef GGML_USE_CPU
    if (cpu_be) ggml_backend_cpu_set_n_threads(cpu_be, n_threads);
#else
    (void)cpu_be; (void)n_threads;
#endif
}

phonos_status engine_init(int n_threads_hint) {
    if (g_initialized) return PHONOS_OK;

    /* one-time device enumeration (loads dynamic backends if built that way) */
    fill_backend_names();

    if (n_threads_hint <= 0) {
        n_threads_hint = omp_get_max_threads();
        if (n_threads_hint <= 0) n_threads_hint = 1;
    }
    g_n_threads = n_threads_hint;
    g_initialized = 1;
    phonos_log(PHONOS_LOG_INFO, "runtime: %d threads, %d backend device(s)",
               g_n_threads, engine_backend_count());
    return PHONOS_OK;
}

void engine_shutdown(void) {
    /* registry is process-global in ggml; nothing to tear down per-engine here */
    g_initialized = 0;
}

/* ---- device selection ---------------------------------------------------- */
/* Returns NULL => caller must use CPU-only path. Honors n_gpu_layers==0 as
 * "CPU forced" so low-power/mobile configs never accidentally touch a GPU. */
ggml_backend_dev_t engine_pick_device(int n_gpu_layers) {
    if (n_gpu_layers <= 0) return NULL;   /* NULL => CPU */

    size_t n = ggml_backend_dev_count();
    /* prefer accelerator-type devices, keep CPU as last resort */
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (ggml_backend_dev_get_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU) continue;
        /* sanity: device must have an active backend instance */
        ggml_backend_t b = ggml_backend_dev_init(d, NULL);
        if (!b) continue;
        ggml_backend_free(b);
        phonos_log(PHONOS_LOG_INFO, "runtime: selected device '%s' (%s)",
                   ggml_backend_dev_name(d), ggml_backend_dev_description(d));
        return d;
    }
    phonos_log(PHONOS_LOG_WARN, "runtime: no GPU available, falling back to CPU");
    return NULL;
}

/* ---- public backend list (for --list-backends / capability probing) ------ */
#define MAX_BACKENDS 16
static char  g_names[MAX_BACKENDS][64];
static int   g_is_cpu[MAX_BACKENDS];
static int   g_count = 0;

static void fill_backend_names(void) {
    g_count = 0;
    size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n && g_count < MAX_BACKENDS; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        snprintf(g_names[g_count], sizeof(g_names[0]), "%s", ggml_backend_dev_name(d));
        g_is_cpu[g_count] = (ggml_backend_dev_get_type(d) == GGML_BACKEND_DEVICE_TYPE_CPU);
        g_count++;
    }
}

int engine_backend_count(void) { return g_count; }

const char* engine_backend_name(int i) {
    if (i < 0 || i >= g_count) return "";
    return g_names[i];
}

int engine_backend_is_cpu(int i) {
    if (i < 0 || i >= g_count) return 0;
    return g_is_cpu[i];
}

/* ---- resolved thread count (used by runtime.c at load) ------------------- */
int engine_n_threads_impl(void) { return engine_n_threads(); }

/* ---- backend registry helpers called from runtime.c ---------------------- */
phonos_status be_init(int n_threads_hint) { return engine_init(n_threads_hint); }
void          be_shutdown(void)           { engine_shutdown(); }
int           be_backend_count(void)      { return engine_backend_count(); }
const char*   be_backend_name(int i)      { return engine_backend_name(i); }
int           be_backend_is_cpu(int i)    { return engine_backend_is_cpu(i); }
