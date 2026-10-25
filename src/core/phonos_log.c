/* phonos_log.c — engine-wide logging + ggml log redirection.
 * One funnel for every message so platform adapters (JNI/Swift) can capture
 * all output with a single callback install. */
#include "phonos_log.h"

#include <ggml.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include <stdlib.h>

static enum phonos_log_level g_min   = PHONOS_LOG_INFO;
static phonos_log_cb         g_cb    = NULL;
static void*                 g_user  = NULL;

void phonos_set_log(enum phonos_log_level min_lvl, phonos_log_cb cb, void* user) {
    g_min  = min_lvl;
    g_cb   = cb;
    g_user = user;
}

static const char* lvl_tag(enum phonos_log_level l) {
    switch (l) {
        case PHONOS_LOG_DEBUG: return "[debug] ";
        case PHONOS_LOG_INFO:  return "[info]  ";
        case PHONOS_LOG_WARN:  return "[warn]  ";
        default:               return "[error] ";
    }
}

void phonos_logv(enum phonos_log_level lvl, const char* fmt, ...) {
    if (lvl < g_min) return;
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);

    if (g_cb) {
        g_cb(lvl, buf, g_user);
    } else {
        fprintf(lvl >= PHONOS_LOG_WARN ? stderr : stdout, "%s%s\n", lvl_tag(lvl), buf);
    }
}

/* Bridge ggml's internal logger into ours so backend messages obey the same
 * level/callback. Installed once in phonos_init. */
static void ggml_log_bridge(enum ggml_log_level lvl, const char* msg, void* user) {
    (void)user;
    enum phonos_log_level p = PHONOS_LOG_INFO;
    switch (lvl) {
        case GGML_LOG_LEVEL_ERROR: p = PHONOS_LOG_ERROR; break;
        case GGML_LOG_LEVEL_WARN:  p = PHONOS_LOG_WARN;  break;
        case GGML_LOG_LEVEL_DEBUG: p = PHONOS_LOG_DEBUG; break;
        default: break;
    }
    /* strip trailing newline ggml adds */
    size_t n = strlen(msg);
    char tmp[1024];
    if (n > 0 && msg[n-1] == '\n') n--;
    if (n >= sizeof(tmp)) n = sizeof(tmp)-1;
    memcpy(tmp, msg, n);
    tmp[n] = '\0';
    phonos_log(p, "%s", tmp);
}

void phonos_install_ggml_logger(void) {
    ggml_log_set(ggml_log_bridge, NULL);
}
