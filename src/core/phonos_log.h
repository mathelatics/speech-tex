#ifndef PHONOS_LOG_H
#define PHONOS_LOG_H

#include "phonos.h"

#define phonos_log(lvl, ...) phonos_logv((lvl), __VA_ARGS__)

void phonos_logv(enum phonos_log_level lvl, const char* fmt, ...)
#ifdef __GNUC__
    __attribute__((format(printf, 2, 3)))
#endif
    ;

/* route ggml internal logs through the phonos logger (call after registry init) */
void phonos_install_ggml_logger(void);

#endif /* PHONOS_LOG_H */
