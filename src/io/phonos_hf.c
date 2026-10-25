/* phonos_hf.c — minimal HF Hub downloader over HTTPS.
 *
 * Platform note: uses libcurl when available (link with -lcurl); falls back to
 * a plain-HTTP redirect-following reader via curl-free sockets is intentionally
 * NOT implemented to keep the surface small — instead we require libcurl at
 * build time when PHONOS_HF=ON (default). Without libcurl, resolve() only
 * accepts local paths and returns PHONOS_ERR_UNSUPPORTED for repo ids.
 */
#define _GNU_SOURCE
#include "phonos_hf.h"
#include "../core/phonos_log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>

#if defined(PHONOS_HAVE_CURL)
#include <curl/curl.h>
#endif

static void mkdir_p(const char* path) {
    char tmp[1024]; size_t n = strlen(path);
    if (n == 0 || n >= sizeof(tmp)) return;
    memcpy(tmp, path, n + 1);
    for (char* p = tmp + 1; *p; p++) {
        if (*p == '/') { *p = '\0'; mkdir(tmp, 0755); *p = '/'; }
    }
    mkdir(tmp, 0755);
}

static const char* hf_token(void) {
    static char tok[256];
    const char* t = getenv("PHONOS_HF_TOKEN");
    if (t && *t) return t;
    const char* home = getenv("HOME");
    if (!home) return NULL;
    snprintf(tok, sizeof(tok), "%s/.cache/huggingface/token", home);
    FILE* f = fopen(tok, "r");
    if (!f) return NULL;
    if (fgets(tok, sizeof(tok), f)) {
        size_t l = strlen(tok);
        while (l && (tok[l-1] == '\n' || tok[l-1] == '\r')) tok[--l] = 0;
        fclose(f);
        return l ? tok : NULL;
    }
    fclose(f);
    return NULL;
}

#if defined(PHONOS_HAVE_CURL)
typedef struct { FILE* fp; } wctx;
static size_t wcb(void* ptr, size_t sz, size_t nm, void* ud) {
    wctx* w = (wctx*)ud;
    return fwrite(ptr, sz, nm, w->fp);
}

/* returns 0 ok, 1 needs-access, 2 notfound, 3 neterr */
static int curl_download(const char* url, const char* dst) {
    CURL* c = curl_easy_init();
    if (!c) return 3;
    char err[CURL_ERROR_SIZE] = {0};
    struct curl_slist* hdrs = NULL;
    const char* tok = hf_token();
    if (tok) {
        char h[512]; snprintf(h, sizeof(h), "Authorization: Bearer %s", tok);
        hdrs = curl_slist_append(hdrs, h);
    }
    char tmp[1024]; snprintf(tmp, sizeof(tmp), "%s.part", dst);
    FILE* fp = fopen(tmp, "wb");
    if (!fp) { curl_easy_cleanup(c); if (hdrs) curl_slist_free_all(hdrs); return 3; }
    wctx w = { fp };
    curl_easy_setopt(c, CURLOPT_URL, url);
    curl_easy_setopt(c, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(c, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(c, CURLOPT_WRITEFUNCTION, wcb);
    curl_easy_setopt(c, CURLOPT_WRITEDATA, &w);
    curl_easy_setopt(c, CURLOPT_ERRORBUFFER, err);
    curl_easy_setopt(c, CURLOPT_FAILONERROR, 0L);
    curl_easy_setopt(c, CURLOPT_CONNECTTIMEOUT, 20L);
    curl_easy_setopt(c, CURLOPT_LOW_SPEED_LIMIT, 1024L);
    curl_easy_setopt(c, CURLOPT_LOW_SPEED_TIME, 60L);
    if (hdrs) curl_easy_setopt(c, CURLOPT_HTTPHEADER, hdrs);

    CURLcode rc = curl_easy_perform(c);
    long code = 0;
    curl_easy_getinfo(c, CURLINFO_RESPONSE_CODE, &code);
    fclose(fp);
    if (hdrs) curl_slist_free_all(hdrs);
    curl_easy_cleanup(c);

    int ret = 0;
    if (rc != CURLE_OK) { PHO_LOG_ERROR("hf: %s: %s", url, err); ret = 3; }
    else if (code == 401 || code == 403) ret = 1;
    else if (code == 404) ret = 2;
    else if (code >= 200 && code < 300) {
        if (rename(tmp, dst) != 0) ret = 3;
    } else ret = 3;
    if (ret) unlink(tmp);
    return ret;
}
#endif /* PHONOS_HAVE_CURL */

phonos_status phonos_hf_resolve(const char* spec, const char* want_file, char** out_path) {
    if (!spec || !out_path) return PHONOS_ERR_INVALID_ARG;
    *out_path = NULL;

    /* local path? */
    struct stat st;
    if (stat(spec, &st) == 0 && S_ISREG(st.st_mode)) {
        *out_path = strdup(spec);
        return *out_path ? PHONOS_OK : PHONOS_ERR_NOMEM;
    }

    /* parse org/name[/file] */
    const char* s1 = strchr(spec, '/');
    if (!s1) { PHO_LOG_ERROR("hf: '%s' is neither a file nor an 'org/name[/file.gguf]' repo id", spec);
               return PHONOS_ERR_INVALID_ARG; }
    const char* s2 = strchr(s1 + 1, '/');
    char org[256], name[256], file[256];
    size_t lo = (size_t)(s1 - spec); if (lo >= sizeof(org)) return PHONOS_ERR_INVALID_ARG;
    memcpy(org, spec, lo); org[lo] = 0;
    if (s2) {
        size_t ln = (size_t)(s2 - s1 - 1); if (ln >= sizeof(name)) return PHONOS_ERR_INVALID_ARG;
        memcpy(name, s1 + 1, ln); name[ln] = 0;
        snprintf(file, sizeof(file), "%s", s2 + 1);
    } else {
        snprintf(name, sizeof(name), "%s", s1 + 1);
        if (!want_file || !*want_file) {
            PHO_LOG_ERROR("hf: repo '%s' given without a file; pass e.g. %s/model.q8.gguf", spec, spec);
            return PHONOS_ERR_INVALID_ARG;
        }
        snprintf(file, sizeof(file), "%s", want_file);
    }

    const char* rev = getenv("PHONOS_HF_REVISION");
    if (!rev || !*rev) rev = "main";

    const char* base = getenv("PHONOS_CACHE_DIR");
    char cache[1024];
    if (base && *base) snprintf(cache, sizeof(cache), "%s/hub/%s/%s", base, org, name);
    else {
        const char* home = getenv("HOME");
        if (!home) home = "/tmp";
        snprintf(cache, sizeof(cache), "%s/.cache/phonos/hub/%s/%s", home, org, name);
    }
    mkdir_p(cache);

    char local[1200];
    snprintf(local, sizeof(local), "%s/%s", cache, file);
    if (stat(local, &st) == 0 && S_ISREG(st.st_mode) && st.st_size > 0) {
        PHO_LOG_INFO("hf: using cached %s", local);
        *out_path = strdup(local);
        return *out_path ? PHONOS_OK : PHONOS_ERR_NOMEM;
    }

#if defined(PHONOS_HAVE_CURL)
    char url[1200];
    snprintf(url, sizeof(url), "https://huggingface.co/%s/%s/resolve/%s/%s", org, name, rev, file);
    PHO_LOG_INFO("hf: downloading %s", url);
    int r = curl_download(url, local);
    if (r == 0) { *out_path = strdup(local); return *out_path ? PHONOS_OK : PHONOS_ERR_NOMEM; }
    if (r == 1) {
        PHO_LOG_ERROR("hf: '%s/%s' is gated or requires authentication."
                      " Accept the model license on https://huggingface.co/%s/%s then"
                      " export PHONOS_HF_TOKEN=<your token> (or `hf auth login`),"
                      " or download the GGUF manually and pass the local path.",
                      org, name, org, name);
        return PHONOS_NEEDS_ACCESS;
    }
    if (r == 2) { PHO_LOG_ERROR("hf: file not found in repo: %s", file); return PHONOS_ERR_IO; }
    return PHONOS_ERR_IO;
#else
    PHO_LOG_ERROR("hf: engine built without libcurl support (PHONOS_HF=OFF);"
                  " download the GGUF manually and pass its local path.");
    return PHONOS_ERR_UNSUPPORTED;
#endif
}
