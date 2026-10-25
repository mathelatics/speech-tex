/* phonos_hf.h — minimal Hugging Face Hub downloader (stdlib only, no deps).
 *
 * Supports:  repo "org/name" or "org/name/file.gguf", revision via env
 *            PHONOS_HF_REVISION (default "main"), token via env
 *            PHONOS_HF_TOKEN or file ~/.cache/huggingface/token.
 * Gated repos return PHONOS_NEEDS_ACCESS with a clear log message telling the
 * user to accept the model license on the website and export a token.
 *
 * Download is atomic (tmp file + rename) and cached under
 * $PHONOS_CACHE_DIR or ~/.cache/phonos/hub/<org>/<name>/<file>.
 */
#ifndef PHONOS_HF_H
#define PHONOS_HF_H

#include "phonos.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Resolve `spec` (a local path, or an HF repo id possibly with a file name)
 * to a local file path. If spec names a repo without a file, `want_file`
 * selects the default artifact (e.g. "model.q8.gguf"). On success writes an
 * owned malloc'd path into *out_path (caller frees). Returns PHONOS_OK or an
 * error (PHONOS_NEEDS_ACCESS for 401/403 on gated repos). */
phonos_status phonos_hf_resolve(const char* spec, const char* want_file, char** out_path);

#ifdef __cplusplus
}
#endif
#endif /* PHONOS_HF_H */
