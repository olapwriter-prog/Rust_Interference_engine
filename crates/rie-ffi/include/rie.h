/*
 * rie.h — C API of the Rust Inference Engine (librie.so / librie.a).
 *
 * Thread-safety: one engine must not be used from two threads at once, except for
 * rie_cancel(), which may be called from any thread to stop a running generation.
 */
#ifndef RIE_H
#define RIE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RieEngine RieEngine;

typedef struct RieConfig {
    int32_t n_threads;   /* 0 = auto: performance cores only */
    int32_t n_ctx;       /* context window, e.g. 4096 */
    int32_t n_batch;     /* prompt batch size, e.g. 64 */
    int32_t use_gpu;     /* 1 = Vulkan/Metal backend (library built with "gpu" feature) */
    int32_t pin_threads; /* 1 = pin workers to performance cores (Android/Linux) */
    int32_t mlock;       /* 1 = lock weights in RAM */
} RieConfig;

typedef struct RieSampling {
    float temperature;   /* 0 = greedy */
    int32_t top_k;
    float top_p;
    float min_p;
    float repeat_penalty;
    uint64_t seed;       /* 0 = random */
    int32_t max_tokens;
} RieSampling;

typedef struct RieStats {
    int32_t prompt_tokens;
    int32_t reused_tokens;   /* served from the KV cache */
    int32_t generated_tokens;
    double prompt_tps;
    double gen_tps;
    int32_t stop_reason;     /* 0 eos, 1 max tokens, 2 stop string, 3 ctx full, 4 cancelled */
} RieStats;

/* Receives UTF-8 text pieces as they are generated. Return 0 to stop. */
typedef int32_t (*RieTokenCallback)(const char *piece, void *user);

RieConfig rie_default_config(void);
RieSampling rie_default_sampling(void);

/* Returns NULL on failure; the reason is written to err (if non-NULL). */
RieEngine *rie_load(const char *gguf_path, const RieConfig *cfg, char *err, size_t err_len);
void rie_free(RieEngine *engine);

int32_t rie_generate(RieEngine *engine, const char *prompt, const RieSampling *sampling,
                     RieTokenCallback cb, void *user, RieStats *stats, char *err, size_t err_len);

/* roles[i] is "system", "user" or "assistant". Shared prefixes reuse the KV cache. */
int32_t rie_chat(RieEngine *engine, const char *const *roles, const char *const *contents, int32_t n,
                 const RieSampling *sampling, RieTokenCallback cb, void *user, RieStats *stats,
                 char *err, size_t err_len);

void rie_cancel(const RieEngine *engine);
void rie_reset(RieEngine *engine);
int32_t rie_bench(RieEngine *engine, int32_t n_prompt, int32_t n_gen, double *pp_tps, double *tg_tps);

/* Returned strings must be released with rie_string_free. */
char *rie_device_info(void);
char *rie_model_info(const RieEngine *engine);
void rie_string_free(char *s);

#ifdef __cplusplus
}
#endif
#endif /* RIE_H */
