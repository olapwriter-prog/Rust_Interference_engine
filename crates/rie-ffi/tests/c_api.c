/* Smoke test of the C API: load, generate, chat with KV reuse, cancel, info. */
#include "rie.h"
#include <stdio.h>
#include <string.h>

static int count = 0;
static int on_piece(const char *piece, void *user) {
    (void)user;
    fputs(piece, stdout);
    return ++count < 1000;
}
static int stop_after_3(const char *piece, void *user) {
    (void)piece;
    int *n = (int *)user;
    return ++(*n) < 3;
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    char err[256] = {0};
    RieConfig cfg = rie_default_config();
    cfg.n_ctx = 256;
    cfg.n_threads = 2;
    RieEngine *e = rie_load(argv[1], &cfg, err, sizeof err);
    if (!e) { fprintf(stderr, "load failed: %s\n", err); return 1; }
    char *info = rie_model_info(e);
    printf("model: %s\n", info);
    rie_string_free(info);

    RieSampling s = rie_default_sampling();
    s.temperature = 0; s.max_tokens = 16; s.seed = 42;
    RieStats st;
    if (rie_generate(e, "the phone", &s, on_piece, NULL, &st, err, sizeof err) != 0) { fprintf(stderr, "gen: %s\n", err); return 1; }
    printf("\n[generated %d tokens, stop %d]\n", st.generated_tokens, st.stop_reason);
    if (st.generated_tokens <= 0) return 1;

    const char *roles[] = {"system", "user"};
    const char *contents[] = {"You are helpful.", "Hi"};
    if (rie_chat(e, roles, contents, 2, &s, on_piece, NULL, &st, err, sizeof err) != 0) { fprintf(stderr, "chat: %s\n", err); return 1; }
    if (rie_chat(e, roles, contents, 2, &s, on_piece, NULL, &st, err, sizeof err) != 0) return 1;
    printf("\n[second chat reused %d of %d prompt tokens]\n", st.reused_tokens, st.prompt_tokens);
    if (st.reused_tokens <= 0) return 1;

    int n = 0;
    s.max_tokens = 100;
    rie_generate(e, "fast", &s, stop_after_3, &n, &st, err, sizeof err);
    printf("[callback stop: stop_reason=%d after %d pieces]\n", st.stop_reason, n);
    if (st.stop_reason != 4) return 1;

    double pp = 0, tg = 0;
    if (rie_bench(e, 16, 8, &pp, &tg) != 0) return 1;
    printf("[bench pp %.1f t/s, tg %.1f t/s]\n", pp, tg);
    rie_free(e);

    if (rie_load("/nonexistent.gguf", &cfg, err, sizeof err) != NULL) return 1;
    printf("[expected load error: %s]\nOK\n", err);
    return 0;
}
