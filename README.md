# RIE — Rust Inference Engine for phones

A from-scratch LLM inference engine in Rust, in the spirit of llama.cpp, built to get the
most out of **phone CPUs and GPUs**. It runs standard **GGUF** models (the files you'd use
with llama.cpp / Ollama / LM Studio) on Android, iOS, Linux, macOS and Windows.

```
rie run  -m qwen2.5-1.5b-instruct-q4_0.gguf --chat -p "Why is the sky blue?"
rie chat -m Llama-3.2-3B-Instruct-Q4_K_M.gguf
rie bench -m gemma-3-1b-it-Q4_0.gguf            # llama-bench style pp/tg numbers
rie bench --synthetic qwen3-1.7b --wtype Q4_0   # no download needed: random weights, real shapes
rie device                                      # SoC clusters, SIMD features, GPUs
```

## What makes it fast on phones

| Problem on phones | What RIE does |
|---|---|
| Matrix multiplies dominate, and phones have int8 dot-product hardware | Weights stay in their ggml block format; activations are quantized to int8 on the fly. Inner loops use **`SDOT`** (Armv8.2 dotprod) and **`SMMLA`** (Armv8.6 i8mm, 2×2 register tiles), detected **at runtime**, so one binary runs the best kernels on every phone, with plain-NEON and scalar fallbacks. |
| Mixed core types (big.LITTLE / DynamIQ) | Reads `/sys` cluster frequencies, `cpu_capacity` and MIDR registers (Cortex-A5xx/A7xx/X-series, Oryon, …), uses **only performance cores** by default, **pins** worker threads to them, and hands out work **dynamically** so faster cores do more of it. Sets QoS *user-interactive* on iOS. |
| Hundreds of small parallel regions per token | Custom fork/join pool: workers spin briefly between jobs (sub-µs hand-off) and then sleep on a condvar, so idle phones don't burn battery. Q/K/V and gate/up matmuls each run in a single parallel region. |
| Prompt processing (prefill) is compute-bound | Batched forward pass: each weight block is unpacked once and reused for 4 tokens (dotprod/AVX2) or 2×2 tiles (i8mm). |
| Limited RAM, apps get killed | Zero-copy **mmap** of the GGUF file (the OS can page weights in and out), optional `mlock`, **f16 or 8-bit (Q8_0) KV cache**, and **ring-buffer KV for sliding-window layers** (Gemma 2/3 keep only `window + batch` positions there). |
| Chat apps re-send the whole conversation | **Prompt-prefix reuse**: only tokens after the longest prefix shared with the previous request are evaluated. |
| GPUs vary widely (Adreno, Mali, Xclipse, PowerVR, Apple) | `rie-gpu` runs the **entire forward pass in portable WGSL** via wgpu (Vulkan on Android, Metal on iOS). No f16 or subgroup extensions needed; weights are uploaded once and only the logits come back to the CPU. |

### SoCs it accounts for

Snapdragon 8 Elite / 8 Elite Gen 5 (Oryon), 8 Gen 3/2/1, 7-series · Dimensity 9400/9300/9200/9000,
8000-series · Tensor G2–G5 · Exynos 2400/2500 · Apple A15–A19 · older Cortex-A53/A55/A7x devices.
Cores are classified from frequency, `cpu_capacity` and micro-architecture. For example, on
Snapdragon 8 Gen 3 it picks X4 + 5×A720 (6 threads) and skips the A520s; on Tensor G4 it picks
X4 + 3×A720; on 8 Elite all 8 Oryon cores. These cases are covered by unit tests.

## Supported models

| Architecture (`general.architecture`) | Models |
|---|---|
| `llama`, `mistral` | Llama 2, **Llama 3.1/3.2**, Mistral 7B, SmolLM/SmolLM2, TinyLlama, DeepSeek-R1-Distill-Llama |
| `qwen2` | **Qwen2.5** 0.5B–7B, Qwen2.5-Coder, DeepSeek-R1-Distill-Qwen |
| `qwen3` | **Qwen3** 0.6B–8B dense |
| `phi3` | Phi-3/3.5-mini, **Phi-4-mini** (fused QKV, partial rotary) |
| `gemma`, `gemma2`, `gemma3` | Gemma 2B/7B, Gemma 2 2B/9B (soft-capping, SWA), **Gemma 3 1B/4B** text (SWA 5:1, dual RoPE base) |
| `granite` | IBM Granite 3.x dense |
| `qwen3moe` | **Qwen3-30B-A3B** and other Qwen3 mixture-of-experts models (CPU backend) |
| `qwen2moe` | Qwen1.5-MoE / Qwen2-MoE (shared expert, CPU backend) |
| `llama` + `*_exps` tensors | Mixtral-style MoE (CPU backend) |

**Quantizations:** F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
(so Q4_K_M, Q5_K_M, Q6_K and Q8_0 files all work). Fast int8 SIMD paths exist for Q4_0, Q4_1,
Q5_0, Q5_1, Q8_0, Q4_K, Q5_K and Q6_K; Q2_K/Q3_K use a generic path.

**Tokenizers:** SentencePiece-BPE (`llama`) and byte-level BPE (`gpt2`) with the Llama-3,
Qwen2, GPT-2, SmolLM and GPT-4o pre-tokenizers. **Chat templates:** Llama 3, ChatML (Qwen,
SmolLM), Gemma, Phi-3, Mistral, Granite, all auto-detected.

### Picking a model for a phone

| Phone RAM | Good choices (instruct, GGUF) |
|---|---|
| 6–8 GB | Qwen3-0.6B/1.7B, Llama-3.2-1B, Gemma-3-1B, SmolLM2-1.7B |
| 12 GB | Llama-3.2-3B, Qwen3-4B, Phi-4-mini, Gemma-3-4B (text) |
| 16–24 GB | Qwen3-8B, Llama-3.1-8B (Q4); Qwen3-30B-A3B (Q3_K/Q4_K, ~3B active parameters per token, so it generates at roughly 3B-model speed) |

On ARM CPUs, **Q4_0** is usually the fastest format (cheapest to unpack, plus i8mm tiles).
**Q4_K_M** gives better quality at about the same size.

## Tool calling, agents and on-device learning

`rie-server` is an OpenAI-compatible server and a drop-in for `llama-server`, built for tool
harnesses such as the stitaP tool harness:

- **Grammar-constrained tool calls:** small models can't emit broken JSON, unknown tools or
  wrong argument names.
- **Online reinforcement learning:** a mixture of low-rank *correction experts* with a
  learned router sits on the output layer. It learns from `/v1/feedback`,
  `/v1/corrections` and, with `--auto-reward`, from tool errors the harness sends back.
- **Phones:** included in the static ARM64 build, so it runs in Termux next to the agent.

See [docs/tool-harness.md](docs/tool-harness.md).

## Repository layout

```
crates/rie-core   engine: GGUF, quant formats, SIMD kernels, thread pool, SoC topology,
                  transformer, tokenizers, sampler, chat templates, Engine API
crates/rie-gpu    wgpu backend: WGSL kernels for the full forward pass
crates/rie-cli    `rie` command-line tool
crates/rie-ffi    C ABI (include/rie.h), Android JNI → librie.so / librie.a
crates/rie-server OpenAI/llama-server-compatible server: constrained tool calls, online RL
bindings/android  Kotlin wrapper (ai.rie.RieEngine) + example Activity
bindings/ios      Swift package (RieEngine) over the C API
scripts/          build-android.sh, build-ios.sh
```

## Building

```bash
cargo build --release                      # CPU-only CLI: target/release/rie
cargo build --release --features gpu -p rie-cli   # + Vulkan/Metal backend (--gpu flag)
```

### Android

```bash
cargo install cargo-ndk
rustup target add aarch64-linux-android x86_64-linux-android
scripts/build-android.sh            # or --gpu; copies librie.so into bindings/android/rie
# Benchmark directly on a phone:
adb push target/aarch64-linux-android/release/rie model.gguf /data/local/tmp/
adb shell /data/local/tmp/rie bench -m /data/local/tmp/model.gguf
adb shell /data/local/tmp/rie device
```

In an app:

```kotlin
val engine = RieEngine.load(modelFile.path)          // threads = performance cores, pinned
engine.chat(listOf(RieEngine.ChatMessage("user", "Hi!"))) { piece -> show(piece); true }
```

### Benchmark your phone (no NDK, no app)

Download the `rie-android-arm64-static` artifact from the latest CI run, or build it
yourself with `scripts/build-android-static.sh`. Then, with the phone connected over USB
debugging:

```bash
scripts/phone-bench.sh                      # synthetic Qwen2.5-0.5B / Llama-3.2-1B / Qwen3-1.7B shapes
scripts/phone-bench.sh my-model.gguf        # plus your own GGUF files
PRESETS="llama-3.2-3b qwen3-4b" WTYPES=Q4_0 THREADS="auto 4 6 8" scripts/phone-bench.sh
```

It prints a markdown report: SoC and core clusters, prompt and generation tok/s across
thread counts, the i8mm vs. dotprod vs. NEON kernel tiers, and the 8-bit KV cache. Set
`ADB=local` to run it inside Termux on the phone itself.

### iOS

```bash
scripts/build-ios.sh                # on macOS; produces bindings/ios/build/RIE.xcframework
```

Then add `bindings/ios` as a Swift package:

```swift
let engine = try RieEngine(modelPath: path)
try engine.chat([.init(role: .user, content: "Hi!")]) { piece in print(piece, terminator: ""); return true }
```

### C / C++

See [`crates/rie-ffi/include/rie.h`](crates/rie-ffi/include/rie.h) and the smoke test
[`crates/rie-ffi/tests/c_api.c`](crates/rie-ffi/tests/c_api.c).

### Rust

```rust
use rie_core::{Engine, EngineConfig, GenerateParams, chat::{Message, Role}};
let mut engine = Engine::load("model.gguf", EngineConfig::default())?;
engine.chat(&[Message::new(Role::User, "Hello")], &GenerateParams::default(), |t| { print!("{t}"); true })?;
```

## Tuning knobs

| Option | Default | Notes |
|---|---|---|
| `--threads` / `n_threads` | number of performance cores (max 8) | Token generation is limited by memory bandwidth; 4–6 threads is often the sweet spot. Try `rie bench -t N`. |
| `--ctx` | 4096 | KV memory = 2 × layers × kv_heads × head_dim × ctx × 2 bytes |
| `--batch` | 64 | Prompt chunk size; larger is faster prefill but uses more scratch memory |
| `--kv q8_0` | f16 | 8-bit KV cache: ~47% less KV memory (logits within ~2% of f16) |
| `--no-pin` | pinning on | Disable if the OS restricts affinity |
| `RIE_DISABLE=i8mm,dotprod,neon,avx2,simd` | — | Force slower kernels (A/B testing, debugging) |

## Correctness testing

* **Kernels:** every quant type × SIMD path is checked against scalar dequantize-then-dot
  references (`tests/kernels.rs`).
* **Whole model:** the optimized forward pass is compared logit-by-logit with an
  independent, naive f32 transformer written straight from the model math
  (`tests/model.rs`). This covers all 8 architectures, batched prefill vs. token-by-token
  decode, sliding-window ring buffers, tied embeddings, and 1–4 threads. The F16/F32 models
  agree to about 0.05%.
* **ARM:** the same tests run under QEMU in CI for three CPU tiers: i8mm+dotprod, dotprod
  only, and a baseline Cortex-A53.
* **GPU:** `rie-gpu` is compared with the CPU backend on Mesa lavapipe (software Vulkan)
  in CI. F16 models agree to about 1e-6.

```bash
cargo test --workspace
cargo test -p rie-core --target aarch64-unknown-linux-gnu   # needs qemu-user + aarch64 gcc
```

## Status and limitations

* All correctness checks so far ran under emulation (QEMU for ARM, lavapipe for the GPU).
  **Performance has not yet been measured on physical phones.** `rie bench` and
  `rie bench --synthetic <preset>` exist to do that.
* The GPU matmul kernels are portable but not yet tuned per vendor (no subgroup or
  cooperative-matrix paths). On many phones the CPU path will be as fast as the GPU or
  faster for token generation, which is limited by memory bandwidth. The GPU mainly helps
  prefill and frees up the CPU. On the GPU, formats other than F16/Q8_0/Q4_0/Q4_K/Q6_K are
  converted to Q8_0 at load time.
* MoE models run on the CPU backend only (the GPU backend reports them as unsupported, and
  the C/Kotlin/Swift API falls back to the CPU). Each token is routed to its top-k experts,
  and during prompt processing tokens are grouped by expert.
* Not supported yet: gpt-oss and DeepSeek-style MoE, vision/audio towers, YaRN
  beyond the trained context, IQ*/TQ* quantizations, speculative
  decoding, SME2 kernels (Armv9.2: Dimensity 9500, 8 Elite Gen 5, Apple M4/A19).
* When the context fills up during generation, the oldest half of the conversation (after
  `n_keep` tokens) is dropped and the rest is re-evaluated, so generation continues
  (`GenerateParams::context_shift`, on by default). The 8-bit KV cache works on both the CPU
  and GPU backends.

## License

MIT OR Apache-2.0
