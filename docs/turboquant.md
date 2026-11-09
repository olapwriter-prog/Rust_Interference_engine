# TurboQuant, layer streaming, quantization and MoEfication

## TurboQuant in RIE

[TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025) quantizes vectors
online, with no calibration:

1. Multiply each vector by a fixed random orthogonal matrix. RIE uses random sign flips
   followed by a normalized Walsh–Hadamard transform over power-of-two blocks (O(d log d),
   no matrix stored).
2. After the rotation every coordinate is close to Gaussian, whatever the input looked
   like: outlier channels (common in RoPE'd keys) are spread over the whole block.
3. So one scalar quantizer, the Lloyd–Max codebook for N(0,1), is near-optimal for every
   coordinate, with a single scale per rotation block.

It is implemented natively in `rie-core/src/turbo.rs`. The [`turbovec`](https://crates.io/crates/turbovec)
crate implements the same idea for vector search, but it needs Rust 1.89 and pulls in
rayon/statrs/rand. It also is a search index, not an attention kernel.

### 4-bit KV cache (`--kv turbo4`)

| | f16 | Q8_0 | TurboQuant-4 |
|---|---|---|---|
| bytes per 128-value head vector | 256 | 136 | **66** |
| bits per value | 16 | 8.5 | **4.125** |

The attention math is unchanged because the rotation is orthogonal:

* **scores:** `q·k = (Rq)·(Rk)`. The query is rotated once per head and int8-quantized,
  and keys are scored straight from their 4-bit codes. The codes become int8 codebook
  values with one byte shuffle (`tbl` on NEON, `pshufb` on AVX2) followed by an int8 dot
  product.
* **values:** `Σ pᵢ vᵢ = Rᵀ Σ pᵢ (R vᵢ)`. The weighted sum is accumulated in the rotated
  domain, then rotated back once per head.

The cache is never decompressed. Checks (`cargo test`):

* Rotated-domain attention equals ordinary attention over the reconstructed keys and values.
* The codebook satisfies the Lloyd–Max conditions.
* The SIMD and scalar paths agree on x86 and on ARM (QEMU).

Measured on a 4-core x86 cloud VM (AVX2), synthetic Qwen3-0.6B-shaped model, Q4_0 weights,
3,000-token prompt:

| KV cache | prompt tok/s | generation tok/s (at ~3k context) | KV memory |
|---|---|---|---|
| f16 | 43.8 | 12.8 | 100% |
| TurboQuant-4 | 43.3 | **14.2** | **26%** |

Generation is faster at long contexts because attention reads 3.9× fewer KV bytes per
token. Expect the gain to grow with context length and on phones, where memory bandwidth
is scarcer. These numbers are not yet measured on phones.

Accuracy: 4-bit KV adds noise. On random-weight test models, logits differ from the f16
cache by up to ~25% (max-relative), against ~2% for Q8_0. Random weights are the worst case,
and the published TurboQuant results report negligible quality loss at 3.5–4 bits on real
models. Still, validate on your model before relying on it, for example with
`rie perplexity --kv turbo4`.

### Compressed memory index

`TurboIndex` stores vectors as 2- or 4-bit codes plus one scale. It scores queries with a
per-query byte lookup table and never decompresses the stored vectors. The on-device
memory (see [tool-harness.md](tool-harness.md#memory-and-tool-knowledge)) uses it at 256
dimensions × 4 bits, which is 132 bytes per observation.

## Layer streaming (`--stream`)

Weights are memory-mapped. With `--stream`, while layer *l* runs, layer *l+1* is prefetched
from flash (`MADV_WILLNEED`), and layer *l−1*'s pages are released (`MADV_DONTNEED`). The
pages are re-read transparently next time. Peak weight RSS becomes about two layers plus the
embeddings and the output head, so a model larger than free RAM keeps running instead of
being killed by the low-memory killer. Speed is then bound by flash read bandwidth (UFS 4.0:
~4 GB/s). Without `--stream`, nothing changes.

## 32/16-bit → 4-bit: `rie quantize`

```bash
rie quantize model-f16.gguf model-q4_0.gguf --type Q4_0     # or Q8_0, F16
```

Every weight matrix is re-quantized. The output head (or the tied embedding) is kept at
Q8_0, and norms and biases stay as they are. Example: a Qwen2.5-0.5B-shaped model goes from
992 MB to 350 MB. For the best 4-bit quality (k-quants, importance matrices), llama.cpp's
`llama-quantize` output loads directly.

## Dense → mixture of experts: `rie moefy`

```bash
rie moefy model-q4_0.gguf model-moe.gguf --experts 8 --used 2 --calib sample.txt
```

1. The dense model is run over the calibration text, and every feed-forward input is
   recorded.
2. In each layer, neurons that fire together are clustered into equal-size experts
   (balanced k-means on activation patterns).
3. A router is fitted by ridge regression to predict each expert's activation mass from the
   input. This is a closed-form solve: no back-propagation and no GPU.
4. At run time only the top `--used` experts are evaluated, and they are summed with weight
   1 (GGUF key `rie.expert_gating = "unit"`).

Facts to keep in mind:

* **Exact at `--used = --experts`**: tested.
* **Saves compute and weight reads, not storage.** All experts remain in the file. On the
  synthetic Qwen2.5-0.5B shape, 8 experts with 2 active raised generation from 35.9 to
  58.5 tok/s and prompt speed from 152 to 328 tok/s (x86, 4 threads).
* **Quality depends on activation sparsity.** The report prints, per layer, how much
  activation mass the router keeps on the calibration set. On random weights it is close to
  `used/experts` (no structure to find). Real models are much sparser, especially
  ReLU-family ones. Measure with `rie perplexity` before and after. If the loss is too high,
  use more active experts or fine-tune the converted model briefly (MoEfication papers
  recover most of the quality that way). Training is outside this engine.
* MoEfied files need RIE: llama.cpp would apply softmax gating to them.
