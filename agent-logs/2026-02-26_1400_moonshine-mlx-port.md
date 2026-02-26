# Moonshine ASR — Full MLX Port (Phase 2)

**Date:** 2026-02-26
**Agent:** Claude Opus 4.6
**Status:** ✅ Completed

## User Intention
Port the Moonshine ASR model (265.9M params, encoder-decoder speech model) from PyTorch/HuggingFace to native MLX on Apple Silicon. The goal was to match the PyTorch baseline's 2.19% WER exactly while running natively without PyTorch dependencies at inference time. This is Phase 2 of a larger project that already had a working eval pipeline and two baselines (Whisper Large v3 in MLX, Moonshine Medium in PyTorch).

## What We Accomplished
- ✅ **Config parsing** — `ModelArgs` / `EncoderArgs` dataclasses that parse HF `config.json` into clean typed fields
- ✅ **Audio embedder** — CMVN + asinh compression + Linear + 2x CausalConv1d (4x downsampling), verified matching PyTorch within atol=1e-4
- ✅ **14-layer encoder** — Custom `MoonshineLayerNorm` (unit_offset), sliding window attention with per-layer window sizes, GELU FFN. Encoder output matches PyTorch within max diff 0.00002
- ✅ **14-layer decoder** — Partial RoPE (32/64 dims), cross-attention with KV caching, gated SiLU FFN. Teacher-forcing logits match within max diff 0.00003
- ✅ **Weight loading** — `sanitize()` function handles prefix stripping, Conv1d axis swapping. 362 weight tensors, 0 missing / 0 extra
- ✅ **Greedy decoding** — Autoregressive generation with KV cache, produces perfect transcriptions
- ✅ **Eval baseline script** — `mlx_baseline.py` plugs into existing eval pipeline
- ✅ **WER validation** — **2.19% WER on 250 shuffled LibriSpeech test-clean samples — exact match with PyTorch baseline**
- ✅ **Performance** — 0.04x RTF (2.75x faster than PyTorch+MPS at 0.11x)

## Technical Implementation

**Architecture (faithful port of HF `moonshine_streaming`):**
```
Audio waveform (16kHz)
  → Embedder (CMVN → asinh → Linear(80→768) → SiLU → CausalConv1d×2)
  → 14 Encoder layers (sliding window attn, MoonshineLayerNorm, GELU FFN)
  → Add positional embeddings (Embedding(4096, 768)) + Linear(768→640)
  → 14 Decoder layers (causal self-attn w/ RoPE, cross-attn, gated SiLU FFN)
  → proj_out(640→32768) → greedy argmax → tokenizer decode
```

**Key design choices:**
- fp32 throughout (no fp16 conversion yet) — eliminates precision as a debugging variable
- Sliding window attention always active — required for numerical correctness
- Simple concatenation-based KV cache (no pre-allocation) — correct before fast
- MLX channels-last convention for Conv1d (vs PyTorch channels-first)

**Files Created:**
- `moonshine/model/__init__.py` — Public API exports
- `moonshine/model/config.py` — `ModelArgs`, `EncoderArgs` dataclasses with `from_dict()` classmethod
- `moonshine/model/embedder.py` — `Embedder`, `CausalConv1d`, `AsinhCompression`
- `moonshine/model/encoder.py` — `Encoder`, `EncoderLayer`, `EncoderAttention`, `EncoderFFN`, `MoonshineLayerNorm`, `make_sliding_window_mask()`
- `moonshine/model/decoder.py` — `Decoder`, `DecoderLayer`, `DecoderSelfAttention`, `DecoderCrossAttention`, `GatedFFN`, `KVCache`
- `moonshine/model/moonshine.py` — Top-level `Moonshine` model (encoder + decoder + proj_out)
- `moonshine/model/weights.py` — `sanitize()`, `load_model()`
- `moonshine/model/generate.py` — `generate()`, `pad_audio()`
- `moonshine/baselines/mlx_baseline.py` — Eval script matching existing baseline pattern

## Bugs & Issues Encountered

1. **RoPE `traditional` flag was inverted** — Used `traditional=False` (split-half) instead of `traditional=True` (interleaved pairs). Caused repetition loops in generation after ~20 tokens.
   - **Symptoms:** Model produced "the dominion of his the dominion of his..." in an infinite loop
   - **Root cause:** MLX's naming is counterintuitive — `traditional=True` = interleaved pairs (0,1),(2,3)... which is what the original RoPE paper and HF's `rotate_half` function use. `traditional=False` = split-half.
   - **Fix:** Changed `traditional=False` → `traditional=True` in `DecoderSelfAttention.__init__`
   - **Diagnosis method:** Systematic comparison — position 0 logits matched perfectly but error grew with position, pointing to RoPE. Direct numerical comparison of `nn.RoPE` output confirmed the mismatch.

2. **Initial encoder comparison showed max diff 5.73** — Looked like a catastrophic encoder bug.
   - **Root cause:** My test called `pt_model.model.encoder(input_values)` without passing `attention_mask`, so PyTorch ran with full attention while MLX used sliding windows.
   - **Fix:** Pass `attention_mask=pt_inputs.attention_mask` to PyTorch encoder. With matching masks, max diff dropped to 0.00002.

3. **Initial weight transpose was wrong** — Plan said "transpose all 2D linear weights."
   - **Root cause:** Both MLX and PyTorch store Linear weights as `(out_features, in_features)` — same convention. No transpose needed. Only Conv1d axes need swapping `(out, in, kernel)` → `(out, kernel, in)`.
   - **Fix:** Removed all 2D transpose logic from `sanitize()`, kept only Conv1d `swapaxes(1, 2)`.

4. **Teacher-forcing comparison had in-place mutation bug** — PyTorch decoder modifies `encoder_hidden_states += positional_embeddings` in-place, corrupting the reference tensor before I copied it to MLX.
   - **Fix:** Clone encoder output with `.clone().numpy()` before any decoder call.

## Key Learnings

- **MLX RoPE naming:** `traditional=True` = interleaved pairs (what most HF models use). `traditional=False` = split-half. This is the opposite of what you'd guess from the name.
- **MLX Linear convention:** Same as PyTorch — weight stored as `(out, in)`, forward computes `x @ W.T`. No transpose needed when porting from PyTorch.
- **MLX Conv1d convention:** Channels-last `(out, kernel, in)` vs PyTorch channels-first `(out, in, kernel)`. Need `swapaxes(1, 2)`.
- **Sliding window attention is not optional:** The model was trained with per-layer sliding windows. Removing them changes encoder output by up to 5.73 (max abs diff). Even though the model "runs" without them, WER would be significantly worse.
- **HF Moonshine's `attention_mask` flow:** The processor DOES create an `attention_mask` (all-ones for unpadded audio). This mask propagates through the embedder's conv layers and triggers the sliding window mask creation in the encoder. Without it, NO sliding window masks are used.
- **In-place mutation in HF decoder:** `encoder_hidden_states += pos_emb` modifies the tensor. During generation this is harmless (cross-attn K/V is cached after first step), but it's a trap for comparison code.
- **Systematic layer-by-layer comparison is essential:** The RoPE bug would have been near-impossible to find from output alone. Comparing embedder → encoder layer 0 → ... → decoder logits per position immediately isolated the growing-error pattern to RoPE.

## Architecture Decisions

- **fp32 first, fp16 later** — Enabled exact numerical comparison at every layer. All bugs were found through numerical comparison, which would have been ambiguous in fp16.
- **No padding mask support** — Our MLX encoder always creates sliding window masks (no padding mask). For single-sample inference this is correct. Batch inference with different-length audio would need padding mask support.
- **Simple KV cache (concatenation)** — Not optimal (O(n) per step due to concat), but correct and simple. Pre-allocated buffers would be faster but add complexity.
- **Reimplement audio preprocessing** — 4 lines of code (`pad_audio`) replaces HF's `Wav2Vec2FeatureExtractor`. The feature extractor just pads for this model (`do_normalize=False`).
- **Tokenizer from HF** — Still uses `PreTrainedTokenizerFast` for decoding. Pure MLX tokenizer would eliminate the transformers dependency entirely.

## Ready for Next Session

- ✅ **Working MLX Moonshine at fp32** — 2.19% WER, 0.04x RTF, all tests passing
- ✅ **Eval pipeline** — `python -m moonshine.baselines.mlx_baseline --max-samples 250` reproduces results
- 🔧 **fp16 conversion** — Switch to `mx.float16` for ~2x memory reduction and potential speed gain
- 🔧 **KV cache optimization** — Pre-allocated buffers instead of concatenation
- 🔧 **Memory profiling** — `tracemalloc` only tracks Python allocations, not MLX unified memory. Need `mx.metal.get_active_memory()` for real GPU memory tracking
- 🔧 **Batch inference** — Would need padding mask support in encoder
- 🔧 **Remove transformers dependency** — Implement tokenizer loading from `tokenizer.json` directly

## Context for Future
This completes Phase 2 of the Moonshine project. The model architecture is fully ported and numerically validated. Phase 3 would focus on optimization: fp16/quantization experiments, KV cache improvements, memory profiling, and throughput tuning. The 0.04x RTF already beats PyTorch+MPS by 2.75x, so the main optimization targets would be memory footprint (fp16/4-bit) and batch processing support.
