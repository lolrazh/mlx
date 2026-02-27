# ASR Benchmark Expansion — Parakeet, Qwen3-ASR, fp16 Moonshine

**Date:** 2026-02-26
**Agent:** Claude Opus 4.6
**Status:** ✅ Completed

## User Intention
Building on the completed Moonshine MLX port (Phase 2), the user wanted to expand the ASR benchmark suite with additional models and configurations. The goal was to understand the full landscape of open-source ASR on Apple Silicon: how different architectures (transducer vs autoregressive vs LLM-as-decoder), precisions (fp32 vs fp16 vs bf16 vs 8-bit), and model sizes perform on the same eval pipeline. Also explored the Handy dictation app's architecture to understand how production apps handle multi-backend inference.

## What We Accomplished
- ✅ **Moonshine fp16 baseline** — Added `--dtype fp16` flag to mlx_baseline.py, ran 250-sample eval: 2.19% WER (identical to fp32), 0.07x RTF (slower than fp32's 0.04x)
- ✅ **Parakeet TDT V3 baseline** — Installed `parakeet-mlx`, wrote `parakeet_baseline.py`, ran 250-sample eval: **1.93% WER, 0.03x RTF** — best accuracy AND speed of all models
- ✅ **Qwen3-ASR 1.7B baseline** — Installed `mlx-audio`, wrote `qwen3_asr_baseline.py`, ran 250-sample eval: 2.30% WER, 0.20x RTF (8-bit quantized)
- ✅ **Handy app deep dive** — Researched architecture: Tauri (Rust+React), `transcribe-rs` crate wrapping whisper.cpp (GGML, Metal/Vulkan) + ONNX Runtime (Parakeet/Moonshine)
- ✅ **Parakeet optimization analysis** — Profiled inference: encoder is 97% of params and ~70% of time, decoder is tiny LSTM. Concluded 0.03x RTF is near the practical ceiling for single-sample inference.

## Technical Implementation

### Moonshine fp16
Added `dtype` parameter to `load_model()` in `weights.py` — converts all float32 weights after loading. Audio input cast to model dtype in `generate.py` by sniffing `model.proj_out.weight.dtype`.

### Parakeet baseline
`parakeet-mlx` takes file paths via `transcribe()`, but our eval pipeline provides numpy arrays. Solution: use `get_logmel()` directly to convert numpy → mel spectrogram, then call `model.generate(mel)`. Text extracted via `"".join(t.text for t in result.tokens)`.

### Qwen3-ASR baseline
`mlx-audio`'s `model.generate()` accepts numpy arrays directly. Returns `STTOutput` with `.text` attribute. Straightforward wrapper.

**Files Created:**
- `moonshine/baselines/parakeet_baseline.py` — Parakeet TDT V3 eval script
- `moonshine/baselines/qwen3_asr_baseline.py` — Qwen3-ASR 1.7B eval script

**Files Modified:**
- `moonshine/model/weights.py` — Added `dtype` parameter to `load_model()`
- `moonshine/model/generate.py` — Cast audio to model weight dtype before encoding
- `moonshine/baselines/mlx_baseline.py` — Added `--dtype` flag, dynamic model naming

### Final Benchmark (250 shuffled samples, LibriSpeech test-clean, seed=42)

| Model | Params | Precision | WER | RTF | Inference |
|---|---|---|---|---|---|
| Parakeet TDT V3 | 627M | bf16 | **1.93%** | **0.03x** | 66s |
| Moonshine MLX | 265.9M | fp32 | 2.19% | 0.04x | 83s |
| Moonshine PyTorch | 265.9M | fp32 | 2.19% | 0.11x | 215s |
| Moonshine MLX | 265.9M | fp16 | 2.19% | 0.07x | 138s |
| Qwen3-ASR 1.7B | ~2.0B | 8-bit | 2.30% | 0.20x | 401s |
| Whisper Large v3 | ~1.55B | fp16 | 3.02% | 0.34x | 677s |

## Bugs & Issues Encountered
1. **Moonshine fp16 is slower than fp32** — Counterintuitive but consistent: 0.07x vs 0.04x RTF. M4's Metal kernels appear better optimized for fp32, and dtype conversion overhead at boundaries (embeddings, RoPE) may contribute.
   - **Resolution:** Documented as expected behavior. fp16 preserves accuracy perfectly but doesn't help speed on M4.

2. **`mlx-audio` downgraded transformers** — Installing `mlx-audio` replaced `transformers==5.2.0` with `5.0.0rc3` and `mlx-lm==0.30.7` with `0.30.5`.
   - **Workaround:** Accepted the downgrade. Haven't verified if other baselines still work with 5.0.0rc3. May need to pin versions in requirements.txt.

3. **HuggingFace dataset flaky connection** — Multiple `[Errno 9] Bad file descriptor` errors when streaming LibriSpeech. Auto-retries (1s backoff, 5 retries) always succeeded.
   - **Workaround:** Built-in retry logic handled it. No code changes needed.

## Key Learnings
- **MLX bf16 vs fp16 vs fp32:** bf16 is the sweet spot for most models (same exponent range as fp32). fp16 on M4 doesn't guarantee speedup — kernel optimization matters more than dtype width.
- **Parakeet TDT architecture:** The "predict token + skip duration" trick (durations [0,1,2,3,4] frames) cuts decoder steps by ~60%. Decoder is a tiny LSTM (11.8M), not a transformer — the encoder (608.9M FastConformer) does all the heavy lifting.
- **Qwen3-ASR's LLM decoder is the bottleneck:** At 1.7B params, each autoregressive step is a full LLM forward pass. 8-bit quantization saves memory but may hurt WER (2.30% vs reported 1.63% at bf16).
- **Handy's `transcribe-rs` architecture:** Two separate inference paths — whisper.cpp/GGML for Whisper (Metal/Vulkan GPU), ONNX Runtime for Parakeet/Moonshine (CPU). No unified GPU path.
- **`parakeet-mlx`'s `transcribe()` API expects file paths**, not arrays. Use `get_logmel()` + `model.generate(mel)` to bypass file I/O when you have raw audio.
- **Our WER normalization pipeline (lowercase + strip punctuation) correctly handles all models** — Parakeet outputs capitalization+punctuation, Qwen3-ASR outputs capitalization+punctuation, Moonshine/Whisper output lowercase. All normalize to the same format.

## Architecture Decisions
- **Used `parakeet-mlx` and `mlx-audio` as black boxes** — Rather than porting Parakeet/Qwen3-ASR ourselves, we wrapped existing MLX implementations. This is the right call: both are actively maintained, well-tested, and our eval pipeline is model-agnostic. Custom ports only make sense when no MLX implementation exists (like Moonshine was).
- **8-bit Qwen3-ASR over bf16** — 4.7 GB for bf16 is excessive for a benchmark comparison. 8-bit at ~2.5 GB is practical, even if WER is slightly worse than the published number.
- **Skipped Parakeet quantization experiments** — At 0.03x RTF and 1.93% WER, Parakeet is already near-optimal. Quantizing the encoder risks WER regression for marginal speed gain.

## Ready for Next Session
- ✅ **5-model benchmark suite** — All results in `moonshine/results/` as JSON, reproducible with `--max-samples 250`
- ✅ **Parakeet baseline script** — `python -m moonshine.baselines.parakeet_baseline --max-samples 250`
- ✅ **Qwen3-ASR baseline script** — `python -m moonshine.baselines.qwen3_asr_baseline --max-samples 250`
- ✅ **fp16 Moonshine** — `python -m moonshine.baselines.mlx_baseline --dtype fp16 --max-samples 250`
- 🔧 **transformers version pinning** — `mlx-audio` pulled in 5.0.0rc3, may conflict with other deps
- 🔧 **Moonshine optimization (Phase 3)** — KV cache pre-allocation, quantization experiments, memory profiling with `mx.metal.get_active_memory()`
- 🔧 **README update** — Should reflect the full 5-model benchmark table

## Context for Future
This session transformed the project from a single-model port into a comprehensive ASR benchmark suite on Apple Silicon. The eval pipeline's model-agnostic design paid off — adding new models is just a thin wrapper script. The Moonshine MLX port remains the most interesting optimization target because we control the full stack. Parakeet is the performance king (best WER + fastest), but it's a black box we can't optimize further without modifying `parakeet-mlx` internals. Next natural steps: optimize our Moonshine port (Phase 3), update the README with full benchmark results, or explore porting a new architecture from scratch.
