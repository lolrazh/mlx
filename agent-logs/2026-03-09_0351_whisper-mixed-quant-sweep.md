# Whisper Large v3 Mixed-Quant Sweep & Memory Profiling

**Date:** 2026-03-09
**Agent:** Codex GPT-5
**Status:** ✅ Completed

## User Intention
User wanted to return from the recent LLM/post-processing detour back to ASR model optimization, specifically to experiment with mixed quantization on Whisper Large so latency and memory could come down without wrecking transcription quality. The deeper goal was not just "make a smaller checkpoint," but to establish a real evaluation loop for Whisper on MLX: baseline first, then a practical recipe search across uniform and mixed-bit layouts to find the best quality-memory frontier, with a bias toward aggressive low-bit ideas inspired by Unsloth-style quantization wins on LLMs.

## What We Accomplished
- ✅ **Reconstructed the repo direction from prior logs** — Confirmed the broader thread is local-first dictation / ASR latency reduction, not just recent LLM fine-tuning. Anchored on the Moonshine/Whisper branch and the fused ASR brief.
- ✅ **Added an in-memory mixed-quant Whisper benchmark harness** — New `moonshine/baselines/whisper_mixed_quant_baseline.py` loads `whisper-large-v3`, applies per-module MLX quantization predicates, and evaluates through the existing ASR pipeline without patching `mlx_whisper`.
- ✅ **Added JSON recipe support for custom mixed layouts** — Recipes can now be defined outside Python and passed in via `--recipe-file`, which makes future quant sweeps much easier.
- ✅ **Added real MLX memory instrumentation to the eval runner** — `moonshine/eval/run_eval.py` now records active and peak MLX memory in addition to the old Python `tracemalloc` metric.
- ✅ **Added a sequential sweep runner** — New `moonshine/baselines/whisper_mixed_quant_sweep.py` runs multiple recipes in one job, clears MLX cache between recipes, and writes a summary JSON.
- ✅ **Created starter Whisper quant recipes** — Built-in recipes plus JSON recipe files for uniform 3/5-bit and more aggressive 3-bit-with-safety-rails layouts.
- ✅ **Established a comparable 25-sample fp16 baseline** — Ran `whisper-large-v3` on 25 shuffled LibriSpeech-clean samples with the updated memory metrics.
- ✅ **Ran an 8-recipe mixed-quant sweep** — Uniform 3/4/5/6-bit decoder, decoder-only mixed recipes, and an encoder+decoder mixed recipe all benchmarked on the same 25-sample slice.

## Technical Implementation

**New benchmark entrypoint:**
```bash
.venv/bin/python -m moonshine.baselines.whisper_mixed_quant_baseline \
  --recipe decoder-mixed-v1 --max-samples 25
```

**Custom JSON recipe path:**
```bash
.venv/bin/python -m moonshine.baselines.whisper_mixed_quant_baseline \
  --recipe-file moonshine/recipes/whisper/decoder_q3_kinda.json \
  --max-samples 25
```

**Sweep runner:**
```bash
.venv/bin/python -m moonshine.baselines.whisper_mixed_quant_sweep \
  --recipe decoder-uniform-4 \
  --recipe decoder-uniform-6 \
  --recipe decoder-mixed-v1 \
  --recipe all-linear-mixed-v1 \
  --recipe-file moonshine/recipes/whisper/decoder_uniform_3.json \
  --recipe-file moonshine/recipes/whisper/decoder_uniform_5.json \
  --recipe-file moonshine/recipes/whisper/decoder_q3_kinda.json \
  --recipe-file moonshine/recipes/whisper/decoder_q3_very_aggressive.json \
  --max-samples 25 \
  --output-json moonshine/results/whisper-mixed-quant-sweep-25.json
```

**25-sample baseline vs sweep summary:**

| Model / Recipe | WER | RTF | MLX Peak | MLX Active End |
|---|---:|---:|---:|---:|
| fp16 baseline | 2.92% | 0.35x | 3.36 GB | 2.88 GB |
| decoder-uniform-4 | 2.92% | 0.52x | 2.25 GB | 1.76 GB |
| decoder-uniform-6 | 2.92% | 0.60x | 2.44 GB | 1.96 GB |
| decoder-mixed-v1 | 2.92% | 0.49x | 2.33 GB | 1.85 GB |
| all-linear-mixed-v1 | 2.92% | 0.51x | 1.77 GB | 1.15 GB |
| decoder-uniform-3 | 2.70% | 0.51x | 2.15 GB | 1.66 GB |
| decoder-uniform-5 | 2.92% | 0.52x | 2.35 GB | 1.86 GB |
| decoder-q3-kinda | 2.92% | 0.50x | 2.31 GB | 1.82 GB |
| decoder-q3-very-aggressive | 2.92% | 0.50x | 2.23 GB | 1.74 GB |

**Files Modified / Added:**
- `moonshine/eval/run_eval.py` — Added MLX active/peak memory reporting and progress logging
- `moonshine/baselines/whisper_mixed_quant_baseline.py` — In-memory mixed-quant Whisper benchmark harness
- `moonshine/baselines/whisper_mixed_quant_sweep.py` — Sequential recipe sweep runner with cache cleanup
- `moonshine/recipes/whisper/decoder_mixed_v1.json` — JSON version of the main 4/6 decoder mixed recipe
- `moonshine/recipes/whisper/decoder_q3_kinda.json` — 3-bit early decoder + 6-bit safety rails
- `moonshine/recipes/whisper/decoder_q3_very_aggressive.json` — More aggressive 3-bit decoder recipe
- `moonshine/recipes/whisper/decoder_uniform_3.json` — Uniform decoder 3-bit
- `moonshine/recipes/whisper/decoder_uniform_5.json` — Uniform decoder 5-bit
- `moonshine/results/whisper-large-v3_librispeech-clean_25.json` — 25-sample fp16 baseline artifact
- `moonshine/results/whisper-large-v3-*.json` — Per-recipe benchmark artifacts
- `moonshine/results/whisper-mixed-quant-sweep-25.json` — Sweep summary artifact

## Bugs & Issues Encountered
1. **`agent-logs/README.md` was missing** — User requested that the log follow the template there, but no such file existed in the repo.
   - **Fix:** Used the established format from prior logs in `agent-logs/` as the de facto template.

2. **Mixed benchmark initially failed outside the venv** — Running with system `python` produced `ModuleNotFoundError: No module named 'mlx_whisper'`.
   - **Fix:** Switched all validation runs to `.venv/bin/python`.

3. **Baseline eval overwrote the canonical Whisper result file** — The 25-sample baseline run wrote into `moonshine/results/whisper-large-v3_librispeech-clean.json`, which previously held the 250-sample canonical baseline.
   - **Fix:** Restored the original 250-sample file and saved the new run separately as `whisper-large-v3_librispeech-clean_25.json`.

4. **Sweep memory accounting was briefly contaminated by cached models** — `mlx_whisper.transcribe.ModelHolder` kept the prior model alive across sweep items, inflating the next recipe's "after load" memory reading.
   - **Fix:** Explicitly reset `ModelHolder.model` / `model_path`, delete the transcribe fn/model refs, run `gc.collect()`, and clear the MLX cache between recipes.

5. **Old `mx.metal.*` memory calls emitted deprecation warnings** — MLX still supports them, but newer top-level `mx.get_*_memory()` APIs are preferred.
   - **Fix:** `run_eval.py` now prefers the top-level APIs and only falls back to `mx.metal.*` if needed.

## Key Learnings
- **MLX supports per-module mixed quantization natively.** `nn.quantize(..., class_predicate=...)` can return per-module quant params, so mixed-bit Whisper experiments do not require a fork of MLX itself.
- **Stock `mlx_whisper` does not expose mixed layouts directly.** The clean workaround is an in-memory loader/benchmark harness, not trying to force everything through a single global `config.json` quant block.
- **On Whisper Large MLX, quantization reduced memory but did not improve speed.** Every tested quant recipe was slower than fp16 on the 25-sample slice, even when WER stayed flat.
- **The strongest win so far is memory, not latency.** `all-linear-mixed-v1` cut MLX peak from 3.36 GB → 1.77 GB and active end from 2.88 GB → 1.15 GB while matching baseline WER on this slice.
- **Aggressive 3-bit decoder quant did not immediately collapse quality.** `decoder-uniform-3` stayed in the same WER band on 25 samples, which keeps the low-bit search alive, but this is nowhere near enough evidence to claim real parity.
- **Python memory metrics are almost useless for MLX model evaluation.** `tracemalloc` stayed near zero while MLX unified memory moved by multiple GB. Real Metal memory instrumentation is mandatory.
- **25 samples are enough to prune obviously bad recipes, not enough to crown a winner.** The next serious filter should be 100 samples, then possibly 250 for the finalists.

## Architecture Decisions
- **Use a practical recipe grid, not fake exhaustiveness.** Full mixed-bit search across Whisper is combinatorial nonsense. We tested a bounded frontier: uniform low-bit baselines, mixed decoder recipes, and one encoder+decoder mixed layout.
- **Prioritize decoder-first recipes.** Whisper latency is dominated by autoregressive decoding, so the decoder is the right place to spend the early quantization budget.
- **Keep recipe definitions externalizable.** JSON-based layouts make it easy to iterate on bit allocations without patching Python logic each time.
- **Measure speed and memory together.** A smaller model that is slower is still potentially useful, but only if the memory win is meaningful enough to justify the latency tax.

## Ready for Next Session
- ✅ **Mixed-quant Whisper harness exists** — Can benchmark any new recipe via CLI or sweep runner
- ✅ **MLX memory metrics are wired into evals** — Future ASR runs now report real active/peak device memory
- ✅ **First mixed-quant frontier established** — fp16 still wins on speed; mixed quant wins on memory
- 🔧 **Run 100-sample confirmation sweep on top candidates** — Suggested set: fp16 baseline, `all-linear-mixed-v1`, `decoder-uniform-3`, `decoder-mixed-v1`
- 🔧 **Decide whether memory win is worth the latency hit** — If not, quantization alone is the wrong lever for Whisper Large
- 🔧 **If latency remains the target, pivot to architecture/model changes** — e.g. Whisper Large Turbo, decoder truncation, or smaller/faster ASR models rather than deeper quant-only work

## Context for Future
This session established the first real Whisper mixed-quant experimentation loop in the repo. The immediate conclusion is not "quantization solved Whisper," but something subtler and more useful: on MLX Whisper Large, mixed quantization appears to be a **memory optimization first**, not a speed optimization. That is a meaningful result because it narrows the search space. If the product goal is strictly lower dictation latency, future sessions should treat quantization as a secondary constraint solver and focus more on decoder architecture, turbo variants, or alternate ASR models. If the goal is fitting Whisper alongside other models in a tighter memory envelope, the mixed-quant path is already paying off and now has the tooling to be pushed much harder.
