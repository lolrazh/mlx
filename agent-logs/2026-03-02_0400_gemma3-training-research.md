# Gemma3-T1 Training, V2 Cross-Test, and ASR Correction Research

**Date:** 2026-03-02
**Agent:** Claude Opus 4.6
**Status:** ✅ Completed
**Building on:** 2026-03-02_0130_lfm25-training-model-exploration.md

## User Intention
Two-part session. First: complete Gemma 3 4B evaluation (zero-shot baseline, LoRA fine-tune, benchmark) and cross-test all four top models on the v2 test set to find the most robust model. Second: frustrated that all models fail compound operations (the "Wispr" test — assemble letters AND replace a word AND remove the instruction), the user wanted deep research into whether the fine-tuning approach itself is fundamentally wrong, how ASR correction models are actually trained in the literature, whether encoder-decoder (T5) is better for this task, whether RL (DPO/GRPO) would help, and how much data is actually needed. The underlying question: "are we teaching the model to memorize patterns or to understand the task?"

## What We Accomplished

### Gemma 3 4B Training & Benchmarking
- ✅ **Gemma 3 4B zero-shot baseline (B17)** — 9% accuracy, 2.28s latency. Echoes commands back.
- ✅ **Gemma 3 1B zero-shot baseline (B18)** — 0% accuracy, garbled `<end_of_turn>` tokens.
- ✅ **Gemma3-T1 LoRA training** — 1000 iters, all 34 layers, 14.9M trainable (0.327%). Required `grad_checkpoint: true` to avoid OOM (18.9 GB → 11.6 GB). Val loss: 7.477 → 0.056 (iter 500, best ever) → 0.094 (iter 1000).
- ✅ **Gemma3-T1 benchmark** — **87% bf16** at iter 1000 (20 exact, 2 partial, 1 fail), 2.52s latency. Iter 500: 83% (lower val loss but lower accuracy — confirms val loss unreliability).
- ✅ **Iter 500 saved** — Best val loss checkpoint preserved in `spoke/adapters-gemma3-t1-best/`.

### V2 Cross-Test (All 4 Models)
- ✅ **Cross-tested all models on v2 test set (23 examples, includes untrained categories)**
  - Qwen3 T11-ext: 74% (17 exact, 5 partial, 1 fail)
  - Llama3 T1: 74% (17 exact, 6 partial, **0 fails**)
  - Gemma3 T1: 70% (16 exact, 6 partial, 1 fail)
  - LFM2 T1b: 65% (15 exact, 7 partial, 1 fail)
- ✅ **Detailed failure analysis** — Universal failures: #15/#16 (XML), #17 (email), #21 (code-aware) — all untrained categories. Llama uniquely has 0 fails across both v3 and v2 tests.

### ASR Correction Research Deep-Dive
- ✅ **T5/encoder-decoder literature** — T5 is the classic for disfluency correction (seq2seq framing). Encoder-decoder has structural advantage for text editing (bidirectional source encoding + cross-attention). But at 4B+ scale, decoder-only matches/beats it (Gemma 2 9B SOTA on GEC, ACL 2025). MLX blocks encoder-decoder anyway.
- ✅ **Four key techniques identified** — Null-edit training, edit representations (CEGER), constrained decoding, hard negative mining. User correctly pointed out null-edit is irrelevant for Spoke (keyword-triggered pipeline — model never sees clean text).
- ✅ **RL methods surveyed** — EPO (edit-weighted DPO, COLING 2025), SimPO (no reference model, NeurIPS 2024), GRPO, ORPO. SimPO most practical for M4 24GB. `mlx-lm-lora` package supports DPO/GRPO/ORPO on MLX. Consensus: SFT data improvement comes first, RL comes after plateau.
- ✅ **Data requirements** — LIMA: 1K curated matches 52K noisy. IBM: quality filtering > quantity. BEA-2025: two-phase training (corrections first, then no-edit at lower LR). **User's instinct of 1-2K is well-supported by literature.**
- ✅ **LoRA validation** — r=8 is defensible for behavioral adaptation tasks. LoRA is correct tool (not full fine-tune) at 535 examples. The 87-91% ceiling is a data ceiling, not architecture ceiling.

## Technical Implementation

### Gemma 3 4B Architecture Notes
- 4.55B params, 34 layers, hidden_size 2560
- **Multimodal model** — config has nested `text_config` for architecture details
- Model type: `gemma3_text` in mlx-lm
- Requires `grad_checkpoint: true` — OOM at 18.985 GB without it, 11.6 GB with it
- Largest zero-shot-to-fine-tuned gain ever: 9% → 87% (+78 pts)

### Key Research Findings
- **CEGER (2025)**: Structured edit commands (`[DELETE]`, `[INSERT]`, `[REPLACE]`). 2.6% WER on LS test-clean.
- **EPO (COLING 2025)**: DPO variant that upweights edit tokens in loss. SOTA on GEC.
- **SimPO (NeurIPS 2024)**: No reference model needed, outperforms DPO, lower memory.
- **`mlx-lm-lora` package**: Third-party MLX library with DPO/GRPO/ORPO/SimPO support. `pip install mlx-lm-lora`.
- **Conservative filtering (IBM EMNLP 2024)**: Train model to abstain from editing when unsure. Reduces overcorrection.
- **SoftCorrect (AAAI 2023)**: Soft error detector + constrained CTC loss + copy mechanism. 26% CER reduction.

**Files Modified:**
- `spoke/config.yaml` — Changed to Gemma 3 4B, added `grad_checkpoint: true`, 34 layers
- `spoke/bench/run_benchmark.py` — Added `gemma3-4b-bf16` and `gemma3-1b-bf16` model entries
- `spoke/LEDGER.md` — B17/B18 baselines, Gemma3-T1 training/benchmarks, finding #43, model comparison table
- `spoke/RESEARCH_2025_2026.md` — Tool-call edit format + Gemma 3 sections (committed from prior session)

**Files Created:**
- `spoke/adapters-gemma3-t1/` — Training checkpoints (iter 100-1000)
- `spoke/adapters-gemma3-t1-best/` — Iter 500 best checkpoint
- `spoke/bench/result_gemma3-1b-bf16_v2.json` — 1B zero-shot: 0%
- `spoke/bench/result_gemma3-4b-bf16_v2.json` — 4B zero-shot: 9%
- V2 cross-test result files for all 4 models

**Commits:**
- `dd956f1` — Add tool-call edit format and Gemma 3 fine-tuning research
- `78d3046` — Gemma3-T1: 87% bf16 — ties Llama, largest zero-shot gain (+78 pts)
- `765e6d7` — V2 cross-test: Qwen3 and Llama tie at 74%, Llama 0 fails

## Bugs & Issues Encountered
1. **Gemma 3 4B OOM without grad_checkpoint** — Peak memory hit 18.985 GB, crashed at iter 60 with `kIOGPUCommandBufferCallbackErrorOutOfMemory`.
   - **Fix:** Added `grad_checkpoint: true` to config.yaml. Peak dropped to 11.6 GB. ~40% speed penalty.
2. **Gemma 3 config has nested `text_config`** — `num_hidden_layers` not at top level because Gemma 3 is multimodal.
   - **Fix:** Access via `config.text_config.num_hidden_layers` (34 layers).
3. **Benchmark result file overwriting** — Running iter 500 benchmark overwrote iter 1000 result file (same `short_name` → same filename).
   - **Workaround:** Re-ran iter 1000 benchmark after saving iter 500. Systemic issue — result filenames don't include checkpoint info.
4. **Training killed by pipe** — `caffeinate -dims mlx_lm.lora --config spoke/config.yaml 2>&1 | head -25` killed the process when pipe closed.
   - **Fix:** Relaunch without pipe, use `run_in_background: true`.

## Key Learnings
- **Zero-shot is meaningless for fine-tuning potential** — Gemma 3 4B: 9% → 87% (+78 pts). Gemma 3 1B at 0% zero-shot doesn't mean it can't be fine-tuned.
- **All 4B models converge to 87-91%** — Data ceiling, not architecture ceiling. Qwen3 (91%), Llama (87%), Gemma (87%). More data needed, not different models.
- **Llama 3.2 3B is the most robust model** — 0 fails on both v3 AND v2 test sets. Only model to solve self-correction #6. Fastest inference (1.60s). Best all-rounder.
- **The "Wispr problem" reveals the gap** — Models can handle single operations but fail compound operations. This is pattern matching, not understanding. Literature confirms: more diverse training data (especially compound ops) is the fix.
- **Null-edit training is irrelevant for Spoke** — The model is only invoked when a keyword trigger is detected. It never sees clean text. Hard negatives within triggered categories are still valuable though.
- **RL comes AFTER better data** — SFT data improvement is strictly higher leverage than DPO/GRPO at current scale. RL is for when the model CAN do the task but makes inconsistent choices.
- **SimPO > DPO for M4 24GB** — No reference model needed, saves ~4GB. Available via `mlx-lm-lora` package.
- **CEGER edit commands are promising** — Instead of generating full corrected text, generate edit commands. Reduces hallucination surface area. Major format change though.

## Architecture Decisions
- **Llama 3.2 3B emerging as production pick** — 87% accuracy, 0 fails, 1.60s latency, smallest memory footprint of the competitive models. Qwen3 has higher peak (91%) but 1 fail and slower (2.67s).
- **Data expansion to 1-2K is the next priority** — Not model architecture, not RL, not quantization. The ceiling is data quality and diversity, especially for compound operations.
- **CEGER/edit-representation format deferred** — High potential but requires complete data regeneration and new inference pipeline. Explore after hitting 1K+ data ceiling.

## Ready for Next Session
- ✅ **All 4 models trained and cross-tested** — Clear comparison across v3 and v2 test sets
- ✅ **Research synthesis complete** — ASR correction literature, RL methods, data strategies all documented
- 🔧 **Dataset needs expansion to 1-2K** — Focus on compound operations, multi-step edits, more diversity per category
- 🔧 **RESEARCH doc needs updating** — Add CEGER, EPO, SimPO, mlx-lm-lora, data strategy findings
- 🔧 **Llama vs Qwen decision** — User leaning toward Llama 3.2 3B for production. May want to train Llama at 2000 iters to see if it matches Qwen3's 91%.

## Context for Future
This session established that the 87-91% ceiling is a **data problem, not a model problem**. The next phase should focus on expanding the dataset to 1-2K examples with emphasis on compound operations, then potentially exploring CEGER-style edit representations as a format change. RL (SimPO/EPO) is the lever after data expansion plateaus. Llama 3.2 3B is emerging as the production pick over Qwen3 4B due to zero fails and faster inference, but needs a 2000-iter run to confirm ceiling.
