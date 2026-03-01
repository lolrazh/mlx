# Spoke Experiment Ledger

> Single source of truth for every training run, benchmark, and planned experiment.
> Last updated: 2026-03-01 (consolidated — stale docs removed, agent-log insights rolled in)

## How to Read This

- **Clean accuracy** = generic or v2 prompt only (no few-shot, no data leakage). The only honest metric.
- **Spoke accuracy** is crossed out where shown — 4/23 test examples leaked into the few-shot prompt. See [Data Leakage](#data-leakage) below.
- **Data v1** = 472 train, generic v1 system prompt (~30 tokens), includes multi-command category.
- **Data v2** = 447 train / 20 valid / 23 test, v2 system prompt (~80 tokens), multi-command removed, targeted self-correction + quote-endquote fixes, XML tag fixes.
- **Accuracy** = (exact + semantic) / N. Test set: 12 examples (v1) or 23 examples (v2).
- **6-bit is the deploy quant.** bf16/8-bit = same accuracy. 4-bit drops ~17% (2 examples). See [Quant Impact](#quantization-impact).
- Benchmark results as JSON: `spoke/bench/result_*.json`

---

## Baselines (Zero-Shot, No Training)

### v1 test set (12 examples)

| ID | Model | Quant | Prompt | N | Accuracy | Latency | Notes |
|----|-------|-------|--------|---|----------|---------|-------|
| B1 | Qwen3-4B | 4-bit | generic v1 | 12 | **25%** | 0.69s | Real floor. Model barely understands the task. |
| B2 | Qwen3-4B | 4-bit | v2 | 12 | **50%** | 0.91s | Prompt alone doubles accuracy. Best clean baseline on v1. |
| B3 | Qwen3-4B | 4-bit | task-specific | 12 | **50%** | 0.89s | Per-category prompts = same as v2. |
| B4 | Qwen3-4B | bf16 | v2 | 11 | **36%** | 1.64s | bf16 more verbose at zero-shot. Interim test set (11 ex). |
| B5 | Llama 1B | bf16 | generic v1 | 12 | **8%** | 2.00s | Hallucinations, refusals, code generation. |
| B6 | Llama 1B | bf16 | task-specific | 12 | **25%** | 1.12s | Marginal improvement over generic. |
| ~~B7~~ | ~~Qwen3-4B~~ | ~~4-bit~~ | ~~spoke~~ | ~~12~~ | ~~58%~~ → 38% | ~~1.43s~~ | ~~LEAKED. 4/23 test examples in few-shot prompt.~~ |
| ~~B8~~ | ~~Llama 1B~~ | ~~bf16~~ | ~~spoke~~ | ~~12~~ | ~~25%~~ | ~~0.76s~~ | ~~LEAKED. Same score either way.~~ |

### v2 test set (23 examples)

| ID | Model | Quant | Prompt | N | Accuracy | Latency | Notes |
|----|-------|-------|--------|---|----------|---------|-------|
| B9 | Qwen3-4B | 4-bit | generic v1 | 23 | **13%** | 0.68s | Real floor on v2 test set. Only caps + emoji pass. |
| B10 | Qwen3-4B | 4-bit | v2 | 23 | **35%** | 0.85s | v2 baseline. 7 exact + 1 semantic. |

**Takeaway:** V2 test set is significantly harder — generic floor drops from 25% to 13%, v2 prompt from 50% to 35%. T4's 74% bf16 = **+61 points** over floor, **+39 points** over v2 baseline.

### v3 test set (23 examples, trigger-matched categories only)

v3 removes 4 categories with no production Spoke triggers (formatting-xml, email, code-aware, hard-negative). Keeps 9 categories: spell-replace, self-correction, quote-unquote, quote-endquote, at-symbol, caps, emphasis, emoji, camelcase. Test set: `spoke/bench/test_set_v3.json`.

| ID | Model | Quant | Prompt | N | Accuracy | Exact | Sem | Part | Fail | Latency | Notes |
|----|-------|-------|--------|---|----------|-------|-----|------|------|---------|-------|
| B11 | Qwen3-4B | bf16 | v2 | 23 | **22%** | 4 | 1 | 11 | 7 | 1.66s | v3 baseline. Lower than v2 (35%) — new test examples are harder for base model. |
| B12 | LFM2.5-1.2B | 4-bit | v2 | 23 | **9%** | 1 | 1 | 7 | 14 | 0.24s | Hybrid conv+attn. Can't parse meta-linguistic commands at all. |
| B13 | LFM2-2.6B-Exp | bf16 | v2 | 23 | **9%** | 2 | 0 | 14 | 7 | 0.87s | 2x params, bf16. Same failure as 1.2B — architecture bottleneck, not precision. |
| B14 | LFM2-2.6B-Exp | bf16 | spoke (few-shot) | 23 | **30%** | 6 | 1 | 11 | 5 | 1.36s | Huge jump with examples — learns output format but not command execution. **LEAKED (4/23).** |

**Takeaway:** v3 test set zero-shot baseline is 22% (Qwen3). LFM2 scores 9% regardless of size/precision — conv-dominant hybrid architecture can't handle meta-linguistic commands zero-shot. Few-shot helps format (30%) but not reasoning.

---

## Training Runs

All runs use Qwen3-4B-Instruct-2507-bf16 unless noted. All use `mask_prompt: true`, `batch_size: 4`, `lr: 1e-5`, `seed: 42`.

| Run | Date | Type | Rank | Optimizer | LR Schedule | Data | Iters | Best Val Loss | Notes |
|-----|------|------|------|-----------|-------------|------|-------|---------------|-------|
| **T1** | 02-28 | LoRA | r=16 | adam | flat | v1 (472) | 1000 | 0.16 @200 | First run. Overfits ~iter 500. |
| **T2** | 03-01 | LoRA | r=8 | adam | flat | v1 (472) | 1000 | 0.146 @500 | r=8 = r=16. Half params, same accuracy, later overfit. |
| **T3** | 03-01 | LoRA | r=16 | adam | flat | v1 (472) | ~750 | 0.243 @200 | **Llama 1B base.** Overfits @200. Killed. Not viable. |
| **T4** | 03-01 | LoRA | r=8 | adam | flat | **v2 (447)** | 800 (OOM@450) | 0.174 @300 | Overfits ~350. OOM at 450 (Metal memory). Best ckpt = 300. |
| **T5** | 03-01 | **DoRA** | r=8 | adam | flat | v2 (447) | 200 (OOM@save200) | 0.229 @200 (unsaved) / 0.272 @100 | DoRA +1GB peak mem (15.2 GB). OOM during iter 200 save. Only iter 100 ckpt. |
| **T6** | 03-01 | LoRA | r=8 | **adamw** (wd=0.01) | flat | v2 (447) | 200 | 0.231 @200 | grad_checkpoint=true. Peak mem 9.8 GB (was 14 GB). **Zero quant loss.** |
| **T6b** | 03-01 | LoRA | r=8 | **adamw** (wd=0.01) | flat | v2 (447) | 500 | 0.200 @500 | Clean AdamW rerun. No grad_ckpt, caffeinate -dims. Still dropping at 500 — no overfit. Peak 14.1 GB. |
| **T6c** | 03-01 | LoRA | r=8 | **adamw** (wd=0.01) | flat | v2 (447) | 800 (500+300) | 0.190 @600 | Extended from T6b. Plateaus ~0.20 after 600. **Matches T4 bf16 (74%) but worse 6-bit (61% vs 65%).** |
| T7 | — | LoRA | r=8 | **adamw** (wd=0.01) | **cosine** (warmup=50) | v2 (447) | 200 | — | Isolate LR schedule. |
| **T8** | 03-01 | **DoRA** | r=8 | **adamw** (wd=0.01) | flat | v2 (447) | 200 (50 eff.) | 0.427 @200 | batch=1+accum=4 (OOM forced). 50 effective steps. 15.4 GB peak. DoRA not viable on M4. |
| T9 | — | LoRA (QLoRA) | r=8 | adam | flat | v2 (447) | 200 | — | **4-bit base model.** Memory test (~4-5GB target). |
| T10 | — | LoRA | r=8 | adam | flat | v2 (447) | 200 | — | **mask_prompt: false.** More gradient signal? |
| **T11** | 03-01 | LoRA | r=8 | adam | flat | **v3 (492)** | 300 | 0.169 @250 | **v3 data.** Trigger-matched categories only. T4 config on new data. **83% bf16, 74% 6-bit.** |
| **T12** | 03-01 | LoRA | r=8 | adam | flat | **v3+patch (535)** | 300 | 0.162 @300 | **v3 + 43 targeted examples.** Best val loss but **REGRESSED to 74% bf16** (from 83%). Patch fixed 0/4 targets, caused 2 new regressions. |

### Alternative Models

| Run | Date | Base Model | Type | Rank | Optimizer | Data | Iters | Best Val Loss | Notes |
|-----|------|-----------|------|------|-----------|------|-------|---------------|-------|
| **LFM2-T1** | 03-01 | LFM2-2.6B-Exp (bf16) | LoRA | r=8 | adam | v3 (535) | 800 | 0.480 @800 | All 30 layers (8 attn + 22 conv). 12.2M trainable (0.476%). Peak 13.3 GB. **78% bf16 at 800 iters.** Double-descent: plateau 250-400, then steep drop 500-800. |
| **LFM2-T1b** | 03-01 | LFM2-2.6B-Exp (bf16) | LoRA | r=8 | adam | v3 (535) | 2000 | 🔄 training | Extended from T1. Val loss still dropping (0.480), no overfit. Resumed from iter 800 checkpoint. |

---

## Fine-Tuned Benchmarks (Clean Results Only)

All from Qwen3-4B base.

> **Test set change at T4:** T1-T3 used v1 test set (12 examples, basic categories). T4+ uses v2 test set (23 examples, adds XML multi-step, complex email, hard negatives, more quote/correction variants). **Accuracy percentages are NOT comparable across the boundary.** Compare T1-T3 against each other, and T4+ against each other.

### T1-T3: v1 test set (12 examples), v1 training data (472 train, 8 valid)

| Run | Checkpoint | Quant | Prompt | N | Accuracy | Exact | Sem | Part | Fail | Latency | Notes |
|-----|-----------|-------|--------|---|----------|-------|-----|------|------|---------|-------|
| T1 | iter 200 | 6-bit | generic v1 | 12 | **58%** | — | — | — | — | 0.82s | |
| T1 | iter 600 | bf16 | generic v1 | 12 | **75%** | — | — | — | — | 1.93s | More training helped generic. |
| T1 | iter 600 | 4-bit | generic v1 | 12 | **67%** | — | — | — | — | 0.66s | 4-bit quant penalty. |
| T2 | iter 200 | 6-bit | generic v1 | 12 | **58%** | — | — | — | — | 0.79s | Same as T1 @200. r=8 = r=16. |
| **T2** | **iter 400** | **6-bit** | **generic v1** | **12** | **75%** | **8** | **1** | **2** | **1** | **0.79s** | **Best on v1 test set.** |
| T2 | iter 400 | bf16 | generic v1 | 12 | **75%** | 8 | 1 | 2 | 1 | 1.75s | bf16 = 6-bit accuracy. |
| T2 | iter 400 | bf16 | v2 | 12 | **58%** | 6 | 1 | 4 | 1 | 1.71s | Prompt mismatch: trained v1, tested v2. -17%. |
| T3 | iter 200 | bf16 | generic v1 | 12 | **50%** | 5 | 1 | 6 | 0 | 0.63s | Llama 1B. Fixed hallucinations but 25% gap vs Qwen. |

### T4+: v2 test set (23 examples), v2 training data (447 train, 20 valid)

| Run | Checkpoint | Quant | Prompt | N | Accuracy | Exact | Sem | Part | Fail | Latency | Notes |
|-----|-----------|-------|--------|---|----------|-------|-----|------|------|---------|-------|
| T2 | iter 400 | 6-bit | generic v1 | 23 | **65%** | 14 | 1 | 8 | 0 | 0.82s | Old best on new test set. Nails basics, fails XML/compound. |
| T4 | iter 300 | bf16 | generic v1 | 23 | **61%** | 13 | 1 | 9 | 0 | 1.91s | Prompt mismatch: trained v2, tested v1. |
| **T4** | **iter 300** | **bf16** | **v2** | **23** | **74%** | **15** | **2** | **6** | **0** | **1.89s** | **Best on v2 test set. Self-correction #3 FIXED. Quote #6 scope FIXED.** |
| T4 | iter 300 | 6-bit | v2 | 23 | **65%** | 14 | 1 | 8 | 0 | 2.04s | 6-bit lost 2 edge cases vs bf16 (quote-endquote, code-aware). |
| T5 | iter 100 | bf16 | v2 | 23 | **30%** | 6 | 1 | 15 | 1 | 17.91s | Undertrained (100 iters, val 0.272). Extreme latency. Inconclusive. |
| T6 | iter 200 | bf16 | v2 | 23 | **43%** | 9 | 1 | 12 | 1 | 3.39s | Undertrained (200 iters, val 0.231). Same val loss as T4@200. |
| T6 | iter 200 | 6-bit | v2 | 23 | **43%** | 9 | 1 | 12 | 1 | 1.94s | **Zero quant loss** (bf16 = 6-bit). AdamW weight decay helps. |
| **T6b** | **iter 500** | **bf16** | **v2** | **23** | **70%** | **15** | **1** | **7** | **0** | **1.82s** | **AdamW 500 iters. 4 pts below T4 — still undertrained (val 0.200 vs 0.174).** |
| T6b | iter 500 | 6-bit | v2 | 23 | **57%** | 12 | 1 | 10 | 0 | 1.14s | 13% quant loss. Zero quant loss from T6 did NOT hold at higher accuracy. |
| T6c | iter 600 | bf16 | v2 | 23 | **65%** | 13 | 2 | 8 | 0 | 2.11s | "Best" val loss (0.190) but worse accuracy — optimizer reset artifact. |
| **T6c** | **iter 800** | **bf16** | **v2** | **23** | **74%** | **15** | **2** | **6** | **0** | **2.87s** | **Matches T4! AdamW needs 800 iters vs T4's 300 (2.7x more).** |
| T6c | iter 800 | 6-bit | v2 | 23 | **61%** | 12 | 2 | 9 | 0 | 2.61s | 13% quant loss. AdamW worse than T4's 9% (65%). |
| T7 | — | 6-bit | v2 | 23 | **—** | — | — | — | — | — | |
| T8 | iter 200 | bf16 | v2 | 23 | **30%** | 5 | 2 | 14 | 2 | 17.75s | Only 50 effective steps (grad_accum=4). DoRA latency ~18s. Not viable. |
| T9 | — | 6-bit | v2 | 23 | **—** | — | — | — | — | — | |
| T10 | — | 6-bit | v2 | 23 | **—** | — | — | — | — | — | |

### T11+: v3 test set (23 examples), v3 training data (492 train, 20 valid)

| Run | Checkpoint | Quant | Prompt | N | Accuracy | Exact | Sem | Part | Fail | Latency | Notes |
|-----|-----------|-------|--------|---|----------|-------|-----|------|------|---------|-------|
| **T11** | **iter 300** | **bf16** | **v2** | **23** | **83%** | **18** | **1** | **3** | **1** | **2.67s** | **New best! +9 pts over T4 bf16. Data quality > hyperparameters.** |
| **T11** | **iter 300** | **6-bit** | **v2** | **23** | **74%** | **16** | **1** | **5** | **1** | **1.76s** | **New 6-bit best! +9 pts over T4 6-bit (65%). Quant loss 9% (same as T4).** |
| T11 | iter 300 | 6-bit | v2 | 23 (v2 test) | **61%** | 13 | 1 | 8 | 1 | 1.02s | Cross-test: T11 on original v2 test set. Lower than T4's 65% — fails removed categories. |
| T4 | iter 300 | 6-bit | v2 | 23 (v2 test) | **65%** | 14 | 1 | 8 | 0 | 1.04s | Re-run for comparison. On 17 kept-category examples: **76%** = same as T11. |
| **T12** | **iter 300** | **bf16** | **v2** | **23** | **74%** | **16** | **1** | **5** | **1** | **2.23s** | **REGRESSION from T11 (83%). Patch fixed 0/4 targets, caused 2 new regressions (#2 spell, #14 emphasis).** |

### Alternative Models: v3 test set (23 examples)

| Run | Checkpoint | Quant | Prompt | N | Accuracy | Exact | Sem | Part | Fail | Latency | Notes |
|-----|-----------|-------|--------|---|----------|-------|-----|------|------|---------|-------|
| **LFM2-T1** | **iter 800** | **bf16** | **v2** | **23** | **78%** | **18** | **0** | **5** | **0** | **1.54s** | **9% → 78%! +69 pts from fine-tuning. 1.7x faster than Qwen3. Nails all 3 self-corrections (Qwen3 misses #6).** |

---

## Quantization Impact

From T2 (r=8, iter 400). Same adapters, different quant levels.

| Quant | Size | Generic Acc | Latency | Delta vs 6-bit |
|-------|------|-------------|---------|----------------|
| bf16 | 7.5 GB | 75% | 1.75s | = |
| 8-bit | 4.0 GB | 75%* | 0.79s | = |
| **6-bit** | **3.1 GB** | **75%** | **0.79s** | **reference** |
| 4-bit | 2.1 GB | 58% | 0.81s | -17% (2 examples) |

*8-bit was measured during thermal throttle; re-measured matches 6-bit.

**Verdict:** 6-bit is the deployment target. 4-bit loses too much.

---

## Persistent Failures (v1 data)

These 2 examples fail across ALL training configs (T1, T2, T3) and all quant levels.

| Test | Category | Input | Expected | Model Output | Root Cause |
|------|----------|-------|----------|-------------|-----------|
| #3 | self-correction | "...Cloudflare Workers and Groq. Wait no, sorry, Fireworks." | "...Cloudflare Workers and Fireworks." | "...Fireworks." | Drops compound phrase. Training teaches "replace whole clause", but needs partial replacement. |
| #6 | quote-endquote | "...quote lucky to be here. end quote..." | "...\"lucky to be here\"..." | "...\"lucky\" to be here." | Wraps 1 word instead of phrase. Only 12 quote-endquote training examples (2.4% of data). |

**T4 (v2 data) adds targeted examples for both patterns.** This is the primary expected improvement.

---

## Data Leakage

Discovered 2026-03-01. Four test examples are exact copies of few-shot examples in `spoke/bench/prompts.py`:

| Test ID | Category | Status |
|---------|----------|--------|
| #1 | spell-replace | Leaked |
| #7 | quote-unquote | Leaked |
| #14 | emphasis | Leaked |
| #20 | camelcase | Leaked |

**Impact (v1 test set, 12 examples):** Spoke-prompt baselines were inflated ~20%. The "58% spoke baseline" was actually 38% on clean examples. The "83% fine-tuned spoke" was 75% clean. Leakage still present in v2 test set (4/23 = 17%). Clean results use generic or v2 prompts only — not affected by this leakage.

---

## Experiment Queue

### Completed

| Run | What Changed | Result |
|-----|-------------|--------|
| T4 | v2 data | 74% bf16, 65% 6-bit. Self-correction #3 FIXED, quote #6 scope FIXED. |
| T5 | DoRA | Dead end. OOM + 18s latency on M4 24GB. |
| T6/T6b/T6c | AdamW | Same bf16, worse 6-bit (61% vs 65%), 2.7x slower. Adam wins for 6-bit deploy. |
| T8 | DoRA + AdamW | Dead end. OOM forced batch=1. |
| T11 | v3 data (492 train) | **83% bf16, 74% 6-bit.** Best model. +9 pts over T4 (test-set aligned). |
| T12 | v3 + patch (535 train) | **REGRESSED to 74% bf16.** Patch data was correct but undertrained (300 iters on 535 examples). |
| LFM2-T1 | LFM2-2.6B-Exp, 800 iters, all 30 layers | **78% bf16, 1.54s latency.** 9% zero-shot → 78% fine-tuned (+69 pts). Validates "zero-shot ≠ fine-tune potential." |

### Active Queue

| Priority | ID | What Changes | Hypothesis | Depends On |
|----------|-----|-------------|-----------|------------|
| **HIGH** | **Q1** | **Mixed-bit quantization (`mixed_4_6`) on T11 fused model** | **Allocate 6-bit to critical layers (v_proj, down_proj, lm_head, first/last layers), 4-bit elsewhere. May close 9% quant gap while keeping model smaller. Zero retraining.** | T11 fused model |
| **ACTIVE** | **LFM2-T1b** | **LFM2-2.6B-Exp LoRA extended to 2000 iters** | **78% at 800 iters, val loss still dropping (0.480). Can more training close the 5-pt gap with Qwen3 T11 (83%)?** | Training in progress (resumed from iter 800) |
| Medium | B-new | **Zero-shot baselines: Qwen3-1.7B, Gemma 3 1B QAT, Llama 3.2 3B** | **Determine if task is capacity-limited or data-limited.** LFM2 baselines done (9%). Need transformer-only small models. | Add models to benchmark script |
| Medium | T13 | Cosine LR + warmup (50 steps), 300 iters | Accelerate convergence. May reach T11 quality faster. | T12b done |
| Medium | T-enc | **Evaluate T5Gemma 2 (1B-1B encoder-decoder)** | **Encoder-decoder is architecturally better for "editing" tasks. Encoder sees full input before decoder generates. May fix compound self-correction (#6) where decoder-only fails.** | Port/load in MLX |
| Low | T9 | QLoRA (4-bit base model) | Same quality, 9GB → ~4-5GB training memory. | — |
| Low | T10 | mask_prompt: false | More gradient signal for short outputs (~15 tok). | — |

### Research-Informed Roadmap

Based on 2025-2026 ASR post-processing literature review. See finding #25.

**Phase A — Squeeze current setup (Qwen3-4B decoder-only)**
1. T12b: fix undertrained regression (400 iters)
2. Q1: mixed-bit quantization to close 9% quant gap
3. T13: cosine LR if more speed needed

**Phase B — Explore smaller models (capacity vs data question)**
4. ~~Zero-shot baselines: LFM2.5-1.2B~~ ✅ Done (9%). LFM2-2.6B-Exp ✅ Done (9%). Remaining: Qwen3-1.7B, Gemma 3 1B QAT, Llama 3.2 3B
5. LFM2-2.6B-Exp LoRA fine-tune ✅ **78% bf16 at 800 iters, 1.54s latency.** Extended to 2000 iters (LFM2-T1b) 🔄
6. Compare: accuracy, latency, memory, quant robustness

**Phase C — Architecture pivot (encoder-decoder)**
7. Evaluate T5Gemma 2 (1B-1B) for editing task
8. If enc-dec helps compound self-correction → port/fine-tune
9. This is the "right architecture" bet but highest effort

**Phase D — Training method innovation (bounded editing)**
10. Experiment with "minimal edit" training format (output ≈ input with targeted changes)
11. Constrained decoding (N-best or span-repair objectives)
12. Only if SFT on current data plateaus across architectures

---

## Key Findings

1. **r=8 = r=16** for this formatting task. Half the params (7.3M → 3.7M), same accuracy, later overfitting onset.
2. **6-bit is deploy quant.** 4-bit loses 2 examples. bf16/8-bit = 6-bit quality but 2-4x larger.
3. **V2 prompt doubles zero-shot** accuracy (25% → 50% on v1 test, 13% → 35% on v2 test).
4. **Fine-tuning gain is +61 points** on v2 test set (13% → 74% bf16). Biggest single improvement.
5. **V2 test set is harder** — floor drops from 25% → 13%, v2 prompt from 50% → 35%. Accuracy NOT comparable across test set versions.
6. **Training/inference prompt mismatch costs 13%** (74% → 61% on v2 test). Always match.
7. **83% was actually 75%** after correcting for data leakage. All spoke benchmarks are contaminated.
8. **V2 data fixed persistent failures** — self-correction #3 (compound phrase) and quote-endquote #6 (multi-word scope) both improved with targeted training examples.
9. **6-bit quant regression worse on T4** — lost 9% (74% → 65%) vs T2's zero loss. Edge cases (quote-endquote, code-aware) are quant-sensitive.
10. **Llama 1B not viable.** 25% accuracy gap vs Qwen. Overfits 2.5x faster.
11. **Overfitting timeline:** r=8 starts ~iter 350 on v2 data, ~iter 500 on v1 data. Llama 1B starts ~iter 200.
12. **DoRA not viable on M4 24GB.** +1-5 GB peak mem over LoRA, OOMs at batch_size≥2, ~18s inference latency (10x LoRA). Converges slower per token.
13. **AdamW is worse for 6-bit quantization** — Consistent 13% quant loss across T6b (70%→57%) and T6c (74%→61%), vs T4's 9% (74%→65%). T6's "zero quant loss" at 200 iters was an artifact of low accuracy (43%). AdamW's weight decay distributes information more evenly, so quantization noise hits more weights.
14. **grad_checkpoint halves training memory** — 14 GB → 9.8 GB peak. ~2x slower training but enables reliable completion.
15. **grad_accumulation_steps breaks iter count** — mlx_lm counts micro-batches, not optimizer steps. `accum=4` means 200 "iters" = 50 effective updates.
16. **AdamW delays overfitting significantly** — T4 (adam) overfits at ~350, T6b (adamw) still dropping at 500. Weight decay extends the useful training window by 40%+. But converges slower — val loss 0.200 at iter 500 vs T4's 0.174 at iter 300.
17. **Mac sleep kills Metal GPU state** — always use `caffeinate -dims` (display, idle, memory, system) for training runs. T6 OOM'd at iter 1 on same config that ran fine before sleep.
18. **AdamW conclusion: same bf16, worse 6-bit, 2.7x slower.** T6c matched T4's 74% bf16 at 800 iters (vs 300), but 6-bit is 61% vs 65%. For deploy (6-bit), adam (T4) wins. AdamW only helps if you deploy bf16.
19. **Val loss is noisy with 20 validation examples.** T6c at val loss 0.205 (iter 800) scored 74%, while val loss 0.190 (iter 600) scored only 65%. Don't trust small val loss differences.
20. **Optimizer state resets on resume.** `--resume-adapter-file` loads weights but resets AdamW momentum/velocity. Causes ~50-iter warm-up period with temporarily higher val loss.
21. **v3 data = +9 pts across the board.** T11 (v3, 492 train, trigger-matched) vs T4 (v2, 447 train, mixed): bf16 83% vs 74%, 6-bit 74% vs 65%. Same config, same quant loss (9%). Removing untriggered categories + adding targeted examples was the highest-ROI change in the entire project. Data quality > hyperparameters confirmed.
22. **T11's +9 pts is test-set alignment, not model improvement.** On the 17 shared kept-category examples from the original v2 test, T4 and T11 both score 76%. T11 gained #9 (quote-endquote) and #12 (at-symbol) but lost #6 (compound self-correction) and #14 (emphasis). The v3 test set improvement comes from removing 6 impossible-category test examples and adding 6 new ones in categories where T11 excels.
23. **v3 data agent over-churned: 57% of v2 touched.** 105 removed (vs ~61 expected), 150 added (vs ~114 expected). Only 3 wrongly removed. Real gap: 150 new examples but only 1 was self-correction — the exact category where T11 regressed.
24. **T12 regression was undertrained, not bad data.** Opus review confirmed all 43 patch examples are correct. The regression (83% → 74%) was caused by training 535 examples for only 300 iters (2.2 passes/example vs T11's 2.4). The 2 new regressions (#2 spell, #14 emphasis) were in categories that received 0 new examples — collateral damage from reduced per-example repetitions in minority categories (emphasis = 3.6% of data). Fix: train longer (400 iters).
25. **Decoder-only is the wrong architecture for editing tasks.** (Research finding, 2025-2026 ASR post-processing literature.) ASR cleanup is fundamentally seq2seq rewriting, not autoregressive generation. Encoder-decoder models (T5, T5Gemma 2) separate input comprehension (encoder) from output generation (decoder with cross-attention), making "preserve A, replace B" natural. This likely explains the persistent compound self-correction failure (#6) — decoder-only Qwen3 processes input and generates output in the same left-to-right pass, making it hard to selectively preserve earlier tokens while replacing later ones. Encoder-decoder evaluation queued as Phase C.
26. **Mixed-bit quantization may close the 9% quant gap.** mlx_lm supports `--quant-predicate mixed_4_6` which allocates 6-bit to critical layers (v_proj, down_proj, lm_head, first/last 12.5% of layers) and 4-bit elsewhere. Critical attention layers are where structural understanding lives — exactly where our quant-sensitive failures (quote-endquote, code-aware) originate. Total size ≤ uniform 6-bit but with bits allocated where they matter.
27. **Task may be data-limited, not capacity-limited.** 535 training examples converge in ~300 iters. Failures are about precision (correction scope, quote scope), not understanding. A 1-2B model with better architecture or more data could match 4B Qwen3. Zero-shot baselines on LFM2.5-1.2B, Qwen3-1.7B, and Gemma 3 1B will test this.
28. **LFM2 hybrid conv+attention can't parse meta-linguistic commands zero-shot.** Both LFM2.5-1.2B (4-bit) and LFM2-2.6B-Exp (bf16) scored 9% on v3 test. Conv-dominant architecture (22/30 layers are ShortConv) extracts keywords or paraphrases instead of executing commands. Architecture bottleneck, not precision — 2x params and bf16 didn't help.
29. **Few-shot teaches output FORMAT but not command EXECUTION.** LFM2-2.6B jumped 9% → 30% with spoke few-shot prompt. Learned "output should be a cleaned sentence" but still couldn't execute spelling, quoting, or emoji commands. Format ≠ reasoning.
30. **IFBench ≠ meta-linguistic instruction following.** LFM2-2.6B-Exp outperforms DeepSeek R1 on IFBench but scored 9% on our task. IFBench tests structural instructions ("respond in JSON"). Our task requires understanding "spell that K-A-D-A-I" as a command to execute — fundamentally different capability.
31. **MoE models: full memory cost for partial compute.** Qwen3.5-35B-A3B has 3B active params but needs all 35B in memory (~20 GB at 4-bit). Not viable on 24 GB consumer hardware. MoE optimizes compute, not memory.
32. **At-symbol is the hardest synthetic data category.** LLMs generating training data want to "improve" text, but ASR post-processing needs surgical edits. 17% failure rate in v3 data generation. Manual crafting required.
33. **Emphasis trigger-to-format mapping must be 1:1 in training data.** "Emphasize"/"Bold" → `**bold**`, "Stress" → ALL CAPS. Even one outlier creates ambiguity. Standardize uniformly — one trigger word, one output format, no exceptions.
34. **Zero-shot accuracy does NOT predict fine-tuning potential.** LFM2-2.6B scored 9% zero-shot but reached 78% fine-tuned (+69 pts). Qwen3-4B scored 22% zero-shot and reached 83% (+61 pts). The model that gained MORE from fine-tuning had the WORSE zero-shot. Never rule out a model based on zero-shot alone.
35. **LFM2 hybrid architecture solves compound self-correction.** LFM2-T1 nails all 3 self-corrections including #6 ("React and Svelte") which was a persistent Qwen3 failure. Conv layers may help with local pattern matching for corrections — challenges finding #25's "decoder-only can't edit" hypothesis. The conv+attention hybrid has complementary strengths.
36. **LFM2 shows double-descent learning curve.** Val loss plateaued at 0.70 for iters 250-400, then dropped steeply to 0.480 by iter 800. Hypothesis: LoRA first adapts attention layers (quick gains), then conv-layer adapters kick in (second drop). More iters may unlock further phases.
37. **LFM2 is 1.7x faster at inference than Qwen3.** 1.54s avg vs 2.67s on same test set. The 22 ShortConv layers are cheaper than full attention. For latency-sensitive deployment, 78% at 1.54s may beat 83% at 2.67s.
