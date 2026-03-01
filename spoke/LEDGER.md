# Spoke Experiment Ledger

> Single source of truth for every training run, benchmark, and planned experiment.
> Last updated: 2026-03-01

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

**Takeaway:** v3 test set zero-shot baseline is 22%. T4-equivalent training on v3 data should push well past 74% since we removed untriggered categories the model can't handle.

---

## Training Runs

All runs use Qwen3-4B-Instruct-2507-bf16 unless noted. All use `mask_prompt: true`, `batch_size: 4`, `lr: 1e-5`, `num_layers: 16`, `seed: 42`.

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

| Priority | Run | What Changes | Hypothesis | Depends On |
|----------|-----|-------------|-----------|------------|
| ~~DONE~~ | ~~T4~~ | ~~v2 data~~ | ~~74% bf16, 65% 6-bit. #3 FIXED, #6 scope FIXED.~~ | ~~Done~~ |
| ~~DONE~~ | ~~T5~~ | ~~DoRA~~ | ~~OOM + 18s latency. Not viable on M4 24GB.~~ | ~~Dead end~~ |
| ~~DONE~~ | ~~T6~~ | ~~AdamW (200 iters)~~ | ~~Zero quant loss! But undertrained.~~ | ~~Rerun as T6b~~ |
| ~~DONE~~ | ~~T8~~ | ~~DoRA + AdamW~~ | ~~OOM forced batch=1. Not viable.~~ | ~~Dead end~~ |
| ~~DONE~~ | ~~T6b~~ | ~~AdamW 500 iters~~ | ~~70% bf16, 57% 6-bit. Still dropping at 500 — needs more iters. Quant loss 13% (worse than T4's 9%).~~ | ~~Done~~ |
| ~~DONE~~ | ~~T6c~~ | ~~AdamW 800 iters~~ | ~~74% bf16 (=T4), 61% 6-bit (worse than T4's 65%). 2.7x slower. **AdamW loses on 6-bit deploy.**~~ | ~~Done~~ |
| ~~DONE~~ | ~~T11~~ | ~~v3 data (492 train, trigger-matched)~~ | ~~**83% bf16, 74% 6-bit. +9 pts over T4. Data quality wins.**~~ | ~~Done~~ |
| Medium | T7 | Cosine LR + warmup (50 steps) | Accelerate convergence — adam converges at 300, maybe cosine gets there at 200? | T4 baseline |
| Medium | T9 | QLoRA (4-bit base model) | Same quality, 9GB → ~4-5GB memory | T4 baseline |
| Medium | T10 | mask_prompt: false | More gradient signal for short outputs (~15 tok) | T4 baseline |
| Low | Expand to 650-750 examples | Target optimal data volume per research | After T6b |
| Low | DPO on persistent failures | Preference learning for edge cases | Only after SFT plateau |

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
