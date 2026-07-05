# Spoke Experiment Ledger

> Single source of truth for every training run, benchmark, and planned experiment.
> Last updated: 2026-07-05 (New-model bake-off round complete: MiniCPM5-1B 43/24/38 (finding #95), Nanbeige4.1-3B 70/40/66 (finding #96), Nemotron-3-Nano-4B parked after dual-pole LR failure (finding #97), Ministral 3 3B invalidated by broken mistral3 text-only loading. Champion unchanged: Qwen3-4B T4-v5split 100/74/82. Modal workspace: sandy-36852; flaky local network cancels attached modal runs — retry loops mandatory.)

## How to Read This

- **Clean accuracy** = generic or v2 prompt only (no few-shot, no data leakage). The only honest metric.
- **Spoke accuracy** is crossed out where shown — 4/23 test examples leaked into the few-shot prompt. See [Data Leakage](#data-leakage) below.
- **Data v1** = 472 train, generic v1 system prompt (~30 tokens), includes multi-command category.
- **Data v2** = 447 train / 20 valid / 23 test, v2 system prompt (~80 tokens), multi-command removed, targeted self-correction + quote-endquote fixes, XML tag fixes.
- **Accuracy** = (exact + semantic) / N. Test set: 12 examples (v1) or 23 examples (v2).
- **DWQ 4-bit was the deploy quant** (96%, 2.1 GB) but weights were lost in cleanup. T3-v5 MLX bf16 is the only surviving model.
- **Broad eval** = 58 unseen examples (`test_set_evals.json`), 0 overlap with training. Best: T4-v5split ckpt2000 at 74%.
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
| B15 | LFM2.5-1.2B | bf16 | v2 | 23 | **9%** | 1 | 1 | 7 | 14 | 0.62s | bf16 = 4-bit (B12). Confirms precision not the bottleneck. |

| B17 | Gemma 3 4B | bf16 | v2 | 23 | **9%** | 0 | 2 | 5 | 16 | 2.28s | Echoes commands back instead of executing them. Repeats "emphasize" as literal text. |
| B18 | Gemma 3 1B | bf16 | v2 | 23 | **0%** | 0 | 0 | 12 | 11 | 5.77s | Garbled `<end_of_turn>` tokens, random Unicode. Unusable zero-shot. |
| B19 | Qwen3-4B | **8-bit** | v2 | 23 | **22%** | 4 | 1 | 12 | 6 | 1.11s | **8-bit = bf16 zero-shot (both 22%).** Confirms 8-bit precision loses nothing at zero-shot. Faster (1.11s vs 1.66s). |
| B20 | Qwen3.5-2B (Modal HF text-only) | bf16 | v2 | 23 | **9%** | 0 | 2 | 1 | 20 | 0.40s | Base cloud benchmark (`result_Qwen-Qwen3.5-2B_modal_v2_test_set_v3.json`). Loader fixed via Transformers 5.2 + `Qwen3_5ForCausalLM`. |
| B21 | Qwen3.5-4B (Modal HF text-only) | bf16 | v2 | 23 | **13%** | 2 | 1 | 5 | 15 | 0.55s | Base cloud benchmark (`result_Qwen-Qwen3.5-4B_modal_v2_test_set_v3.json`). Still weak on command execution. |
| B22 | Flan-T5-base (220M, Modal HF) | bf16 | t5 prefix | 23 | **17%** | 3 | 1 | 3 | 16 | 0.13s | Encoder-decoder zero-shot. T5 prefix format ("Correct this transcription: ..."). 32K SentencePiece vocab can't generate emoji = hard 13% ceiling (3/23 emoji tests). |

**Takeaway:** v3 test set zero-shot baseline is 22% (Qwen3). LFM2 scores 9% regardless of size/precision/quantization — conv-dominant hybrid architecture can't handle meta-linguistic commands zero-shot. Gemma 3 4B also 9% (echoes commands). Gemma 3 1B 0% (garbled). Few-shot helps format (30%) but not reasoning. But 9% zero-shot → 83-87% fine-tuned, so zero-shot is meaningless for predicting fine-tune potential.

### v5 test set (131 examples) — Zero-shot with Spoke prompt

| ID | Model | Quant | Prompt | Temp | N | Accuracy | Exact | Sem | Part | Fail | Latency | Notes |
|----|-------|-------|--------|------|---|----------|-------|-----|------|------|---------|-------|
| B23 | Qwen3.5-4B (MLX) | bf16 | spoke-full | 0.0 | 131 | **37%** | 45 | 3 | 58 | 25 | 5.33s | Echoes spell commands, leaves "quote-unquote" verbatim, describes emoji instead of rendering. |
| B24 | Qwen3.5-4B (MLX) | bf16 | spoke-full | 0.2 | 131 | **37%** | 44 | 4 | 58 | 25 | 6.51s | Temperature has no effect. |
| B25 | Qwen3.5-4B (MLX) | bf16 | spoke-full | 0.6 | 131 | **37%** | 45 | 4 | 58 | 24 | 6.00s | Temperature has no effect. |

### broad58 — Zero-shot with Spoke prompt

| ID | Model | Quant | Prompt | Temp | N | Accuracy | Exact | Sem | Part | Fail | Latency | Notes |
|----|-------|-------|--------|------|---|----------|-------|-----|------|------|---------|-------|
| B26 | Qwen3.5-4B (MLX) | bf16 | spoke-full | 0.0 | 58 | **41%** | 19 | 5 | 21 | 13 | 5.69s | Same failure modes as v5. |
| B27 | Qwen3.5-4B (MLX) | bf16 | spoke-full | 0.6 | 58 | **34%** | 17 | 3 | 30 | 8 | 6.48s | Temp hurts zero-shot broad — more partials. |

**Takeaway:** Temperature (0.0 to 0.6) has zero effect on zero-shot Qwen3.5-4B accuracy. The model doesn't understand *how* to execute editing directives, so adding randomness doesn't help. Greedy decoding is optimal for copy-heavy tasks.

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
| **T11-ext** | 03-02 | LoRA | r=8 | adam | flat | **v3 (535)** | 2000 | 0.107 @450 | **All 36 layers** (vs T11's 16). 16.5M trainable (0.411%). Peak 18.6 GB. Val loss plateaued ~0.11 by iter 400, slowly rose to 0.156 by iter 2000. Train loss 0.000 from iter 1200. **91% bf16 at iter 2000, 3.15s latency.** |
| **T2-v4** | 03-03 | LoRA | r=8 | adam | flat | **v4 (1201)** | 2000 | 0.065 @1100 | **V4 data** (535 v3 + 377 new regular + 289 hard negatives). All 36 layers. Peak 18.6 GB. Val loss best at iter 1100, overfit to 0.091 by iter 2000. Both checkpoints score 100%. **100% bf16 (23/23 exact) at iter 2000, 1.82s latency.** |
| **T3-v5** | 03-05 | LoRA | r=8 | adam | flat | **v5 (1287)** | 2000 | — | **Cloud (Modal L40S, HF+PEFT).** V5 data (1201 v4 + 86 targeted: multi-step, spell-compound, emphasis-caps, meta-language). Trained with v2 prompt. Checkpoint 1200 selected. **100% on v3 test (23/23), 69% on broad eval (58 ex). Fixes Wispr Flow scoping bug. Model at `spoke/models/spoke-qwen3-t3-v5-mlx/`.** |
| **EPO-w3** | 03-07 | LoRA | r=8 | adam | flat | **v5 (1287)** | 1200 | — | **Cloud (Modal L40S, HF+PEFT). EPO loss (edit_weight=3.0).** Same config as T3-v5 except EPO loss upweights edit tokens 3x. Best eval_loss at step 600. **87% core23 (-13 pts), 66% broad58 (-3 pts). REGRESSION. Over-editing failure: drops words, over-scopes. Dead end.** |
| **T4-v5split** | 03-07 | LoRA | r=8 | adam | flat | **v5-split (1046)** | 1000 | 0.115 @1000 | **Cloud (Modal L40S, HF+PEFT). 80:10:10 stratified split** (1046 train / 131 valid / 131 test). Same recipe as T3-v5. Eval loss monotonically decreasing (131 val examples = actually meaningful). **100% on v3 test (23 ex), 79% on v5 test (131 ex). 0 fails. Hard-neg 100%, at-symbol 20%, multi 50%, spell 67%.** |
| **T5-v4prompt** | 03-08 | LoRA | r=8 | adam | flat | **v5-split (1046)** | 2000 | 0.152 @1000 | **Cloud (Modal L40S, HF+PEFT). V4 prompt in training data** (v2 + quote/at-symbol/multi rules, 121 tok). Same split as T4. Tests "train rich, deploy lean" hypothesis. **79% with v4 inference (-3 vs baseline), 73% with v2 inference (-9). REGRESSION. Richer prompt creates dependency — model relies on explicit rules at inference. 4 hard fails with v2 (vs 1 for T4). "Train lean, deploy lean" confirmed as correct strategy (finding #91).** |
| **Qwen3-8B-T1** | 03-08 | LoRA | r=8 | adam | flat | **v5-split (1046)** | 2000 | 0.107 @1000 | **Cloud (Modal L40S, HF+PEFT). Qwen3-8B (2x params of 4B).** Same recipe as T4. Best eval_loss 0.107 at step 1000, rose to 0.175 by step 2000. **Step 1000: 72% (8 fails). Step 2000: 73% (8 fails). -10 pts vs 4B baseline (82%). All 8 fails are quote category — outputs literal "quote-unquote" text. Bigger model = worse for copy-heavy editing (finding #92).** |
| **T5-v4prompt-4k** | 03-08 | LoRA | r=8 | adam | flat | **v5-split (1046)** | 4000 | 0.107 @1000 | **Cloud (Modal L40S, HF+PEFT). Same v4-prompt-trained model as T5, extended to 4000 steps.** Step 3000: 81%, step 4000: 82%. V4 prompt penalty (-3 pts at 2k) erased by 4k steps — recovers to parity with v2 baseline but does NOT exceed it. 2 hard fails at both checkpoints. |
| **T6-v2-4k** | 03-09 | LoRA | r=8 | adam | flat | **v5-split (1046)** | 4000 | 0.107 @1000 | **Cloud (Modal L40S, HF+PEFT). Same recipe as T4-v5split, extended to 4000 steps (15.3 epochs).** V2 prompt throughout. Train loss 0.00007 by step 4000, eval_loss rose from 0.107 (step 1000) to 0.213 (step 4000). **Step 3000: 81%, step 4000: 80%. Both WORSE than step 2000 baseline (82%). Over-memorization confirmed — model degrades past 2000 steps (~7.7 epochs). 0 fails at all checkpoints. Inverted-U training curve: 79% → 82% → 81% → 80% (finding #93).** |

### Alternative Models

| Run | Date | Base Model | Type | Rank | Optimizer | Data | Iters | Best Val Loss | Notes |
|-----|------|-----------|------|------|-----------|------|-------|---------------|-------|
| **LFM2-T1** | 03-01 | LFM2-2.6B-Exp (bf16) | LoRA | r=8 | adam | v3 (535) | 800 | 0.480 @800 | All 30 layers (8 attn + 22 conv). 12.2M trainable (0.476%). Peak 13.3 GB. **78% bf16 at 800 iters.** Double-descent: plateau 250-400, then steep drop 500-800. |
| **LFM2-T1b** | 03-01 | LFM2-2.6B-Exp (bf16) | LoRA | r=8 | adam | v3 (535) | ~900 | 0.466 @~900 | Extended from T1, killed at ~1320 (plateaued). Best ckpt = resumed iter 100. **83% bf16, 19 exact, 1.66s. Ties Qwen3-T11.** |
| **LFM2.5-T1** | 03-02 | LFM2.5-1.2B (bf16) | LoRA | r=8 | adam | v3 (535) | 1000 | 0.506 @550 | All 16 layers. 5.5M trainable (0.475%). Peak 6.5 GB. Plateaued at val loss ~0.51 from iter 400. No double-descent. **70% bf16 at iter 1000, 0.63s latency.** |
| **Llama3-T1** | 03-02 | Llama 3.2 3B Instruct (bf16) | LoRA | r=8 | adam | v3 (535) | 1000 | 0.074 @550 | All 28 layers. 12.2M trainable (0.378%). Peak 15.2 GB. Fastest convergence ever (val 0.074 by iter 550). Overfitting from iter 650 but recovered. **87% bf16 at iter 1000, 1.60s latency.** |
| **Gemma3-T1** | 03-02 | Gemma 3 4B IT (bf16) | LoRA | r=8 | adam | v3 (535) | 1000 | 0.056 @500 | All 34 layers. 14.9M trainable (0.327%). Peak 11.6 GB (grad_checkpoint). OOM'd at 18.9 GB without grad_ckpt. Lowest val loss ever (0.056). Overfitting from iter 500. **87% bf16 at iter 1000, 2.52s latency. Ties Llama.** |
| **Llama3-T2** | 03-03 | Llama 3.2 3B Instruct (bf16) | LoRA | r=8 | adam | v4 (1201) | 2000 | 0.083 @700 | All 28 layers. 12.2M trainable (0.378%). Peak 15.2 GB. Converged fast (0.083 by iter 700), overfit to 0.155 by iter 2000. Iter 2000 still better than iter 700 (91% vs 83%). **91% bf16 at iter 2000, 1.90s. +4 pts over T1 (87%). Doesn't match Qwen3's 100%.** |
| **Qwen3-8bit** | 03-03 | Qwen3-4B-Instruct-2507 (**8-bit**) | QLoRA | r=8 | adam | v4 (1201) | 800 (killed@~850) | 0.276 @800 | All 36 layers. 16.5M trainable (0.411%). Peak 16.4 GB. Starting val loss 6.110 (vs bf16's 2.843 — 8-bit dequant noise). **96% at iter 800, 2.08s latency.** Matches DWQ deploy accuracy but 4% below bf16's 100%. 8-bit QLoRA is slower per-iter than bf16 LoRA (dequant overhead) despite lower memory. |
| **Muon-YOLO** | 03-03 | Qwen3-4B-Instruct-2507-bf16 | LoRA | r=8 | **muon** (lr=2e-4) | v4 (1201) | 900 (died@~940) | 0.123 @850 | **16 layers, max_seq_length=256.** 7.3M trainable (0.182%). Peak 13.3 GB. ~0.17 it/sec (10x slower than Adam — Newton-Schulz overhead dominates for small LoRA params). Val loss beat Adam T11 (0.123 vs 0.169) with same 16 layers, but **78% bf16, 1.63s latency.** Worse than T11 Adam (83%) despite more data. New camelCase regression (lowercased useTranscription). Muon on LoRA is a dead end locally — slower AND less accurate than Adam. |

| **Nemotron3-T1** | 07-05 | Nemotron-3-Nano-4B (bf16, `nemotron_h` hybrid Mamba-2) | LoRA | r=8 | adam | **v5-split (1046)** | 2000 | — | **Cloud (Modal L40S, HF+PEFT, dedicated torch-2.9 mamba image). Champion recipe + Mamba in_proj/out_proj targets (10.3M trainable, 0.26%). **22% core23 (12 hard fails!) / 26% broad58 / 31% v5-131. Failure mode: echoes input VERBATIM — zero commands executed.** lr=1e-5 too weak to move Mamba mixer weights off the copy prior (finding #97).** |
| **Nemotron3-T2-lr2e4** | 07-05 | Nemotron-3-Nano-4B (bf16, `nemotron_h`) | LoRA | r=8 | adam (wd=0.01) | **v5-split (1046)** | 2000 | — | **Gemma 3n-style recipe (lr=2e-4, constant_with_warmup 0.03, grad_norm=0.3). **0% broad58 / 0% v5-131 — but the OPPOSITE failure: edits execute correctly ("quote-unquote brilliant" → "brilliant" ✓) then repetition collapse ("oorspronkelijke tekst:" loops).** Task learned, stopping broken. Mid-run checkpoints not retained (save_total_limit=5 kept only 1600-2000, all degenerate). Viable LR window may exist at 2e-5–5e-5 but unproven. PARKED (finding #97).** |
| **Ministral3-T1** | 07-05 | Ministral-3-3B-Instruct-2512 (unsloth bf16, `mistral3` VLM) | LoRA | r=8 | adam | **v5-split (1046)** | 2000 | — | **Cloud (Modal L40S, HF+PEFT). INVALID — text-only load path broken.** Trained "successfully" but merged model outputs truncated fragments (0% — all 212 partials, 0 fails). Diagnostic: BASE model through the same `config=text_config` + `AutoModelForCausalLM` path outputs multilingual token salad → the mistral3 text-only path doesn't remap `language_model.*` VLM weight prefixes (analog of finding #83's Unsloth VLM export bug). NOT a Ministral quality verdict. Fix requires loading full `Mistral3ForConditionalGeneration` for text-only use. **PARKED.** |
| **Nanbeige41-T1** | 07-05 | Nanbeige4.1-3B (bf16, `model_type=llama`) | LoRA | r=8 | adam | **v5-split (1046)** | 2000 | — | **Cloud (Modal L40S, HF+PEFT). Champion recipe. Reasoning-tuned 3B (claims to beat Qwen3-4B-2507 on Arena-Hard). 14.2M trainable (0.36%), `enable_thinking=False` enforced. **70% core23, 40% broad58, 66% v5-131. 0 hard fails on core/v5.** Mid-pack: beats MiniCPM5-1B everywhere but 13-21 pts below plain Llama 3.2 3B on core23. Chat/reasoning pedigree does not transfer to copy-heavy editing (finding #96).** |
| **MiniCPM5-T1** | 07-04 | MiniCPM5-1B (bf16, `model_type=llama`) | LoRA | r=8 | adam | **v5-split (1046)** | 2000 | 0.239 @2000 (final; min mid-run, wandb tpw3vwnt) | **Cloud (Modal L40S, HF+PEFT). Champion recipe (lr=1e-5, alpha=16, dropout=0.05, v2 prompt as-is). Standard Llama arch — zero pipeline changes, `enable_thinking=False` enforced. Train loss 0.0027 (memorized). **43% core23, 24% broad58, 38% v5-131. Far below LFM2.5-1.2B's 70% core23. Partial-heavy failure (76/131 partials, only 5 fails): lowercased camelCase, emoji described + rendered, dropped filler words. 1B-class precision ceiling (finding #95).** |

### Encoder-Decoder Models (T5)

All T5 runs use **full fine-tuning** (no LoRA) on Modal L40S (48 GB VRAM). Prefix format: `"Correct this transcription: {input}" → "{output}"`. No chat templates. Trained with `Seq2SeqTrainer` + `DataCollatorForSeq2Seq`. Script: `spoke/cloud/train_t5.py`.

| Run | Date | Base Model | Params | LR | Batch | Data | Steps | Best Val Loss | Notes |
|-----|------|-----------|--------|------|-------|------|-------|---------------|-------|
| **T5-base-v1** | 03-07 | Flan-T5-base | 248M | 3e-4 | 8 | v4 (1201) | 2000 | 0.249 @200 | Full fine-tune. lr=3e-4 caused rapid overfitting — best eval_loss at step 200 (1.3 epochs). **Step 200: 48%. Step 2000: 57%.** Over-memorization confirmed: higher val_loss (0.574) = better accuracy (+9 pts). |
| **T5-large-v1** | 03-07 | Flan-T5-large | 783M | 3e-4 | 8 | v4 (1201) | 2000 | 0.157 @200 | Full fine-tune. Same rapid overfitting. **Step 200: 70%. Step 2000: 70%.** Large plateaus earlier — both checkpoints identical accuracy despite 3.6x val_loss difference. |
| **T5-base-v2** | 03-07 | Flan-T5-base | 248M | 1e-5 | 4 | v4 (1201) | 2000 | 1.152 @2000 | Full fine-tune. lr=1e-5 far too low — eval_loss monotonically decreasing (1.246→1.152, never overfits) but model barely learns. **17% (0 exact, 4 sem) = zero-shot level.** |
| **T5-large-v2** | 03-07 | Flan-T5-large | 783M | 1e-5 | 4 | v4 (1201) | 2000 | 0.772 @2000 | Full fine-tune. lr=1e-5. Same pattern as base: smooth decrease (0.902→0.772), zero overfitting, but underfitting. **30% (1 exact, 6 sem, 7 partial, 9 fail).** Large model recovers +13 pts over base at low LR due to stronger priors. |
| **T5Gemma2-v1** | 03-07 | T5Gemma 2 1B-1B | 2.1B | 5e-5 | 8×2 | v4 (1201) | 2000 | 0.398 @100 | Full fine-tune. FlanEC recipe (lr=5e-5, linear+warmup, wd=0). **0% accuracy — labels missing EOS token.** 26.6 epochs, train_loss→0.00015, catastrophic repetition. Degenerate output until max_new_tokens. |
| **T5Gemma2-v2** | 03-07 | T5Gemma 2 1B-1B | 2.1B | 5e-5 | 8×2 | v4 (1201) | 300 | 0.359 @225 | Full fine-tune. Same recipe, fewer steps. **Still 0% — same EOS bug.** Healthy eval_loss curve but model can't stop generating. |
| **T5Gemma2-v3** | 03-07 | T5Gemma 2 1B-1B | 2.1B | 5e-5 | 8×2 | v4 (1201) | 300 | 0.402 @225 | Full fine-tune. **EOS fix: append eos_token_id to labels.** FlanEC recipe + wd=0.01. **70% (16 exact, 0 sem, 7 partial). Emoji 3/3 ✓.** Spell-replace 0/3, caps hallucination 0/2. Latency 0.84s. |
| **Qwen35-T2-cloud** | 03-04 | Qwen3.5-4B (Unsloth, Modal L40S) | LoRA | r=8 | adam | v4 (1201) | 2000 | — | **Cloud Unsloth + Qwen3.5-4B VLM. ~2 it/s after fast-path fix. MLX conversion succeeded (mlx-lm 0.30.7) but model generates incoherent garbage — 0% accuracy. Hybrid DeltaNet+attention architecture broken in MLX inference. Abandoned.** |
| **Qwen35-2B-cloud-smoke** | 03-06 | Qwen3.5-2B (HF text-only, Modal L40S) | LoRA | r=8 | adam | v5 (1287) | 50 | — | **HF text-only smoke path (`Qwen3_5ForCausalLM` + `text_config`) now works. train_steps_per_second=0.777. core23 moved 9% → 22%.** |
| **Qwen35-4B-cloud-smoke** | 03-06 | Qwen3.5-4B (HF text-only, Modal L40S) | LoRA | r=8 | adam | v5 (1287) | 50 | — | **HF text-only smoke path works at 4B too. train_steps_per_second=1.035. core23 moved 13% → 30%.** |
| **Llama3-cloud-v5-v1** | 03-06 | Llama 3.2 3B Instruct (HF, Modal L40S) | LoRA | r=8 | adam | v5 (1287) | 1200 | 0.2765 @200 | **Cloud HF+PEFT, 256 seq. eval_loss best (step 200) = 35%, step 1200 = 78% (16 exact, 2 sem, 5 partial, 0 fail). eval_loss is unreliable — step 200 barely trained.** |
| **Llama3-cloud-v5-3k** | 03-06 | Llama 3.2 3B Instruct (HF, Modal L40S) | LoRA | r=8 | adam | v5 (1287) | 3000 | 0.2785 @200 | **Cloud HF+PEFT, 256 seq, 9.3 epochs. Step 3000 = 83% (18 exact, 1 sem, 4 partial, 0 fail). +5 pts over 1200 steps. 0 fails. eval_loss rose 0.28→0.39 but accuracy kept improving.** |
| **Gemma3n-cloud-v5-v1** | 03-06 | Gemma 3n E2B-it (HF text-only, Modal L40S) | LoRA | r=8 | adam | v5 (1287) | 1200 | 0.659 @600 | **Cloud HF+PEFT, 256 seq. ~4.47B params (2B effective). Step 600 = 70%, step 1200 = 65%. Overfit after step 600. 4 persistent fails on quotes. lr=1e-5 likely 20x too low (Google recommends 2e-4).** |
| **Llama3-cloud-v4-v1** | 03-06 | Llama 3.2 3B Instruct (HF, Modal L40S) | LoRA | r=8 | adam | **v4 (1201)** | 2000 | 0.182 @800 | **V4 vs V5 A/B test. Cloud HF+PEFT, 256 seq, 6.7 epochs. Step 2000 = 87% (19 exact, 1 sem, 3 partial, 0 fail). Confirms v5 interference: v4=87% > v5=83%. Cloud-vs-local gap only 4 pts (87% vs 91%).** |
| **Gemma3n-E2B-v2** | 03-06 | Gemma 3n E2B-it (HF text-only, Modal L40S) | LoRA | r=16 | adam (wd=0.01) | **v4 (1201)** | 1200 | 0.697 @600 | **Google-recommended hyperparams: lr=2e-4, constant_with_warmup, warmup_ratio=0.03, max_grad_norm=0.3. 65% → 91% core23. 59% broad58. Both ckpt 600 and 1200 score identically. ~1.0 GB at 4-bit.** |
| **Gemma3n-E4B-v1** | 03-07 | Gemma 3n E4B-it (HF text-only, Modal L40S) | LoRA | r=16 | adam (wd=0.01) | **v4 (1201)** | 1200 | 0.643 @500 | **Same Google hyperparams as E2B-v2. 96% core23 (matches Qwen3 DWQ!). 59% broad58 (same as E2B). Both ckpt 500 and 1200 identical. ~2.0 GB at 4-bit.** |
| **Llama3-v4-v2** | 03-07 | Llama 3.2 3B Instruct (HF, Modal L40S) | LoRA | r=16 | adam (wd=0.01) | **v4 (1201)** | 1200 | 0.215 @400 | **"Optimal" hyperparams: lr=2e-4, constant_with_warmup, warmup=0.03, max_grad_norm=0.3. 78% at both ckpt 400 and 1200 — WORSE than old lr=1e-5 (91%). Higher LR causes different failure profile (emoji garbled, emphasis ignored, quote scope wrong). 0 fails.** |
| **Qwen3-v4-v2** | 03-07 | Qwen3-4B-Instruct-2507 (HF, Modal L40S) | LoRA | r=16 | adam (wd=0.01) | **v4 (1201)** | 1200 | 0.153 @100 | **Same "optimal" hyperparams as Llama. 96% at step 1200 — 4 pts below old lr=1e-5 (100%). Best eval_loss at step 100 (1/3 epoch!), then overfit. Truncates profanity sentence on quote-unquote. 0 fails.** |
| **Gemma3n-E4B-v5-2k** | 03-07 | Gemma 3n E4B-it (HF text-only, Modal L40S) | LoRA | r=16 | adam (wd=0.01) | **v5 (1287)** | 2000 | 0.595 @100 | **Google hyperparams (lr=2e-4, constant_with_warmup, warmup=0.03, grad_norm=0.3), seq_length=512. 83% core23 (19 exact, 0 sem, 4 partial, 0 fail). 64% broad58 (37 exact, 0 sem, 21 partial, 0 fail). V5 hurt core (-13 pts vs v4's 96%) but helped broad (+5 pts vs v4's 59%). Step 2000 = identical to best ckpt (epoch 1.2): 83% core, 64% broad, same failures. Model converges very early.** |
| **Qwen3-T2-cloud** | 03-04 | Qwen3-4B-Instruct-2507 (Unsloth, Modal L40S) | LoRA | r=8 | adam | v4 (1201) | 2000 | — | **Original cloud fast-path run. Packing enabled (1201→327 packed seqs), lora_dropout=0.0. MLX-converted benchmark = 35% (5 exact / 3 semantic / 11 partial / 4 fail), 1.65s latency. This run was not apples-to-apples, so packing/overexposure was a valid confound, but later strict-parity rerun showed the main remaining gap is post-training MLX conversion/inference, not this setup alone.** |
| **Qwen3-T2-cloud-parity** | 03-04 | Qwen3-4B-Instruct-2507 (Unsloth, Modal L40S) | LoRA | r=8 | adam | v4 (1201) | 2000 | 0.152 @2000 | **Strict local-parity cloud rerun: packing OFF, lora_dropout=0.05, mlx-style mask_prompt labels, collator, and batch ordering. MLX-converted benchmark still = 35% (5 exact / 3 semantic / 11 partial / 4 fail, 2.38s). But direct Modal HF benchmark of the same merged bf16 model = 87% (20 exact / 3 partial / 0 fails, 0.28s). Training is mostly fine; the big regression is in MLX conversion and/or MLX inference.** |

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
| T11-ext | iter 400 | bf16 | v2 | 23 | **83%** | 18 | 1 | 4 | 0 | 3.20s | Best val loss region (0.110). Matches T11 at same iter count but all 36 layers. 0 fails. |
| **T11-ext** | **iter 2000** | **bf16** | **v2** | **23** | **91%** | **21** | **0** | **2** | **0** | **3.15s** | **Previous best. 83% → 91% (+8 pts). 0 fails. Only 2 partials (both emphasis CAPS vs bold). 2000 iters + all 36 layers.** |

### T2-v4: v3 test set (23 examples), v4 training data (1201 train, 20 valid)

| Run | Checkpoint | Quant | Prompt | N | Accuracy | Exact | Sem | Part | Fail | Latency | Notes |
|-----|-----------|-------|--------|---|----------|-------|-----|------|------|---------|-------|
| T2-v4 | iter 1100 | bf16 | v2 | 23 | **100%** | **23** | **0** | **0** | **0** | **2.24s** | Best val loss (0.065). Perfect score. |
| **T2-v4** | **iter 2000** | **bf16** | **v2** | **23** | **100%** | **23** | **0** | **0** | **0** | **1.82s** | **NEW ALL-TIME BEST. 91% → 100%. 0 partials, 0 fails. Faster inference than iter 1100. V4 data (1201 train) broke the 91% ceiling.** |

### T3-v5: v3 test set (23 examples), v5 training data (1287 train), cloud HF+PEFT

| Run | Checkpoint | Quant | Prompt | N | Accuracy | Exact | Sem | Part | Fail | Latency | Notes |
|-----|-----------|-------|--------|---|----------|-------|-----|------|------|---------|-------|
| **T3-v5** | **ckpt 1200** | **bf16 (MLX)** | **v2** | **23** | **100%** | **23** | **0** | **0** | **0** | **4.28s** | **Cloud HF+PEFT on Modal L40S → MLX convert. V5 data + v2 prompt. Matches T2-v4's 100%. Fixes Wispr Flow multi-word scoping bug.** |

### T3-v5: Broad Eval (58 unseen examples, `test_set_evals.json`)

| Run | Quant | Prompt | N | Accuracy | Exact | Sem | Part | Fail | Latency | Notes |
|-----|-------|--------|---|----------|-------|-----|------|------|---------|-------|
| T2-v4 DWQ | DWQ 4-bit | v2 | 58 | **67%** | 38 | 1 | 18 | 1 | 0.89s | Previous best on broad eval. |
| **T3-v5** | **bf16 (MLX)** | **v2** | **58** | **69%** | **38** | **2** | **16** | **2** | **4.28s** | **New best. +2 pts over DWQ. V5 data improved multi (14%→43%), spell (75%→88%), quote (50%→75%). Regressed emoji (100%→50%), disfluency (75%→50%).** |

**Category breakdown (T3-v5 vs DWQ-T2, broad eval):**

| Category | DWQ-T2 | T3-v5 | Delta |
|----------|--------|-------|-------|
| multi | 14% (1/7) | **43% (3/7)** | **+29** |
| quote-unquote | 50% (2/4) | **75% (3/4)** | **+25** |
| spell-replace | 75% (6/8) | **88% (7/8)** | **+12** |
| emoji | 100% (4/4) | 50% (2/4) | -50 |
| disfluency | 75% (3/4) | 50% (2/4) | -25 |
| passthrough | 62% (10/16) | 62% (10/16) | 0 |
| self-correction | 100% (6/6) | 100% (6/6) | 0 |

### T4-v5split: v5 test set (131 examples, stratified 80:10:10 split)

| Run | Quant | Prompt | N | Accuracy | Exact | Sem | Part | Fail | Latency | Notes |
|-----|-------|--------|---|----------|-------|-----|------|------|---------|-------|
| T4-v5split-ckpt1000 | bf16 (Modal) | v2 | 131 | 79% | 96 | 7 | 28 | 0 | 0.26s | Step 1000 (best eval_loss 0.115). 0 fails. Hard-neg 100%. at-symbol 20%. |
| **T4-v5split-ckpt2000** | **bf16 (Modal)** | **v2** | **131** | **82%** | **103** | **4** | **24** | **0** | **0.24s** | **NEW BEST. Step 2000 beats step 1000 despite worse eval_loss (0.164 vs 0.115). at-symbol 60% (+40), self-corr 89% (+16). 6th confirmation eval_loss ≠ accuracy.** |
| T4-v5split-ckpt2000 | bf16 (Modal) | v3 | 131 | 79% | 98 | 5 | 28 | 0 | 0.24s | v3 prompt (191 tok) helps multi (+40) and caps (+14) but kills hard-neg (-21) and emphasis (-25). Net wash. |
| T5-v4prompt-ckpt2000 | bf16 (Modal) | v4 | 131 | 79% | 99 | 5 | 26 | 1 | 0.25s | Trained with v4 prompt (121 tok). -3 pts vs T4 baseline. 1 fail (spell). Prompt dependency created. |
| T5-v4prompt-ckpt2000 | bf16 (Modal) | v2 | 131 | 73% | 87 | 8 | 32 | 4 | 0.25s | Same model, v2 inference. -9 pts. 4 fails (quotes output as literal "quote-unquote"). Prompt mismatch penalty. |
| Qwen3-8B-ckpt1000 | bf16 (Modal) | v2 | 131 | 72% | 84 | 10 | 29 | 8 | 0.57s | Qwen3-8B (2x params). 8 fails — all quotes as literal "quote-unquote". Bigger model worse. |
| Qwen3-8B-ckpt2000 | bf16 (Modal) | v2 | 131 | 73% | 88 | 8 | 27 | 8 | 0.39s | Step 2000 barely better (+1 pt). Same 8 fails. Extra training doesn't fix quote failure mode. |
| T5-v4prompt-ckpt3000 | bf16 (Modal) | v4 | 131 | 81% | 99 | 7 | 23 | 2 | 0.25s | V4-trained 4k run, step 3000. +2 pts vs step 2000 (79%). 2 fails persist. |
| T5-v4prompt-ckpt4000 | bf16 (Modal) | v4 | 131 | 82% | 100 | 7 | 22 | 2 | 0.26s | V4-trained 4k run, step 4000. Recovers to parity with v2 baseline. 2 fails. |
| T6-v2-4k-ckpt3000 | bf16 (Modal) | v2 | 131 | 81% | 102 | 4 | 25 | 0 | 0.25s | V2-trained 4k run, step 3000. -1 pt vs step 2000 baseline. 0 fails. |
| **T6-v2-4k-ckpt4000** | **bf16 (Modal)** | **v2** | **131** | **80%** | **102** | **3** | **26** | **0** | **0.25s** | **V2-trained 4k run, step 4000. -2 pts vs step 2000 baseline. Over-memorization. 0 fails.** |
| T4-v5split-ckpt2000 | bf16 (Modal) | v2 (t=0.6) | 131 | **82%** | 103 | 4 | 24 | 0 | 0.26s | temp=0.6 = identical to greedy. Temperature has no effect on fine-tuned models. |

**Category breakdown (v5 test set 131 ex, step 2000):**

| Category | N | v2 step1k | v2 step2k | v3 step2k | 2k vs 1k | v3 vs v2 |
|----------|---|-----------|-----------|-----------|----------|----------|
| at-symbol | 5 | 20% | **60%** | 60% | **+40** | 0 |
| multi | 10 | 50% | 30% | **70%** | -20 | **+40** |
| spell | 21 | 67% | 67% | **71%** | 0 | +5 |
| emoji | 10 | 70% | **80%** | 80% | +10 | 0 |
| self-correction | 19 | 74% | **89%** | 74% | **+16** | -16 |
| quote | 12 | 83% | 83% | **92%** | 0 | +8 |
| caps | 7 | 86% | 86% | **100%** | 0 | +14 |
| emphasis | 8 | 88% | 88% | 62% | 0 | **-25** |
| hard-negative | 29 | 100% | **100%** | 79% | 0 | **-21** |
| camelcase | 5 | 100% | 100% | 100% | 0 | 0 |
| disfluency | 3 | 100% | 100% | 100% | 0 | 0 |
| meta | 2 | 100% | 100% | 100% | 0 | 0 |

### T4-v5split: Broad Eval (58 unseen examples, `test_set_evals.json`)

| Run | Quant | Prompt | Temp | N | Accuracy | Exact | Sem | Part | Fail | Latency | Notes |
|-----|-------|--------|------|---|----------|-------|-----|------|------|---------|-------|
| **T4-v5split-ckpt2000** | **bf16 (Modal)** | **v2** | **0.6** | **58** | **74%** | **41** | **2** | **13** | **2** | **0.27s** | **NEW ALL-TIME BEST broad58. +5 pts over Qwen3.5 (71%), +5 pts over T3-v5 (69%). 80:10:10 split = better generalization than full-data training. 2 fails: at-symbol echo, meta quote-unquote.** |

### Gemma 3n E4B v1: Broad Eval (58 unseen examples, `test_set_evals.json`)

| Run | Quant | Prompt | N | Accuracy | Exact | Sem | Part | Fail | Latency | Notes |
|-----|-------|--------|---|----------|-------|-----|------|------|---------|-------|
| **Gemma3n-E4B-v1** | **bf16** | **v2** | **58** | **59%** | **32** | **2** | **24** | **0** | **0.74s** | **Same broad score as E2B despite +5 on core23. 0 fails. Fixed disfluency (0/4→1/4) but regressed spell (5/8→4/8). Broad gap is data-limited, not capacity-limited.** |

### Gemma 3n E2B v2: Broad Eval (58 unseen examples, `test_set_evals.json`)

| Run | Quant | Prompt | N | Accuracy | Exact | Sem | Part | Fail | Latency | Notes |
|-----|-------|--------|---|----------|-------|-----|------|------|---------|-------|
| **Gemma3n-E2B-v2** | **bf16** | **v2** | **58** | **59%** | **29** | **5** | **24** | **0** | **1.19s** | **0 fails. Disfluency worst (0/4 — paraphrases instead of minimal edit). Multi weak (1/6). 8 pts below Qwen3 DWQ (67%).** |

**Category breakdown (Gemma3n-E2B-v2 vs DWQ-T2, broad eval):**

| Category | DWQ-T2 | Gemma3n-E2B-v2 | Delta |
|----------|--------|----------------|-------|
| disfluency | 75% (3/4) | **0% (0/4)** | **-75** |
| emoji | 100% (4/4) | 50% (2/4) | -50 |
| multi | 14% (1/7) | 17% (1/6) | +3 |
| spell-replace | 75% (6/8) | 63% (5/8) | -12 |
| passthrough | 62% (10/16) | 62% (8/13) | 0 |
| self-correction | 100% (6/6) | 67% (4/6) | -33 |
| at-symbol | — | 75% (3/4) | — |
| camelcase | — | 100% (1/1) | — |
| quote-unquote | 50% (2/4) | 75% (3/4) | +25 |

**Test set fixes applied (2026-03-06):** ID 45 emphasis `**bold**`→CAPS, ID 46 recategorized disfluency→emoji, ID 48 added trailing period, IDs 25/26 curly→straight quotes, ID 36 period placement. These fixed scoring artifacts, not model behavior.

### Alternative Models: v3 test set (23 examples)

| Run | Checkpoint | Quant | Prompt | N | Accuracy | Exact | Sem | Part | Fail | Latency | Notes |
|-----|-----------|-------|--------|---|----------|-------|-----|------|------|---------|-------|
| LFM2-T1 | iter 800 | bf16 | v2 | 23 | **78%** | 18 | 0 | 5 | 0 | 1.54s | 9% → 78%! +69 pts. 1.7x faster than Qwen3. Nails all 3 self-corrections. |
| **LFM2-T1b** | **iter ~900** | **bf16** | **v2** | **23** | **83%** | **19** | **0** | **3** | **1** | **1.66s** | **TIES Qwen3-T11! 19 exact (one MORE than Qwen3). 1.6x faster. Best LFM2 checkpoint.** |
| **LFM2.5-T1** | **iter 1000** | **bf16** | **v2** | **23** | **70%** | **16** | **0** | **7** | **0** | **0.63s** | **1.2B model, 9% → 70% (+61 pts). 4.2x faster than Qwen3. 0 fails. Phone-deployable at ~1.2 GB.** |
| Llama3-T1 | iter 500 | bf16 | v2 | 23 | **83%** | 19 | 0 | 4 | 0 | 1.62s | Best val loss region (0.078). Ties Qwen3-T11/LFM2-T1b. 0 fails. |
| **Llama3-T1** | **iter 1000** | **bf16** | **v2** | **23** | **87%** | **20** | **0** | **3** | **0** | **1.60s** | **26% → 87% (+61 pts). 1.7x faster than Qwen3. 0 fails. Beats 83% ceiling by 4 pts.** |
| Gemma3-T1 | iter 500 | bf16 | v2 | 23 | **83%** | 19 | 0 | 4 | 0 | 2.07s | Best val loss region (0.056). 0 fails. Misses emphasis and camelCase. |
| **Gemma3-T1** | **iter 1000** | **bf16** | **v2** | **23** | **87%** | **20** | **0** | **2** | **1** | **2.52s** | **9% → 87% (+78 pts). Ties Llama. 1 fail (self-correction #3). Wider than Llama but 1 more fail.** |
| Llama3-T2 | iter 700 | bf16 | v2 | 23 | **83%** | 19 | 0 | 4 | 0 | 1.77s | Best val loss (0.083). Worse than iter 2000 despite better val loss. |
| **Llama3-T2** | **iter 2000** | **bf16** | **v2** | **23** | **91%** | **20** | **1** | **2** | **0** | **1.90s** | **87% → 91% with v4 data (+4 pts). 0 fails. Still 9 pts below Qwen3's 100%.** |
| **Muon-YOLO** | **iter 900** | **bf16** | **v2** | **23** | **78%** | **17** | **1** | **4** | **1** | **1.63s** | **16 layers + Muon + 256 seq. Worse than Adam T11 (83%) despite v4 data. New camelCase regression. 1 fail (emoji). Dead end for LoRA.** |
| Llama3-cloud-v5-v1 | step 200 (best eval) | bf16 | v2 | 23 | **35%** | 6 | 2 | 11 | 4 | — | Cloud Modal HF. eval_loss best = barely trained (0.6 epochs). Trap. |
| **Llama3-cloud-v5-v1** | **step 1200** | **bf16** | **v2** | **23** | **78%** | **16** | **2** | **5** | **0** | **—** | **Cloud Modal HF, v5 data. 0 fails. Below local v4 result (91%).** |
| **Llama3-cloud-v5-3k** | **step 3000** | **bf16** | **v2** | **23** | **83%** | **18** | **1** | **4** | **0** | **—** | **Cloud Modal HF, v5 data, 9.3 epochs. +5 pts over 1200 steps. 0 fails. Still 8 pts below local v4 (91%).** |
| Gemma3n-cloud-v5 | step 600 (best eval) | bf16 | v2 | 23 | **70%** | 10 | 6 | 3 | 4 | — | Cloud Modal HF. Gemma 3n E2B text-only. Best checkpoint by eval_loss. |
| **Gemma3n-cloud-v5** | **step 1200** | **bf16** | **v2** | **23** | **65%** | **12** | **3** | **4** | **4** | **—** | **Cloud Modal HF. Overfit past step 600. 4 fails on quotes. lr=1e-5 too low.** |
| **Gemma3n-E2B-v2** | **step 600** | **bf16** | **v2** | **23** | **91%** | **20** | **1** | **2** | **0** | **0.51s** | **Google hyperparams (lr=2e-4, warmup, grad_norm=0.3). 65% → 91% (+26 pts). 0 fails. ~1.0 GB at 4-bit.** |
| Gemma3n-E2B-v2 | step 1200 | bf16 | v2 | 23 | **91%** | 20 | 1 | 2 | 0 | 0.52s | Same score as step 600. Identical errors on both checkpoints. |
| **Gemma3n-E4B-v1** | **step 500** | **bf16** | **v2** | **23** | **96%** | **20** | **2** | **1** | **0** | **1.20s** | **Matches Qwen3 DWQ! 0 fails. Fixed E2B's Cloudflare hallucination + emphasis word-swap.** |
| Gemma3n-E4B-v1 | step 1200 | bf16 | v2 | 23 | **96%** | 20 | 2 | 1 | 0 | 1.20s | Same as step 500. Both checkpoints identical. |
| Llama3-cloud-v4 | step 800 (best eval) | bf16 | v2 | 23 | **74%** | 13 | 4 | 6 | 0 | — | Cloud Modal HF, v4 data. eval_loss best = undertrained. |
| **Llama3-cloud-v4** | **step 2000** | **bf16** | **v2** | **23** | **87%** | **19** | **1** | **3** | **0** | **—** | **Cloud Modal HF, v4 data. Confirms v4 > v5 for 3B. 4 pts below local (91%). 0 fails.** |
| Llama3-v4-v2 | step 400 (best eval) | bf16 | v2 | 23 | **78%** | 17 | 1 | 5 | 0 | 0.16s | LR experiment: lr=2e-4, r=16. WORSE than lr=1e-5 (91%). Emoji garbled, emphasis ignored. |
| **Llama3-v4-v2** | **step 1200** | **bf16** | **v2** | **23** | **78%** | **17** | **1** | **5** | **0** | **0.14s** | **Same failures as ckpt 400. Higher LR converges early but to a worse solution.** |
| **Qwen3-v4-v2** | **step 1200** | **bf16** | **v2** | **23** | **96%** | **21** | **1** | **1** | **0** | **0.21s** | **LR experiment: lr=2e-4, r=16. 4 pts below lr=1e-5 (100%). Truncates profanity sentence. 0 fails.** |
| **Qwen3-T2-cloud** | **iter 2000** | **bf16** | **v2** | **23** | **35%** | **5** | **3** | **11** | **4** | **1.65s** | **Original cloud fast-path run. Packing ON (1201→327 packed seqs), dropout=0.0. Confounded and not apples-to-apples. 35% after MLX conversion.** |
| **Qwen3-T2-cloud-parity (MLX)** | **iter 2000** | **bf16** | **v2** | **23** | **35%** | **5** | **3** | **11** | **4** | **2.38s** | **Strict local-parity rerun: packing OFF, dropout=0.05, mlx-style mask_prompt/collator/batch ordering. Still 35% after MLX conversion, so the original packing theory does not explain the full regression.** |
| **Qwen3-T2-cloud-parity (Modal HF)** | **iter 2000** | **bf16** | **v2** | **23** | **87%** | **20** | **0** | **3** | **0** | **0.28s** | **Exact same merged bf16 model benchmarked directly on Modal with Transformers before MLX conversion. 35% → 87% proves the main quality loss is downstream in MLX conversion and/or MLX inference, not in the cloud training itself.** |
| **Qwen3-T2-cloud-parity-nothink (Modal HF)** | **iter 2000** | **bf16** | **v2** | **23** | **74%** | **17** | **0** | **6** | **0** | **0.28s** | **Hard no-thinking path enforced in training formatter/tokenization with runtime `<think>` guard. Parity profile + `packing=off` + Adam + `max_grad_norm=0.0`. Result matches the prior lower cloud band (`74%`).** |
| **Qwen3-T2-cloud-ultra-nothink (Modal HF)** | **iter 2500** | **bf16** | **v2** | **23** | **83%** | **19** | **0** | **4** | **0** | **0.50s** | **No-thinking enforced + ultra profile (`r=32`, `alpha=64`, `dropout=0.05`, `rsLoRA=True`, `packing=off`). Improves over parity-nothink (74% → 83%) but still below local MLX 100%.** |
| **Qwen3-T2-cloud-parity-templatefix (Modal HF)** | **iter 2000** | **bf16** | **v2** | **23** | **74%** | **17** | **0** | **6** | **0** | **0.49s** | **Template-level no-thinking enforcement: tokenizer chat template with `<think>` was replaced before training and benchmarking, and benchmark prompt building hard-failed on `<think>`. Score remained `74%`, so ignored `enable_thinking` was real but not the sole source of the quality gap.** |
| **Qwen35-2B-cloud-base (Modal HF)** | **base** | **bf16** | **v2** | **23** | **9%** | **0** | **2** | **1** | **20** | **0.40s** | **Base-model probe after HF loader fix (`Qwen3_5ForCausalLM`).** |
| **Qwen35-2B-cloud-smoke50 (Modal HF)** | **iter 50** | **bf16** | **v2** | **23** | **22%** | **2** | **3** | **11** | **7** | **0.29s** | **50-step HF smoke (`spoke-qwen35-2b-hf-smoke50-20260306`). Improves over base but remains far below Qwen3 parity quality.** |
| **Qwen35-4B-cloud-base (Modal HF)** | **base** | **bf16** | **v2** | **23** | **13%** | **2** | **1** | **5** | **15** | **0.55s** | **Base-model probe after HF loader fix (`Qwen3_5ForCausalLM`).** |
| **Qwen35-4B-cloud-smoke50 (Modal HF)** | **iter 50** | **bf16** | **v2** | **23** | **30%** | **5** | **2** | **6** | **10** | **0.45s** | **50-step HF smoke (`spoke-qwen35-4b-hf-smoke50-20260306`). Best Qwen3.5 cloud probe so far, still weak.** |
| **Qwen35-4B-HF-v5-1500 (Modal HF)** | **step 1500** | **bf16** | **v2** | **23** | **96%** | **22** | **0** | **1** | **0** | **0.38s** | **Full 1500-step HF+PEFT run (`spoke-qwen35-4b-hf-v5-v2prompt-1500-20260306-0625`). r=8, alpha=16, dropout=0.05, lr=1e-5, adam, max_seq=256, 12 target modules (incl. in_proj_z/a/b/qkv, out_proj for DeltaNet layers). V5 data. 96% core23 (1 miss: Kibbinay→Kibinaay). 71% broad58 — BEST BROAD EVER. 0 fails.** |
| **Qwen35-4B-r16-rsLoRA (Modal HF)** | **step 2000** | **bf16** | **v2** | **23** | **91%** | **19** | **2** | **2** | **0** | **—** | **rsLoRA experiment (`spoke-qwen35-4b-r16-rslora`). r=16, alpha=16, use_rslora=True (scale=4.0), dropout=0.05, lr=1e-5, adam, max_seq=256. V5 data 2000 steps. REGRESSION: 91% core (vs 96% at r=8). 62% broad (vs 71%). More capacity = more overfitting for copy-heavy tasks. Dead end.** |
| **Qwen35-4B-lr5e5 (Modal HF)** | **step 300 (best eval)** | **bf16** | **v2** | **23** | **91%** | **19** | **2** | **1** | **1** | **0.38s** | **lr=5e-5 experiment. Best eval_loss (0.175) at step 300 (epoch 0.93). 1 fail (quote-unquote). eval_loss minimum but accuracy WORSE than last ckpt.** |
| **Qwen35-4B-lr5e5 (Modal HF)** | **step 1500 (last)** | **bf16** | **v2** | **23** | **96%** | **22** | **0** | **1** | **0** | **0.69s** | **lr=5e-5 last checkpoint. Matches baseline core (96%). 0 fails. 1 partial (at-symbol: @app.py → .app.py). eval_loss 0.28 but accuracy better than step 300.** |
| **Qwen3-4B-EPO-w3 (Modal HF)** | **step 600 (best eval)** | **bf16** | **v2** | **23** | **87%** | **19** | **1** | **3** | **0** | **0.22s** | **EPO loss (edit_weight=3.0). `spoke-qwen3-4b-epo-w3`. Massive regression from baseline 100%. Over-editing: drops "React and", "Okay", "really", "absolutely". EPO upweights edit tokens → model learns aggressive editing → fails on copy-heavy task. Dead end.** |

### Cloud Training: Qwen3.5 (Unsloth History + HF Text-Only Probes)

| Run | Date | Base Model | GPU | Type | Rank | Optimizer | Data | Steps | Status | Notes |
|-----|------|-----------|-----|------|------|-----------|------|-------|--------|-------|
| **Qwen35-T1** | 03-04 | unsloth/Qwen3.5-4B (VLM, 32 layers) | L40S (48 GB) | LoRA bf16 | r=8 | adamw_torch (wd=0) | v4 (1201) | 2000 | **COMPLETED / ABANDONED** | Legacy Unsloth cloud run family; MLX-converted outputs were unusable (0-35% bands). |
| **Qwen35-2B-HF-smoke50** | 03-06 | Qwen/Qwen3.5-2B (text-only HF path) | L40S (48 GB) | LoRA bf16 | r=8 | adam | v5 (1287) | 50 | **COMPLETED** | `train_steps_per_second=0.777`; core23 `22%` after smoke finetune. |
| **Qwen35-4B-HF-smoke50** | 03-06 | Qwen/Qwen3.5-4B (text-only HF path) | L40S (48 GB) | LoRA bf16 | r=8 | adam | v5 (1287) | 50 | **COMPLETED** | `train_steps_per_second=1.035`; core23 `30%` after smoke finetune. |
| **Qwen35-4B-HF-v5-1500** | 03-06 | Qwen/Qwen3.5-4B (text-only HF path) | L40S (48 GB) | LoRA bf16 | r=8 | adam | v5 (1287) | 1500 | **COMPLETED** | `spoke-qwen35-4b-hf-v5-v2prompt-1500-20260306-0625`. 12 target modules (incl. DeltaNet in_proj_z/a/b/qkv, out_proj). **96% core23, 71% broad58 — BEST BROAD EVER.** 0 fails. |
| **Qwen35-4B-r16-rsLoRA** | 03-07 | Qwen/Qwen3.5-4B (text-only HF path) | L40S (48 GB) | LoRA bf16 | r=16 | adam | v5 (1287) | 2000 | **COMPLETED** | `spoke-qwen35-4b-r16-rslora`. rsLoRA (use_rslora=True, scale=4.0). **91% core23, 62% broad58. REGRESSION from r=8 (96%/71%). Dead end.** |
| **Qwen35-4B-lr5e5** | 03-07 | Qwen/Qwen3.5-4B (text-only HF path) | L40S (48 GB) | LoRA bf16 | r=8 | adam | v5 (1287) | 1500 | **COMPLETED** | `spoke-qwen35-4b-lr5e5`. lr=5e-5 (5x baseline). Overfitting visible at step 400 (epoch 1.25). Best eval_loss at step 300 (0.175). **Step 300: 91% core, 64% broad. Step 1500: 96% core, 66% broad. Last checkpoint beats "best" — eval_loss unreliable (finding #54). Same core as baseline (96%) but -5 pts broad (71% → 66%). Higher LR didn't help.** |

**Cloud pipelines now used:**
- Legacy Unsloth path: `spoke/cloud/train.py` (kept for historical experiments).
- Current HF+PEFT path: `spoke/cloud/train_hf.py` + `spoke/cloud/benchmark.py` + `spoke/cloud/merge_adapter_checkpoint.py` with `transformers==5.3.0`. Generalized multimodal text-only detection for `qwen3_5` and `gemma3n`. Uses `AutoModelForCausalLM` for all causal models. Supports `--data-dir` for switching between data versions (default `/data`, v4 at `/data/v4`).
- Shared Modal Volumes: `spoke-model-cache`, `spoke-training-data`, `spoke-output`.

**Key Unsloth setup issues resolved historically** (cost ~$2-3 in failed Modal runs):
1. `<EOS_TOKEN>` placeholder bug — Unsloth replaces eos_token, TRL validates it. Fix: `get_chat_template(tokenizer, "qwen3-instruct")`.
2. TRL version — must pin `trl==0.22.2` (all Unsloth notebooks use this). TRL 0.23+ has breaking EOS checks.
3. Data collation — nested `messages` column crashes tensor creation. Fix: `remove_columns()` after formatting.
4. `causal_conv1d` needs nvcc — skip in `debian_slim` (optional perf dep for Qwen3.5 hybrid layers).
5. `paged_adam_32bit` not in transformers OptimizerNames — use `adamw_torch` with `weight_decay=0.0`.
6. SFTConfig doesn't accept `max_seq_length` — set only in `from_pretrained`.

---

## Quantization Impact

### T2-v4 (v4 data, 1201 train, iter 2000) — Current best

### Naive (round-to-nearest) quantization

| Quant | Size | BPW | Accuracy | Exact | Sem | Part | Fail | Latency | Delta vs bf16 |
|-------|------|-----|----------|-------|-----|------|------|---------|---------------|
| **bf16** | **7.5 GB** | **16** | **100%** | **23** | **0** | **0** | **0** | **1.82s** | **—** |
| 6-bit | 3.1 GB | 6.5 | 96% | 22 | 0 | 1 | 0 | 0.94s | -4% |
| mixed 4/6 | 2.2 GB | 4.75 | 91% | 21 | 0 | 2 | 0 | 0.87s | -9% |
| 4-bit | 2.1 GB | 4.5 | 87% | 19 | 1 | 3 | 0 | 0.78s | -13% |

### Learned quantization (DWQ)

DWQ = Distilled Weight Quantization. Uses bf16 model as teacher to fine-tune quantization scales/biases (125.9M trainable params, 3.1% of total). Calibrated on v4 training data (512 iters, ~42 min, 14.1 GB peak).

| Quant | Size | BPW | Accuracy | Exact | Sem | Part | Fail | Latency | Delta vs bf16 |
|-------|------|-----|----------|-------|-----|------|------|---------|---------------|
| **DWQ 4-bit** | **2.1 GB** | **4.5** | **96%** | **22** | **0** | **1** | **0** | **0.88s** | **-4%** |

DWQ 4-bit failure: emphasis #21 (drops "absolutely" — same as naive 6-bit). Training data gap, not quant artifact.

### Summary

| Method | Size | Accuracy | Latency | Notes |
|--------|------|----------|---------|-------|
| bf16 | 7.5 GB | 100% | 1.82s | Reference |
| naive 6-bit | 3.1 GB | 96% | 0.94s | Good but large |
| **DWQ 4-bit** | **2.1 GB** | **96%** | **0.88s** | **Deploy target. Same accuracy as 6-bit, 33% smaller, fastest.** |
| naive mixed 4/6 | 2.2 GB | 91% | 0.87s | Generic heuristic underperforms DWQ |
| naive 4-bit | 2.1 GB | 87% | 0.78s | DWQ recovers +9 pts at same size |

**Ad-hoc generalization tests (5 novel inputs, not in train/test):** bf16=5/5, **DWQ 4-bit=5/5**, 6-bit=3/5, mixed 4/6=2/5. DWQ matches bf16 on all ad-hoc tests. Naive quant models fail on self-correction and complex spell-replacements — test set accuracy flatters them.

**Verdict:** DWQ 4-bit (2.1 GB, 96%, 0.88s) was the deploy target. Matches naive 6-bit accuracy at 33% less size and 2x faster than bf16.

> **⚠️ DWQ model weights deleted (2026-03-06)** during disk cleanup. Only config files remain at `spoke/models/qwen3-t2-v4-dwq4/` (also cleaned up). All local adapter weights and fused model weights were also lost. T3-v5 MLX bf16 (`spoke/models/spoke-qwen3-t3-v5-mlx/`) is the only surviving model. DWQ quantization of T3-v5 is pending.

### Historical: T2-v1 (v1 data, 472 train, iter 400)

| Quant | Size | Acc | Latency | Delta vs 6-bit |
|-------|------|-----|---------|----------------|
| bf16 | 7.5 GB | 75% | 1.75s | = |
| 8-bit | 4.0 GB | 75% | 0.79s | = |
| **6-bit** | **3.1 GB** | **75%** | **0.79s** | **reference** |
| 4-bit | 2.1 GB | 58% | 0.81s | -17% |

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
| LFM2-T1b | LFM2-2.6B-Exp, extended to ~900 iters | **83% bf16, 1.66s latency.** Ties Qwen3-T11. 19 exact (one more). 1.6x faster inference. Best LFM2 checkpoint. |
| LFM2.5-T1 | LFM2.5-1.2B bf16, 1000 iters, all 16 layers | **70% bf16, 0.63s latency.** 9% → 70% (+61 pts). 0 fails. Plateaued at iter 400 (val 0.51). No double-descent. 4.2x faster than Qwen3. |
| B16 | Llama 3.2 3B bf16 zero-shot, v2 prompt | **26%.** Between Qwen3 (35%) and LFM2 (9%). 12 partials, 5 fails. |
| Llama3-T1 | Llama 3.2 3B bf16, LoRA r=8 adam, 1000 iters, all 28 layers | **87% bf16, 1.60s latency.** 26% → 87% (+61 pts). 0 fails. Lowest val loss ever (0.074). Iter 1000 > iter 500 (87% vs 83%) despite worse val loss. |
| T11-ext | Qwen3-4B bf16, LoRA r=8 adam, 2000 iters, **all 36 layers** | **91% bf16, 3.15s latency. NEW ALL-TIME BEST.** 83% → 91% (+8 pts over T11). 0 fails. Only 2 partials (both emphasis). Val loss 0.156 (worst) but best accuracy (5th confirmation val loss is noise). |
| B17/B18 | Gemma 3 4B/1B bf16 zero-shot, v2 prompt | **4B: 9%, 1B: 0%.** Gemma echoes commands instead of executing. Same as LFM2 zero-shot (9%). |
| Gemma3-T1 | Gemma 3 4B IT bf16, LoRA r=8 adam, 1000 iters, all 34 layers, grad_checkpoint | **87% bf16, 2.52s latency. Ties Llama.** 9% → 87% (+78 pts, largest gain ever). OOM'd without grad_checkpoint (18.9 GB). With grad_ckpt: 11.6 GB peak. Lowest val loss ever (0.056). 1 fail (self-correction #3). |
| **T2-v4** | **Qwen3-4B bf16, v4 data (1201 train), 2000 iters, all 36 layers** | **100% bf16, 1.82s latency. NEW ALL-TIME BEST.** V4 data (535 v3 + 377 regular + 289 hard negatives) broke the 91% ceiling. Both iter 1100 and 2000 score 100%. Val loss overfit after iter 1100 but accuracy unaffected. Generalizes to novel inputs (Celero→Silero, Gamma→Gemma) not in training data. Known gap: drops "Flow" from multi-word "Whisper Flow" spell-replace. |
| **Llama3-T2** | **Llama 3.2 3B bf16, v4 data (1201 train), 2000 iters, all 28 layers** | **91% bf16, 1.90s latency.** 87% → 91% with v4 data (+4 pts). 0 fails. Best val loss 0.083 at iter 700, overfit to 0.155 by iter 2000. Iter 2000 still outperforms iter 700 (91% vs 83%). Doesn't match Qwen3's 100% — 3B capacity ceiling. |
| **Muon-YOLO** | **Qwen3-4B bf16, Muon optimizer (lr=2e-4), 16 layers, 256 seq, v4 data, ~900 iters** | **78% bf16, 1.63s latency. Dead end.** Worse than Adam T11 (83%) despite 2x more data. 10x slower per-iter (Newton-Schulz overhead). New camelCase regression. Muon not viable for LoRA on M4. |
| **T3-v5** | **Qwen3-4B, cloud HF+PEFT (Modal L40S), v5 data (1287 train), v2 prompt, ckpt 1200** | **100% v3 test, 69% broad eval (58 ex). NEW BROAD EVAL BEST.** V5 targeted data (+86 examples: multi-step, spell-compound, emphasis-caps, meta-language) improved multi 14%→43%, spell 75%→88%, quote 50%→75%. Regressed emoji/disfluency. Fixes Wispr Flow scoping bug. Only surviving model: `spoke/models/spoke-qwen3-t3-v5-mlx/`. |
| **Qwen35-HF-smoke-2B/4B** | **Qwen3.5 text-only HF cloud probes (Modal L40S, 50 steps, v5+v2)** | **Compatibility fixed, quality still poor.** Base scores: 2B `9%`, 4B `13%`; after 50-step smoke: 2B `22%`, 4B `30%` on core23. Useful as pipeline validation, not quality candidates. |
| **Llama3-cloud-v5** | **Llama 3.2 3B Instruct, cloud HF+PEFT (Modal L40S), v5 data, 1200-3000 steps** | **78% at 1200 steps, 83% at 3000 steps. 0 fails at both. Below local v4 result (91%) — v5 data causes interference at 3B scale (confirmed by v4 A/B test = 87%). eval_loss best (step 200) was a trap (35%).** |
| **Llama3-cloud-v4** | **Llama 3.2 3B Instruct, cloud HF+PEFT (Modal L40S), v4 data, 2000 steps** | **87% at step 2000. 0 fails. Confirms v5 interference: v4=87% > v5=83%. Cloud-vs-local gap only 4 pts (87% vs 91%). Pipeline is sound — data is the variable.** |
| **T4-v5split** | **Qwen3-4B, cloud HF+PEFT (Modal L40S), v5-split (1046 train), v2 prompt, 2000 steps** | **100% on v3 test (23 ex), 82% on v5 test (131 ex). 0 fails. NEW BEST on meaningful test set.** Stratified 80:10:10 split (1046/131/131). Step 2000 > step 1000 (82% vs 79%) despite worse eval_loss (0.164 vs 0.115). v3 prompt (191 tok) = net wash: helps multi (+40) but kills hard-neg (-21). Step 2000 + v2 prompt is optimal config. |
| **T5-v4prompt** | **Qwen3-4B, cloud HF+PEFT (Modal L40S), v5-split (1046 train), v4 prompt in training data (121 tok), 2000 steps** | **79% v4 inference (-3 pts), 73% v2 inference (-9 pts). REGRESSION.** Richer prompt creates dependency — model relies on explicit rules. 4 hard fails with v2 (quotes as literal "quote-unquote"). "Train lean, deploy lean" confirmed correct (finding #91). |
| **Qwen3-8B-T1** | **Qwen3-8B, cloud HF+PEFT (Modal L40S), v5-split (1046 train), v2 prompt, 2000 steps** | **72% step 1000, 73% step 2000. 8 hard fails (all quotes). -10 pts vs 4B.** Larger model outputs literal "quote-unquote" instead of wrapping in quotes. Stronger base priors resist mechanical transformations. 4B is the right capacity for copy-heavy editing (finding #92). |
| **T5-v4prompt-4k** | **Qwen3-4B, cloud HF+PEFT (Modal L40S), v5-split (1046 train), v4 prompt, 4000 steps** | **Step 3000: 81%, step 4000: 82% (v4 inference). Recovers to parity with v2 baseline (82%) but doesn't exceed it.** 2x compute to erase v4 prompt penalty. 2 hard fails persist. |
| **T6-v2-4k** | **Qwen3-4B, cloud HF+PEFT (Modal L40S), v5-split (1046 train), v2 prompt, 4000 steps** | **Step 3000: 81%, step 4000: 80%. REGRESSION.** More training hurts v2-trained models. Inverted-U curve: 79% → 82% → 81% → 80%. Over-memorization at 15.3 epochs. 0 fails. 2000 steps is the sweet spot (finding #93). |
| **Gemma3n-cloud-v5** | **Gemma 3n E2B-it, cloud HF+PEFT (Modal L40S), v5 data, 1200 steps** | **70% best (step 600), 65% last (step 1200). 4 persistent fails on quotes. lr=1e-5 was 20x too low per Google's recommendation (2e-4). Novel architecture (AltUp/PLE/LAuReL) likely needs different hyperparameters.** |
| **Gemma3n-E2B-v2** | **Gemma 3n E2B-it, cloud HF+PEFT (Modal L40S), v4 data, 1200 steps, Google hyperparams (lr=2e-4, constant_with_warmup, warmup=0.03, grad_norm=0.3, wd=0.01, r=16)** | **91% core23 (20 exact, 1 sem, 2 partial, 0 fail). 59% broad58 (29 exact, 5 sem, 24 partial, 0 fail). 65% → 91% just from fixing hyperparams. Both ckpt 600 and 1200 identical. ~1.0 GB at 4-bit = half of Qwen3 DWQ. Disfluency (0/4) and multi (1/6) are weakest categories on broad.** |
| **Gemma3n-E4B-v1** | **Gemma 3n E4B-it, cloud HF+PEFT (Modal L40S), v4 data, 1200 steps, same Google hyperparams as E2B-v2** | **96% core23 (20 exact, 2 sem, 1 partial, 0 fail) — MATCHES Qwen3 DWQ! 59% broad58 (32 exact, 2 sem, 24 partial, 0 fail) — same as E2B. Extra capacity closed core23 gap (+5 pts) but didn't help broad eval. Broad gap is data-limited, not capacity-limited. ~2.0 GB at 4-bit.** |

### Active Queue

| Priority | ID | What Changes | Hypothesis | Depends On |
|----------|-----|-------------|-----------|------------|
| ~~HIGH~~ | ~~T2-v4-6bit~~ | ~~Fuse + 6-bit quantize T2-v4~~ | ~~DONE. 6-bit=96%, 4-bit=87%, mixed_4_6=91%. Naive quant gap reduced from 9% to 4% (6-bit).~~ | ✅ |
| ~~HIGH~~ | ~~Learned quant~~ | ~~DWQ 4-bit on T2-v4 fused model~~ | ~~DONE. DWQ 4-bit=96% at 2.1 GB — matches naive 6-bit accuracy at 33% less size. 5/5 on ad-hoc tests (vs 3/5 for naive 6-bit). 0.88s latency (fastest). Dynamic quant too slow for 4B on M4 24GB (~220s/layer).~~ | ✅ |
| ~~HIGH~~ | ~~Muon~~ | ~~Train with Muon optimizer~~ | ~~DONE. 78% bf16 at iter 900 (16 layers, 256 seq, v4 data). Worse than Adam T11 (83%) with same 16 layers and less data. Newton-Schulz overhead made it 10x slower per-iter (~0.17 vs ~2 it/sec). New camelCase regression. Dead end for LoRA on M4 — optimizer step overhead dominates when trainable params are tiny.~~ | ✅ |
| ~~HIGH~~ | ~~Cloud pipeline~~ | ~~Modal + Unsloth cloud training for Qwen3.5-4B~~ | ~~DONE. Pipeline working: upload_data.py → train.py (Modal L40S) → download_model.py. Pin trl==0.22.2, use get_chat_template for EOS fix, remove nested columns. Cost ~$2-3 debugging, ~$0.50-1.00/run.~~ | ✅ |
| ~~HIGH~~ | ~~Llama3-T2~~ | ~~Llama 3.2 3B on v4 data, 2000 iters~~ | ~~DONE. 91% bf16 at iter 2000 (+4 pts over T1). Doesn't match Qwen3's 100% — 3B model has a capacity ceiling. 0 fails, 1.90s latency.~~ | ✅ |
| ~~HIGH~~ | ~~Qwen3-8bit~~ | ~~8-bit QLoRA baseline on Qwen3-4B, v4 data~~ | ~~DONE. 96% at iter 800 (killed early). 8-bit = bf16 at zero-shot (both 22%), but fine-tuned ceiling is 96% vs 100%. 8-bit QLoRA is actually SLOWER per-iter than bf16 LoRA (dequant overhead). Peak 16.4 GB (vs 18.6 GB bf16). Not worth it for speed — bf16 is faster AND better.~~ | ✅ |
| ~~Medium~~ | ~~rsLoRA~~ | ~~r=16, scale=4.0 (rsLoRA scaling) on Qwen3.5-4B~~ | ~~DONE. 91% core, 62% broad — REGRESSION from r=8 (96%/71%). More capacity = more overfitting for copy-heavy tasks. Dead end.~~ | ✅ |
| ~~Medium~~ | ~~lr-5e-5~~ | ~~lr=5e-5 (5x baseline) on Qwen3.5-4B~~ | ~~DONE. Step 1500: 96% core (= baseline), 66% broad (-5 pts vs 71%). Faster convergence but worse generalization. lr=1e-5 confirmed correct.~~ | ✅ |
| ~~Medium~~ | ~~EPO~~ | ~~Edit-weighted loss (edit_weight=3.0) on Qwen3 4B~~ | ~~DONE. 87% core (-13 pts), 66% broad (-3 pts). Over-editing: model drops words. EPO wrong for copy-heavy tasks. Dead end.~~ | ✅ |
| **HIGH** | v6-data | Build v6 data targeting weak categories: at-symbol (20-60%), multi (30-50%), spell (67%), emoji (80%) | Only remaining accuracy lever (findings #90, #93). Augmentation ideas preserved from NEXT_RUNS.md (deleted 2026-07-04): semantic rephrase of spell instructions ("spell that K-A-D-A-I" → "the spelling is K-A-D-A-I" → "it should be K-A-D-A-I"), lowercase-input duplicates for passthrough/hard-negative (teaches casing repair as side effect), emphasis-scope examples with nearby distractors. Hand-review mandatory (T12 lesson). | — |
| Medium | Q1 | Mixed-bit quantization (`mixed_4_6`) on T2-v4 fused model | Allocate 6-bit to critical layers, 4-bit elsewhere. May close quant gap further. Zero retraining. | T2-v4 fused model |
| Low | ministral-fix | Proper text-only loading for `mistral3` (load `Mistral3ForConditionalGeneration`, target LoRA at `language_model.*`) | Ministral3-T1 was invalidated by broken weight loading, not model quality. Only worth doing if a 3B slot matters after Gemma 4 E4B. | — |
| Low | nemotron-midlr | Nemotron-3-Nano lr=2e-5 or 5e-5, save_total_limit≥20 | lr=1e-5 echoes, lr=2e-4 collapses — if a window exists it's in between, and early checkpoints must be retained this time. Only worth it if hybrid-architecture broad-generalization (finding #81) motivates it. | — |
| Medium | expand-test | Expand test set from 23 → 50+ examples | 100% on 23 examples is thin. Need harder/novel examples for confidence. Include multi-word spell-replace (Wispr Flow edge case). | — |
| ~~Low~~ | ~~Qwen3.5~~ | ~~Qwen3.5 4B on v4 data~~ | ~~DONE probe phase. HF text-only path now works (2B/4B), but quality is too low at current recipe (30% best smoke).~~ | ✅ |
| Low | B-new | Zero-shot baselines: Qwen3-1.7B | Determine if task is capacity-limited or data-limited. | Add model to benchmark script |
| **DONE** | T-enc | Evaluate T5Gemma 2 (1B-1B encoder-decoder) | 70% accuracy. Emoji works (3/3). EOS bug fixed. Encoder-decoder 30 pts behind decoder-only. | — |
| Low | T9 | QLoRA (4-bit base model) | Same quality, less training memory. | — |

### Research-Informed Roadmap

Based on 2025-2026 ASR post-processing literature review. See finding #25.

**Phase A — Squeeze current setup (Qwen3-4B decoder-only)**
1. T12b: fix undertrained regression (400 iters)
2. Q1: mixed-bit quantization to close 9% quant gap
3. T13: cosine LR if more speed needed

**Phase B — Explore smaller models (capacity vs data question)**
4. ~~Zero-shot baselines: LFM2.5-1.2B~~ ✅ 9%. ~~LFM2-2.6B-Exp~~ ✅ 9%. ~~Llama 3.2 3B~~ ✅ 26%. ~~Gemma 3 4B~~ ✅ 9%. ~~Gemma 3 1B~~ ✅ 0%. Remaining: Qwen3-1.7B
5. LFM2-2.6B-Exp LoRA ✅ **83% bf16, 1.66s** (T1b). LFM2.5-1.2B LoRA ✅ **70% bf16, 0.63s** (T1).
6. Compare: accuracy, latency, memory, quant robustness — see model comparison table below

**Phase B takeaway:** **Qwen3-4B at 91% (T11-ext) is the accuracy champion. Llama 3.2 3B and Gemma 3 4B tie at 87%.** All 4B-class models converge to 87-91% — suggesting the data ceiling is near 91% with current 535 examples. Gemma 3 4B had the largest zero-shot-to-fine-tuned gain (+78 pts) and lowest training memory with grad_checkpoint (11.6 GB). Llama is fastest at inference (2x Qwen3). Architecture matters less than training duration and data quality at this scale.

**Phase C — Architecture pivot (encoder-decoder) — COMPLETE**
7. ~~Evaluate T5Gemma 2 (1B-1B)~~ ✅ — 70% with EOS fix. Emoji 3/3 ✓. Still 30 pts behind decoder-only.
7b. **Flan-T5-base/large full fine-tuning on Modal** ✅ — T5-base peaks at 57%, T5-large at 70% (lr=3e-4). Hard ceiling from 32K SentencePiece vocab (can't generate emoji = 13% loss).
7c. **T5Gemma 2 1B-1B full fine-tuning on Modal** ✅ — 256K vocab solves emoji. EOS bug discovered and fixed (0% → 70%). Encoder-decoder conclusion: not the right architecture for Spoke (finding #76).
8. Deploy path: HF Transformers on Modal for inference, or llama.cpp GGUF, or `mlx-examples/t5/` (inference only).

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
38. **LFM2.5-1.2B validates "phone-deployable" model.** 9% → 70% (+61 pts) at 0.63s latency, 6.5 GB peak training mem. No double-descent (plateaued at val 0.51 from iter 400). Capacity matters: halving params from 2.6B → 1.2B costs 13 accuracy points (83% → 70%) but gains 2.6x speed. Iter 1000 beat iter 500 (70% vs 65%) despite higher val loss — confirms val loss is unreliable as stopping criterion (finding #19).
39. **Llama 3.2 3B zero-shot = 26%.** Lands between Qwen3-4B (35%) and LFM2 (9%). 12 partials, 5 fails. Outputs literal "quote-unquote" instead of converting to quotation marks — classic instruction-following gap.
40. **wandb integrated for live training dashboards.** `report_to: wandb` in config.yaml. Built-in callback in mlx-lm. Also supports `swanlab`.
41. **Llama 3.2 3B hit 87% — briefly the best before T11-ext.** 26% zero-shot → 87% fine-tuned (+61 pts). Broke the 83% ceiling. 0 fails (first model to achieve this). 1.60s latency (2x faster than Qwen3). Lowest val loss ever recorded (0.074 at iter 550). Converges fastest of all models.
42. **T11-ext: 91% — Qwen3-4B was massively undertrained.**
43. **Gemma 3 4B: 87% — ties Llama, largest zero-shot-to-fine-tuned gain.**
44. **V4 data broke the 91% ceiling → 100%.** 535→1201 training examples (377 new regular + 289 hard negatives). Same model (Qwen3-4B), same config (r=8, adam, all 36 layers), same 2000 iters. The 91% ceiling was a DATA ceiling, not a model ceiling. More diverse data > more epochs on same data. Both iter 1100 (best val loss 0.065) and iter 2000 (val loss 0.091, overfit) score 100%. Model generalizes to novel inputs not in training data (Celero→Silero, Gamma→Gemma). Known gap: drops "Flow" from "Whisper Flow" on multi-word product name spell-replace.
46. **DWQ 4-bit recovers 9 pts over naive 4-bit at same size.** DWQ (distilled weight quantization) uses the bf16 model as a teacher to fine-tune quantization scales/biases. Results: 87% → 96% at identical 2.1 GB. Matches naive 6-bit (3.1 GB) in both benchmark accuracy (96%) and ad-hoc generalization (5/5 vs 3/5). 0.88s avg latency — fastest model tested (2x faster than bf16). Task-specific calibration on v4 data was key: DWQ learned which weight rounding decisions matter for ASR post-processing. 512 iters, ~42 min on M4, 14.1 GB peak memory, 125.9M trainable params (3.1% — quant scales/biases only). Dynamic quant was attempted first but too slow (~220s/layer, 56 layers) on M4 24GB.
48. **8-bit QLoRA: 96% accuracy but slower than bf16 LoRA.** Qwen3-4B 8-bit QLoRA on v4 data: 96% at iter 800 (killed early, val loss 0.276). Starting val loss 6.110 (vs bf16's 2.843) — 8-bit dequant noise inflates cross-entropy even though zero-shot accuracy is identical (both 22%). Peak 16.4 GB (vs 18.6 GB bf16 — 12% savings). ~0.12 it/sec (vs ~2 it/sec bf16 — 17x slower). The dequant overhead on every matmul makes 8-bit training slower despite lower memory. 8-bit QLoRA is NOT a speed optimization — it's a memory optimization for models that don't fit in bf16. For Qwen3-4B which fits in bf16 on 24GB, bf16 LoRA is both faster and more accurate. Muon optimizer (`optimizer: muon` in config.yaml) is already wired up in mlx-lm 0.30.5 — not blocked.
47. **Llama 3.2 3B caps at 91% on v4 data — 3B capacity ceiling.** V4 data lifted Llama from 87% (T1, v3) to 91% (T2, v4), same +4 pt pattern. But Qwen3 4B went 91% → 100%. The extra 1B params matter for edge cases (quote-endquote scope, multi-word patterns). Llama iter 2000 beat iter 700 (91% vs 83%) despite 2x worse val loss (0.155 vs 0.083) — sixth confirmation val loss is unreliable. 0 fails at both checkpoints. Llama fixed emoji (✝️ → 🙏) between iter 700 and 2000 — more training helps rare mappings.
45. **Muon optimizer: dead end for LoRA on M4.** ~~Research shows Muon-trained models lose ~0.5% accuracy on quantization vs >3% for Adam.~~ **Tested: 78% bf16 with 16 layers (worse than Adam T11's 83% with same layers and less data).** Newton-Schulz overhead made each iter ~10x slower (0.17 vs ~2 it/sec) — the 5 matrix multiplications per parameter per step dominate when trainable params are tiny (7.3M LoRA). Muon converges faster per-iteration (val 0.123 vs Adam's 0.169 at 16 layers) but the 10x wall-clock cost negates it. New camelCase regression (lowercased useTranscription). Muon was designed for pretraining billions of params on GPU clusters — optimizer overhead is negligible there. For LoRA fine-tuning on M4, Adam is both faster and more accurate. Quant robustness claim untested (accuracy too low to be worth quantizing).
50. **The 35% cloud regression is mostly a post-training MLX-path issue, not just an Unsloth training mismatch.** The original cloud fast-path run was confounded by packing (1201 → 327 packed seqs) and `lora_dropout=0.0`, so overtraining was a fair first hypothesis. But the strict parity rerun disabled packing, restored `lora_dropout=0.05`, and matched local `mlx_lm` prompt masking, collation, and length-based batch ordering; after MLX conversion it still scored 35% (5 exact / 3 semantic / 11 partial / 4 fail). Then the exact same merged bf16 model, benchmarked directly on Modal with Transformers before MLX conversion, scored 87% (20 exact / 3 partial / 0 fails, 0.28s avg latency). Conclusion: cloud training is substantially healthier than the MLX benchmark suggests; the main regression is in MLX conversion and/or MLX inference. Packing remains a real confound for the original fast-path run, but it is not the primary explanation for the parity rerun's 35%. Unsloth's `save_pretrained_merged` still mangles `config.json` (nests `rope_theta`, strips `bos_token_id`, changes `eos_token_id` format), so export metadata still needs patching before `mlx_lm.convert`.
51. ~~**Qwen3.5-4B is broken on MLX.**~~ **CORRECTED: The Unsloth VLM export was broken, not Qwen3.5 itself.** The 0% result was from Unsloth's `save_pretrained_merged` which exported the VLM model (`Qwen3_5ForConditionalGeneration`) instead of text-only. The HF text-only merge path (`Qwen3_5ForCausalLM` + `text_config`) produces valid merged bf16 models scoring 96% on Modal. MLX conversion of the HF text-only merge has not been retested. See finding #83.
49. **16 layers caps at 78-83% regardless of optimizer or data.** Muon-YOLO (16 layers, v4 data 1201 train, 900 iters) = 78%. Adam T11 (16 layers, v3 data 492 train, 300 iters) = 83%. T11-ext (36 layers, same data, 2000 iters) = 91%. T2-v4 (36 layers, v4 data, 2000 iters) = 100%. The upper layers (17-36) carry critical capabilities — likely where structural understanding (quoting, casing, multi-word scope) lives. Layer reduction is not viable for this task.
52. **Qwen3.5 HF text-only loading is now operational on cloud.** Upgrading Modal images to `transformers==5.2.0` and forcing `Qwen3_5ForCausalLM` with `text_config` resolved the `model type qwen3_5 not recognized` blocker and enabled reproducible base/smoke probes.
53. **Qwen3.5 remains low-ROI under the current training recipe.** Core23 moved from base `9%/13%` (2B/4B) to `22%/30%` after 50-step smokes. This confirms pipeline viability but not quality viability versus the established Qwen3 parity path.
54. **eval_loss best checkpoint is harmful for this task.** Both Llama and Gemma 3n "best by eval_loss" checkpoints scored worse than the last checkpoint (Llama: 35% vs 78%, Gemma: 70% vs 65%). With 20 validation examples, eval_loss is noise. `load_best_model_at_end` selected barely-trained checkpoints. Always benchmark accuracy directly.
55. **Llama 3.2 3B maintains zero fails across all cloud runs.** 78% (1200 steps) and 83% (3000 steps) — every example gets at least partial credit. No catastrophic failures. This contrasts with Gemma 3n's 4 persistent hard fails on quote handling.
56. **Gemma 3n E2B lr=1e-5 is 20x too low.** Google's official QLoRA guide recommends lr=2e-4 for Gemma. Community runs use 2e-5 to 2e-4. Our 1e-5 is below the floor of any published successful Gemma 3n fine-tuning. Also recommends max_grad_norm=0.3 (vs our 1.0) and cosine/linear scheduler with warmup.
57. **V5 data causes interference at 3B scale — CONFIRMED.** Llama v4 cloud = 87% vs v5 cloud = 83% (same pipeline, comparable epochs). Local v4 = 91%. The 86 harder v5 examples (multi-step, spell-compound) exceed 3B capacity, causing interference with easier patterns. Qwen3 at 4B is unaffected (100% on both). Cloud-vs-local gap is only 4 pts (87% vs 91%), likely due to minor config differences (max_seq_length 256 vs 512).
59. **Gemma 3n E2B hyperparams matter enormously.** Switching from lr=1e-5 to Google's recommended lr=2e-4 (with constant_with_warmup, warmup=0.03, max_grad_norm=0.3, weight_decay=0.01, r=16) jumped Gemma 3n E2B from 65% to 91% core23. Both ckpt 600 and 1200 score identically (91%) — model converges by epoch 2. Broad eval = 59% (vs Qwen3 DWQ 67%). 0 fails on both evals. Weakest: disfluency (0/4, paraphrases), multi (1/6), emphasis (wrong word substitution).
60. **Gemma 3n E4B matches Qwen3 DWQ on core23 (96%) at ~2.0 GB.** E2B=91% at ~1.0 GB, E4B=96% at ~2.0 GB. Extra capacity closed the core23 gap (+5 pts) but broad58 stayed at 59% for both (vs Qwen3's 67%). The 8pt broad gap is data-limited, not capacity-limited — both Gemma sizes fail on the same categories (disfluency, multi-step, passthrough edits).
61. **Prompt injection vulnerability in E4B.** Broad eval passthrough "ignore previous instructions" — E4B answered the question ("The capital of Italy is Rome") while E2B correctly passed it through. Larger models' stronger instruction-following can backfire for verbatim transcription tasks.
58. **Llama 3.2 3B cloud pipeline is sound.** v4 data cloud = 87% vs local = 91% (4 pt gap). The remaining gap is within noise for 23 test examples (~1 example difference). Cloud HF+PEFT training produces near-local-parity results when data is controlled.
62. **"Optimal" LoRA LR from papers HURTS copy-heavy editing tasks.** Community consensus (lr=2e-4, Meta torchtune lr=3e-4) degraded both Llama (91%→78%, -13 pts) and Qwen3 (100%→96%, -4 pts). Same r=16, same data, same scheduler. Our original lr=1e-5 was correct. Papers optimize for chat/instruction-following where the model learns new behavior. Our task is copy-heavy text editing where the model must learn to *barely* change the input — higher LR pushes too far from base behavior. Exception: Gemma 3n, whose architecture (MatFormer + PLE) is specifically designed for efficient adaptation and benefits from higher LR (Google's own recommendation).
63. **Higher LR creates qualitatively different failures.** Llama at lr=2e-4 outputs `�praying hands` instead of the emoji, ignores emphasis commands, and mis-scopes quotes — these are NOT the same errors as lr=1e-5. Higher LR doesn't just "overfit more"; it learns a different, worse editing policy for this task. Both ckpt 400 (epoch 1.3) and ckpt 1200 (epoch 4) had identical failures — the wrong policy is established early.
64. **Qwen3 at lr=2e-4 peaks at step 100 (1/3 epoch) by eval_loss.** Best eval_loss 0.153 at step 100, then diverges to 0.388 by step 1200. Despite this, step 1200 benchmarks 96% — again confirming val loss is unreliable (finding #19). But step 100 merge failed on tokenizer (chat_template missing), so we couldn't compare.
65. **Task-specific LR tuning is mandatory.** One-size-fits-all "LoRA needs 10x higher LR than full FT" breaks down for tasks requiring conservative behavior. The right LR depends on how far from base behavior the task demands: chat/reasoning → high LR (2e-4), copy-heavy editing → low LR (1e-5), Gemma 3n → follow architecture-specific guidance (2e-4).
66. **T5 encoder-decoder has a hard 13% accuracy ceiling from vocab.** T5's 32K SentencePiece vocab (trained on C4 text) cannot generate emoji Unicode codepoints. 3/23 test examples are emoji → always fail. Maximum theoretical accuracy on v3 test: 87%. This is a vocabulary limitation, not a training limitation — no amount of fine-tuning can fix it.
67. **Over-memorization pattern confirmed for encoder-decoder.** T5-base step 2000 (val_loss=0.574) scored +9 pts better than step 200 (val_loss=0.249). Same divergent pattern as decoder-only Qwen3/Llama. Finding #19 (val loss unreliable) generalizes across architectures.
68. **T5 full fine-tuning: lr=3e-4 causes rapid overfitting.** Best eval_loss at step 200 (~1.3 epochs) for both T5-base and T5-large. This matches the "2 epochs then diverge" pattern from LoRA runs. Encoder-decoder doesn't magically avoid overfitting on 1201 examples.
69. **Encoder-decoder doesn't outperform decoder-only on this task (so far).** Flan-T5-large (783M full fine-tune) = 70% vs Qwen3-4B (LoRA r=8) = 100%. Even accounting for the 13% emoji ceiling, T5-large would max at 87% — below Qwen3's 100%. The "right architecture" hypothesis (finding #25) didn't hold. Possible explanations: (a) Flan-T5 is a 2023 model with weaker base capabilities, (b) 32K vocab is too small, (c) more training/data needed.
70. **T5 inference is extremely fast.** Flan-T5-base: 0.12-0.28s, Flan-T5-large: 0.22s on L40S. Encoder-decoder decode is much faster than autoregressive decoder-only because outputs are short (median 15 tokens). For latency-critical deployment, a 70% T5-large at 0.22s could be preferable to 96% Qwen3 DWQ at 0.88s.
71. **lr=1e-5 is optimal for LoRA but way too low for full fine-tuning.** LoRA updates 0.2-0.4% of params → large effective per-param update. Full FT updates ALL params → negligible per-param update at lr=1e-5. T5-base: 57% (lr=3e-4) vs 17% (lr=1e-5). T5-large: 70% vs 30%. lr=3e-4 overfits but learns the task; lr=1e-5 never overfits but barely learns anything. Ideal T5 full FT LR is likely ~5e-5 to 1e-4 (untested). This does NOT invalidate finding #62 — "our lr=1e-5 was correct" applies specifically to LoRA fine-tuning where the effective update scale is amplified.
72. **T5-large has stronger priors than T5-base at low LR.** At lr=1e-5 (underfitting regime), T5-large (783M) scored 30% vs T5-base (248M) 17%. The extra capacity lets the model extract more from marginal learning. But at lr=3e-4 (adequate LR), both plateau at similar accuracy (70% vs 57%), suggesting the LR bottleneck matters more than model size for T5.
73. **T5Gemma 2's Gemma tokenizer doesn't append EOS to labels — this causes 0% accuracy.** Standard T5 SentencePiece always appends `</s>` (id=1), but Gemma's tokenizer adds BOS at start, NOT EOS at end. Without EOS in training labels, the model never learns to stop generating → degenerate repetition until max_new_tokens. Fix: explicitly append `eos_token_id` to label sequences. This single line of code changed accuracy from 0% → 70%.
74. **T5Gemma 2 (256K vocab) solves Flan-T5's emoji ceiling.** Emoji tests: 3/3 pass (💔, 🙏, 🔥). Flan-T5's 32K SentencePiece vocab can't encode emoji Unicode → hard 13% accuracy loss. T5Gemma 2 with Gemma's 256K tokenizer has no such limitation. This confirms the vocab hypothesis from finding #66.
75. **T5Gemma 2 1B-1B (2.1B params, full FT) = 70%, same as Flan-T5-large (783M).** Despite 2.7x more parameters and a modern architecture (Gemma 3 based), T5Gemma 2 doesn't outperform Flan-T5-large. Possible explanations: (a) 4 epochs may not be enough for 2.1B params (Flan-T5-large ran 26.6 epochs), (b) encoder-decoder cross-attention fundamentally struggles with character-level copy operations (spell-replace 0/3, caps hallucination).
76. **Encoder-decoder conclusion: not the right architecture for Spoke.** Tested across Flan-T5 (base/large) and T5Gemma 2 (1B-1B). Best encoder-decoder: 70%. Best decoder-only: 100% (Qwen3), 96% (Gemma 3n). The 30-pt gap is consistent. Encoder-decoder's cross-attention loses fine-grained character-level control needed for spell-replace and capitalization. Decoder-only's autoregressive copying mechanism is better suited to this task.
77. **V5 data interference confirmed on Gemma 3n E4B: 96% → 83% core23.** Third model showing v5 regression (Llama -4 pts, Gemma -13 pts). Only Qwen3 4B is immune (100% on both). The 86 harder v5 examples (multi-step, spell-compound) interfere with simpler patterns on sub-4B effective capacity models.
78. **V5 data helps broad eval even when hurting core.** Gemma 3n E4B: 59% → 64% broad58 (+5 pts). Disfluency improved from 0/4 to 1/4 (first ever success). Trade-off is bad: -13 core for +5 broad. Better path: improve training data quality (v6) rather than volume (v5).
79. **`load_best_model_at_end` with 20 val examples is actively harmful.** Gemma 3n E4B v5 2k: best eval_loss at epoch 1.2 (step 100, eval_loss=0.595). Model barely trained (1.2 epochs). Previous E4B-v1 "best" was step 500 (epoch 5). The unreliable metric picks early checkpoints that haven't learned the task. Always benchmark the last checkpoint.
80. **Gemma 3n E4B converges very early on v5 data.**
81. **Qwen3.5-4B at r=8 lr=1e-5 = 96% core / 71% broad — BEST BROAD SCORE EVER.** Run `spoke-qwen35-4b-hf-v5-v2prompt-1500-20260306-0625`. Uses 12 LoRA target modules (vs Qwen3's 7) because DeltaNet layers have extra projections (in_proj_z/a/b/qkv, out_proj). V5 data, 1500 steps. Qwen3.5's hybrid DeltaNet+attention architecture provides +2 pts on broad (71% vs Qwen3's 69%) despite slightly lower core (96% vs 100%). The DeltaNet's linear attention may help with diverse patterns.
82. **rsLoRA r=16 HURTS copy-heavy editing tasks.** Qwen3.5-4B with r=16 + rsLoRA (scale=alpha/√r=4.0 vs standard alpha/r=1.0): 91% core (vs 96% at r=8), 62% broad (vs 71%). More LoRA capacity → more overfitting on a task that requires conservative behavior. Standard LoRA r=8 with dampened scaling (1.0) is the right inductive bias for copy-heavy editing. This extends finding #1 (r=8 = r=16 for standard LoRA) — rsLoRA doesn't unlock higher rank, it makes it worse.
83. **Qwen3.5-4B is NOT broken on MLX — the Unsloth VLM export was broken.**
84. **lr=5e-5 = same core (96%) but -5 pts broad (66% vs 71%) on Qwen3.5-4B.** Higher LR converges faster (best eval_loss at step 300 vs step ~1000 for lr=1e-5) but overshoots on generalization. The "best by eval_loss" checkpoint (step 300) scored WORSE than the last checkpoint (step 1500): 91% vs 96% core, 64% vs 66% broad. This is the strongest evidence yet that eval_loss minimum is harmful for checkpoint selection (finding #54). lr=1e-5 remains the correct LR for Spoke.
86. **EPO (edit-weighted loss) HURTS copy-heavy tasks — causes over-editing.** Qwen3 4B with EPO (edit_weight=3.0): 87% core (-13 pts from baseline 100%), 66% broad (-3 pts from 69%). EPO upweights edit tokens 3x in the loss, but for Spoke ~90% of output tokens should be copies. By telling the model "editing = 3x more valuable than copying," it learned to edit MORE aggressively: drops "Okay", "really", "React and", "absolutely" — words that should be preserved. Standard SFT's equal token weighting is the correct inductive bias for copy-heavy tasks because 90% of the gradient signal SHOULD go to copy tokens. EPO is designed for tasks (like GEC) where models under-edit; Spoke's failure mode is over-editing. Dead end.
85. **lr=5e-5 last checkpoint has a new at-symbol regression.** Converts @app.py → .app.py instead of keeping the @ symbol. The baseline (lr=1e-5) handles this correctly. Higher LR creates subtle new failure modes even when headline accuracy matches. Finding #51 was wrong. The HF text-only merge path (`Qwen3_5ForCausalLM` + `text_config`) produces valid merged bf16 models that score 96% on Modal. The 0% result was from the Unsloth VLM export path which corrupted config.json. MLX conversion of the HF text-only merged model has not been retested yet. Step 2000 (epoch ~12.5) = identical scores to "best" checkpoint (epoch 1.2): 83% core23 (same 4 failures), 64% broad58 (same 21 partials). Additional training neither helps nor hurts — the model hits its ceiling quickly. Contrast with Qwen3 4B which benefits from extended training (100% at iter 1100+).

87. **80:10:10 stratified split reveals true model accuracy is 79%, not 100%.** Previous "100%" was on 23 cherry-picked examples. With a proper 131-example test set (stratified by category, including hard negatives), Qwen3 4B scores 79% with 0 fails. The 23-example test set was too small to distinguish a 79% model from a 100% model — just luck that all 23 fell in the passing 79%. Category breakdown reveals at-symbol (20%) and multi-step (50%) as the weakest categories, invisible to the old test set. Val loss with 131 examples was monotonically decreasing (no overfitting noise), confirming the old 20-example val set was pure noise.
88. **Step 2000 > step 1000 on v5 test (82% vs 79%) despite worse eval_loss (0.164 vs 0.115).** 6th confirmation that eval_loss minimum is harmful for checkpoint selection on this task. Extended training improves at-symbol (20%→60%), self-correction (74%→89%), emoji (70%→80%). More epochs = better pattern generalization even when cross-entropy overfits.
89. **v3 prompt (191 tok) is a net wash — helps multi (+40 pts) but kills hard-neg (-21) and emphasis (-25).** The disfluency rule ("sorry", "scratch that", "actually": drop the wrong part) causes the model to strip genuine discourse markers ("I mean,", "Actually,") from hard-negative inputs. The emphasis rule ("Emphasis/bold: ALL CAPS") conflicts with existing emphasis training. Multi-step rule is the only clear win.
90. **Inference-time prompt engineering is a dead end for fine-tuned models.** Tested v4 prompt (v2 + quote/at-symbol/multi rules, 121 tok): 79%. v4 without at-symbol rule: 76%. Every additional rule creates cross-category interference — quote rule bleeds into emphasis (wraps bold words in quotes), multi rule makes self-correction too aggressive, at-symbol rule inserts @ into spell/camelCase outputs. The model was trained with v2 and performs best with v2. Prompt modifications at inference time cannot improve accuracy; they just shift errors between categories. **v2 prompt (83 tok) + step 2000 = 82% is the optimal config. Next gains require more training data, not prompt changes.**
91. **Training with richer prompt (v4) HURTS accuracy.** v4 train + v4 inference = 79% (-3 pts vs v2 baseline 82%). v4 train + v2 inference = 73% (-9 pts). The extra rules during training create prompt dependency — the model learns to rely on explicit rules being present, so removing them at inference causes failures (quotes output as literal "quote-unquote" text, 4 hard fails vs 1). The data already encodes correct behavior through input→output pairs; explicit rules add noise, not signal. **"Train lean, deploy lean" beats "train rich, deploy lean" for copy-heavy editing tasks.**
92. **Qwen3-8B (2x params) scores 10 pts WORSE than Qwen3-4B on copy-heavy editing.** 8B: 72-73% with 8 hard fails. 4B: 82% with 0 fails. All 8 fails are in the quote category — the 8B model outputs literal "quote-unquote" text instead of wrapping in quotes. Larger models have stronger base priors about what constitutes "natural" text, making them LESS willing to apply mechanical transformations. Consistent with DRES finding that reasoning models over-delete (semantic abstraction bias). **For conservative copy-edit tasks, moderate capacity (4B) > large capacity (8B). The 4B model has just enough capacity to learn editing rules without overthinking them.**
94. **Temperature has zero effect on Spoke accuracy — both zero-shot and fine-tuned.** Zero-shot Qwen3.5-4B: 37% at temp=0.0/0.2/0.6 (v5-131). Fine-tuned Qwen3-4B T4-v5split: 82% at both temp=0.0 and temp=0.6 (v5-131). The model has learned a deterministic editing policy — sampling randomness can't improve a task that's fundamentally "copy input with surgical edits." Greedy decoding is optimal. T4-v5split at temp=0.6 also scores **74% on broad58 — NEW ALL-TIME BEST** (+5 over Qwen3.5's 71%, +5 over T3-v5's 69%). The 80:10:10 split (1046 train, 131 val) produced better generalization than full-data training (1287 train, 20 val), likely due to meaningful validation-based checkpoint selection.
96. **Chat/reasoning pedigree does NOT transfer to copy-heavy editing.** Nanbeige4.1-3B beats Qwen3-4B-2507 on Arena-Hard-v2 and Multi-Challenge, yet lands at 70% core23 / 40% broad58 / 66% v5-131 — 13-21 pts below plain Llama 3.2 3B on core23 and 32 pts below the Qwen3-4B champion on v5-131. Extends finding #30 (IFBench ≠ meta-linguistic instruction following) to alignment/reasoning benchmarks generally: Arena-style wins measure eloquent generation, Spoke needs disciplined copying. Zero hard fails though — the model is well-behaved, just imprecise.
97. **Nemotron-3-Nano-4B (hybrid Mamba-2) has no working LR on the standard recipes — dual failure modes at both poles.** lr=1e-5 (champion recipe): model echoes input verbatim, ZERO commands executed — 22% core23 with 12 hard fails (every other fine-tuned model: 0-1). The copy prior is the trivial loss minimizer and 1e-5 never escapes it on Mamba mixer weights. lr=2e-4 (Gemma 3n recipe): edits execute correctly but generation degenerates into multilingual repetition loops ("oorspronkelijke tekst:") — 0% with all-partial scores. The viable window, if any, is between (2e-5–5e-5, untested). Ops tax was severe: ~10 failed launches (mamba-ssm/causal-conv1d kernel wheels are ABI-broken for torch 2.6 conda AND pip; source builds hit missing sdist files + triton>=3.5 requirement; NVIDIA's trust_remote_code needs transformers 4.x while the pipeline runs 5.3; generation_config ships invalid top_p; tokenizer artifacts aren't 5.x→4.x portable). Required a dedicated torch-2.9 image (`SPOKE_MAMBA_IMAGE=1`). Between the ops cost and the dual failure, PARKED — plain-transformer models cost zero setup and score better.
95. **MiniCPM5-1B (July 2026, 1B-class SOTA) still hits the 1B precision ceiling: 43% core23 / 24% broad58 / 38% v5-131 on the champion recipe.** Standard Llama architecture, trained with zero pipeline changes (lr=1e-5, r=8, 2000 steps, v5-split). Memorized training data (train loss 0.0027) but generalizes as a paraphraser, not a copy-editor: 76/131 partials vs only 5 hard fails. Typical misses: lowercased camelCase ("navigationbar"), emoji described AND rendered ("Broken heart 💔"), dropped filler words that should be preserved, comma insertions. Being a newer/stronger 1B did not overcome the capacity floor — LFM2.5-1.2B (70% core23 on v3 data) remains the small-tier reference. Consistent with finding #47 (3B ceiling) extended downward: sub-2B models can't hold surgical copy-edit precision on this recipe/data.
93. **2000 steps is the sweet spot for 1046 v2-trained examples — more training hurts.** Inverted-U accuracy curve: step 1000 = 79%, step 2000 = 82% (peak), step 3000 = 81%, step 4000 = 80%. Training past ~7.7 epochs causes over-memorization that degrades generalization monotonically. Zero hard fails at all checkpoints — the model doesn't hallucinate, it just under-executes on more examples. Interestingly, v4-trained models DO benefit from extended training (79% → 82% from 2k → 4k), possibly because the extra prompt rules act as a regularizer. But v4 only recovers to v2's 2k-step level — never exceeds it. **The accuracy ceiling is a data problem, not a training duration problem.** Next gains require v6 data targeting weak categories (at-symbol 60%, multi 30%, spell 67%).

### Model Comparison (Phase B Summary)

| Model | Params | Zero-shot | Fine-tuned | Latency | Peak Train Mem | Speed vs Qwen3 |
|-------|--------|-----------|------------|---------|----------------|-----------------|
| **Qwen3-4B (T2-v4)** | **4B** | **35%** | **100%** | **1.82s** | **18.6 GB** | **1.0x (reference)** |
| Qwen3-4B (T11-ext) | 4B | 35% | 91% | 3.15s | 18.6 GB | — |
| Llama 3.2 3B (T2) | 3B | 26% | 91% | 1.90s | 15.2 GB | 1.0x same |
| Llama 3.2 3B (T1) | 3B | 26% | 87% | 1.60s | 15.2 GB | 1.1x faster |
| Gemma 3 4B (T1) | 4.6B | 9% | 87% | 2.52s | 11.6 GB* | 0.7x slower |
| LFM2-2.6B (T1b) | 2.6B | 9% | 83% | 1.66s | 13.3 GB | 1.1x faster |
| Llama 3.2 3B (cloud, v5) | 3B | 26% | 83% | — | — | Cloud Modal HF |
| Gemma 3n E2B (cloud, v5) | 4.5B (2B eff) | — | 70% | — | — | Cloud Modal HF, lr too low |
| **Gemma 3n E2B v2 (cloud, v4)** | **4.5B (2B eff)** | **—** | **91%** | **0.51s** | **—** | **Cloud Modal HF, Google hyperparams. 59% broad.** |
| **Gemma 3n E4B (cloud, v4)** | **8B (4B eff)** | **—** | **96%** | **1.20s** | **—** | **Cloud Modal HF, Google hyperparams. 59% broad. Matches Qwen3 DWQ on core23.** |
| Llama 3.2 3B (v4-v2, lr=2e-4) | 3B | 26% | 78% | 0.14s | — | Cloud Modal HF, "optimal" LR WORSE |
| Qwen3-4B (v4-v2, lr=2e-4) | 4B | 35% | 96% | 0.21s | — | Cloud Modal HF, "optimal" LR -4 pts |
| LFM2.5-1.2B (T1) | 1.2B | 9% | 70% | 0.63s | 6.5 GB | 2.9x faster |
| Flan-T5-large (v1, lr=3e-4) | 783M | 17%* | 70% | 0.22s | — | Cloud Modal HF. Full FT. *13% emoji ceiling. Best with lr=3e-4. |
| Flan-T5-large (v2, lr=1e-5) | 783M | 17%* | 30% | 0.22s | — | Cloud Modal HF. Full FT. lr too low → underfitting. |
| Flan-T5-base (v1, lr=3e-4) | 248M | 17%* | 57% | 0.28s | — | Cloud Modal HF. Full FT. *13% emoji ceiling. |
| Flan-T5-base (v2, lr=1e-5) | 248M | 17%* | 17% | 0.14s | — | Cloud Modal HF. Full FT. lr too low → zero-shot level. |
| **T5Gemma 2 1B-1B (v3)** | **2.1B** | **—** | **70%** | **0.84s** | **—** | **Cloud Modal HF. Full FT. FlanEC recipe + EOS fix. Emoji 3/3 ✓. No vocab ceiling.** |
| **Gemma 3n E4B (v5, 2k steps)** | **8B (4B eff)** | **—** | **83%** | **0.57s** | **—** | **Cloud Modal HF, v5 data, 2000 steps. Broad=64%. Step 2000 = best ckpt. V5 interference: -13 pts core vs v4.** |
| **Qwen3.5-4B (HF, v5, 1500 steps)** | **4B** | **13%** | **96%** | **0.38s** | **—** | **Cloud Modal HF text-only, v5 data, r=8, lr=1e-5. Broad=71% (BEST). 12 target modules.** |
| Qwen3.5-4B (r16 rsLoRA) | 4B | 13% | 91% | — | — | Cloud Modal HF, v5 data, r=16+rsLoRA. Broad=62%. Regression — dead end. |
| Qwen3.5-4B (lr=5e-5) | 4B | 13% | 96% | 0.69s | — | Cloud Modal HF, v5 data, lr=5e-5. Broad=66%. Same core as baseline but -5 broad. |
| Qwen3-4B (EPO w=3.0) | 4B | 35% | 87% | 0.22s | — | Cloud Modal HF, v5 data, EPO loss (edit_weight=3.0). Broad=66%. **REGRESSION: -13 core, -3 broad vs baseline.** Over-editing failure mode. Dead end. |
| MiniCPM5-1B (T1, v5-split) | 1.1B | — | 43% | 0.23s | — | Cloud Modal HF, v5-split, champion recipe. Broad=24%, v5-131=38%. Partial-heavy: learns task shape, lacks copy precision. Below LFM2.5-1.2B (70% core). |
| Nanbeige4.1-3B (T1, v5-split) | 3B | — | 70% | — | — | Cloud Modal HF, v5-split, champion recipe. Broad=40%, v5-131=66%. 0 hard fails. Reasoning pedigree (Arena-Hard > Qwen3-4B) doesn't transfer; below plain Llama 3.2 3B. |
| Nemotron-3-Nano-4B (lr=1e-5) | 4B (Mamba-2 hybrid) | — | 22% | 6.65s | — | Cloud Modal HF (torch-2.9 mamba image), v5-split. Broad=26%, v5-131=31%. Echoes input verbatim — 12 hard fails. Finding #97. |
| Nemotron-3-Nano-4B (lr=2e-4) | 4B (Mamba-2 hybrid) | — | — | — | — | Gemma-recipe probe. 0% broad/v5: edits correct but repetition collapse. Parked. Finding #97. |

*Gemma 3 4B: 18.9 GB without grad_checkpoint (OOM), 11.6 GB with grad_checkpoint enabled.
*T5 models: zero-shot uses T5 prefix format, not v2 prompt. Flan-T5 accuracy ceiling is 87% due to vocab limitation (can't generate emoji). T5Gemma 2 has no such ceiling (256K Gemma vocab).
