# Broad Eval Benchmark + System Prompt Research

**Date:** 2026-03-05
**Agent:** Claude Opus 4.6
**Status:** ✅ Completed
**Building on:** `2026-03-04_2130_inference-audit-1.7b-launch.md`

## User Intention
User wanted to stress-test the deploy model (DWQ 4-bit Qwen3-4B) on a broader, unseen eval set ("Spoke - Evals.csv", 58 examples) to understand real-world accuracy beyond the 23-example v3 test set. After discovering the model scored 67% (vs 96% on v3), the investigation pivoted to root-cause analysis of failures and whether the system prompt (v2 vs full Spoke prompt) was the bottleneck. This led to deep research on how system prompts interact with fine-tuning.

## What We Accomplished

- ✅ **Qwen3-1.7B-T1 training completed** — 2000 iters, v4 data, T2-v4 parity config. 74% bf16 accuracy, 0.83s latency. Spell-replace is capacity bottleneck (0/3 exact).
- ✅ **Broad eval benchmark (58 unseen examples)** — Converted "Spoke - Evals.csv" to JSON test set with auto-categorization. Zero overlap with training data confirmed.
- ✅ **DWQ vs BF16 comparison on broad evals** — DWQ=67%, BF16=66%. Statistically identical. Failures are a data problem, not quantization.
- ✅ **Comprehensive failure root-cause analysis** — 19 failures categorized into 7 root causes. 9-12 of 16 real failures are trainable with ~60-80 targeted examples.
- ✅ **Prompt × Model matrix experiment** — Tested 4 combinations: fine-tuned DWQ + v2 (67%), fine-tuned DWQ + Spoke (67%), base + Spoke (36%), fine-tuned BF16 + v2 (66%). System prompt has zero effect on fine-tuned model.
- ✅ **System prompt + fine-tuning research** — Found PromptIntern (Microsoft), Generative Prompt Internalization, OpenAI best practices. Synthesized into actionable options.
- ✅ **Memory updated** — All findings, eval results, prompt research, and Qwen3-1.7B results saved to MEMORY.md.

## Technical Implementation

### Eval Pipeline
- `spoke/bench/csv_to_testset.py` — Converts CSV (input,ideal) to JSON with heuristic auto-categorization (spell-replace, self-correction, multi, passthrough, etc.). Deduplicates by input text.
- Added `spoke-full` prompt mode to `run_benchmark.py` — full Spoke prompt (1,042 tokens) with detailed rules and 7 few-shot examples.
- Verified 0/58 eval examples appear in training data (true OOD test).

### Prompt × Model Matrix
| | V2 (83 tok) | Spoke (1,042 tok) |
|---|---|---|
| Fine-tuned DWQ | 67% / 0.89s | 67% / 6.29s |
| Fine-tuned BF16 | 66% / 2.57s | — |
| Base Qwen3-4B | 35% (LEDGER) | 36% / 6.55s |

### Failure Root Causes (DWQ, 19 failures)
| Root Cause | Count | Trainable? |
|---|---|---|
| Not real failures (eval expects impossible) | 3 | Remove from eval |
| Multi-step chaining (2-4 ops) | 6 | Yes — need diverse 3+ op combos |
| Spell format variants | 2 | Yes — "It's X by the way" format |
| Punctuation precision | 3 | Partially |
| Instruction edge cases (meta-language, tempting questions) | 2 | Partially |
| Word dropping | 1 | Yes |
| Long output truncation | 1 | Partially |

**Files Modified:**
- `spoke/bench/run_benchmark.py` — Added `SPOKE_FULL_PROMPT` constant, `spoke-full` prompt mode, choice in argparse
- `spoke/bench/csv_to_testset.py` — New file: CSV-to-JSON converter with auto-categorization
- `spoke/bench/test_set_evals.json` — New file: 58 deduplicated eval examples from Spoke Evals CSV
- `spoke/bench/result_qwen3-t2-v4-dwq4_v2.json` — DWQ results on broad eval (67%)
- `spoke/bench/result_qwen3-4b-bf16+lora_v2.json` — BF16+LoRA results on broad eval (66%)
- `spoke/bench/result_qwen3-t2-v4-dwq4_spoke-full.json` — DWQ + Spoke prompt results (67%)
- `spoke/bench/result_qwen3-4b_spoke-full.json` — Base model + Spoke prompt results (36%)
- `spoke/bench/result_qwen3-1.7b-bf16+lora_v2.json` — 1.7B fine-tuned results (74%, v3 test)

## Key Learnings

- **Fine-tuned models ignore system prompt rules at inference.** The v2→Spoke prompt swap changed nothing (67%→67%) but added 5.4s latency. Behavior is weight-encoded, not prompt-driven. The system prompt is training signal, not runtime instruction.
- **DWQ quantization = mild regularization.** DWQ matched or beat bf16 on broad evals (67% vs 66%). Quantization smooths overfitting — T2 overfit val loss from 0.065→0.091, and DWQ's weight rounding may have helped generalization. No accuracy cost for 2.9x speedup.
- **96% on v3 test was inflated.** Broad eval (58 unseen) shows 67%. The v3 test set has zero multi-step examples and zero implicit corrections — categories where the model fails.
- **Multi-step is a data diversity problem, not a data quantity problem.** Training has 107 multi-step examples but only 22 with 3+ operations. The failing combos (spell+caps on same word, correction cascades, 4+ chained ops) are underrepresented.
- **3 of 19 "failures" are actually correct model behavior.** Eval examples #1-3 expect implicit world-knowledge corrections (Celerobad→Silero VAD) without explicit spell commands. The v2 prompt says "every output word must be in the input or produced by an explicit directive" — model correctly refuses to hallucinate.
- **PromptIntern (Microsoft, 2407.02211) enables zero-prompt inference.** 3-stage curriculum: full prompt → progressive shortening → empty prompt. Model internalizes rules into weights. 90%+ token reduction, 4.2x speedup, no accuracy loss. Relevant for sub-100ms latency targets.
- **Qwen3-1.7B-T1 = 74% bf16, 0.83s.** Half the params, 3.2x faster, but 26% accuracy hit. All 3 spell-replace tasks failed — phonetic reasoning requires more capacity. Not viable as deploy model without accepting accuracy tradeoff.

## Architecture Decisions

- **Keep v2 prompt for production** — Spoke prompt adds 5.4s latency for zero accuracy gain on fine-tuned model. The 83-token v2 prompt is correct for inference.
- **Data augmentation > prompt engineering** — The path to fixing 67%→85%+ is targeted training examples (multi-step chains, spell variants, meta-language), not prompt changes.
- **DWQ 4-bit remains deploy model** — 67% broad / 96% v3 / 0.88s / 2.1 GB. Quantization isn't the bottleneck.

## Ready for Next Session

- ✅ **Broad eval pipeline ready** — `test_set_evals.json` (58 ex, 0 train overlap) as permanent broad eval
- ✅ **Failure taxonomy documented** — Root causes mapped to specific training data gaps
- 🔧 **Generate 60-80 targeted training examples** — Multi-step (3+ ops), spell format variants ("It's X by the way"), meta-language traps, tempting questions
- 🔧 **Consider PromptIntern approach** — Could train with Spoke prompt then progressively remove for zero-prompt inference
- 🔧 **MLX pruning library** — User's side project (separate repo, open-source, from first principles). Not started.

## Context for Future
The broad eval revealed the model's real-world accuracy is 67%, not the 96% we saw on v3. The gap is entirely data-driven (multi-step chaining, spell format variants). Three research-backed options exist: (A) add targeted training examples (quickest, keeps 0.9s latency), (B) retrain with detailed Spoke prompt as training signal, (C) PromptIntern curriculum for zero-prompt inference. The user is excited about all three and wants to explore them. The pruning side project is also on the wishlist.
