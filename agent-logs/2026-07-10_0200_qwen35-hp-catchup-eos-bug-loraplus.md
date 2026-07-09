# Qwen3.5-4B Hyperparameter Catch-Up: a Real Merge Bug Fixed, LoRA+ Closes Most of the Gemma Gap

**Date:** 2026-07-09 evening → 2026-07-10 (session spanned overnight)
**Agent:** Claude Sonnet 5, with 3 parallel Sonnet research agents + a Fable second-opinion consult
**Status:** Completed — bug found and fixed in 3 files, seed-noise band measured, LoRA+ arm run and confirmed as new Qwen3.5-4B best. Combined arm (LoRA+ + dropout=0 + alpha=4) also completed (3 launch attempts, 2 killed by local-network connection drops) — result: ties LoRA+ on core/broad, drops v5-131 by 8 pts. Plain LoRA+ remains the recipe; investigation closed.

## User Intention
User noticed Gemma 4 E4B got a full research-backed hyperparameter treatment (finding #107 — 3 research agents, a 6-arm matrix, 3-seed paired confirm) after asking "did we tune HPs for Gemma?", but Qwen3.5-4B never got the same treatment — its best number (96% core23 / 71% broad58) was a single seed, single recipe from March. Asked for the same rigor: research the model-specific literature, design an experiment matrix, run it, and update the ledger with everything. Along the way, explicitly asked for a Fable second-opinion consult before executing (twice — once on the experiment plan, once on a proposed bug fix), and was firmly cost-conscious about seed-confirmation runs and matrix size ("do you think money grows on trees").

## What We Accomplished
- **3 parallel research agents**: official Qwen/Alibaba guidance (found ms-swift documents LoRA lr=1e-4 vs full-FT lr=1e-5 — our recipe used the full-FT number), community practices (Unsloth/Axolotl — thin coverage, inconsistent DeltaNet module naming across sources), and linear-attention/DeltaNet LoRA literature (a real ICML 2025 paper showing LoRA works on linear projections but fails on SSM/recurrent-state matrices — validated our target-module choice as architecturally correct).
- **Fable consult #1** (on the experiment matrix plan): caught that "single recipe, no HP search" was wrong (two prior deviations existed, both regressions — rsLoRA r=16, lr=5e-5), fixed the plan (gate on seed noise first, run LoRA+ at the same lr=1e-5 base proven for Gemma rather than chaining it to a possibly-noisy LR winner, use paired per-example comparison, benchmark a checkpoint grid not one fixed vintage), and verified the DeltaNet module naming directly against the installed transformers source (no risk there).
- **Seed-noise gating** (seeds 43, 44 of the existing recipe) hit a real, serious bug: benchmarks came back at 0% with every example showing the correct answer followed by runaway hallucinated conversation turns.
- **Root-caused and fixed the bug** (3 files: `merge_adapter_checkpoint.py`, `train_hf.py`, `benchmark.py`) — see finding #109. A second independent investigation (a delegated background agent) found this was a recurrence of a documented 2026-03-06 incident, only ever patched for one of two export paths.
- **Re-ran the gate post-fix**: 96% core23 (flat across 3 seeds), 64-71% broad58 (7-pt noise band) — confirms a real, wide gap vs Gemma's 82.8% broad58.
- **Ran arm E (LoRA+ optimizer, same lr=1e-5 base as Gemma)**: 100% core23 / 76% broad58 / 88% v5-131 — clears the noise band on both metrics, closes most of the gap to Gemma in one run.
- **Fable consult #2** (on the eos_token_id bug fix): confirmed the diagnosis, validated the fix is safe across the model roster this script serves, flagged an open loose end (the March "good" artifact's correct eos list has no traceable code path — possibly a hand-patch, meaning `train_hf.py`'s own export may have carried the same latent bug), and recommended defense-in-depth + a runaway-generation alarm — both implemented.
- **User declined further seed-confirmation and the remaining individual HP arms** (A: lr=1e-4, C: dropout=0, D: alpha=4) on cost grounds; instead requested one combined run (LoRA+ + C + D). Flagged and avoided a concrete risk: stacking arm A's lr=1e-4 bump on top of LoRA+ would have pushed the LoRA+ B-matrix's effective LR to 1.6e-3 — close to the exact value (3.2e-3, from a 2e-4 base) that diverged in Gemma's own matrix (finding #107) — so A was dropped from the combo, keeping just LoRA+ + dropout=0 + alpha=4.
- **LEDGER updated**: header, two new findings (#109 bug, #110 result), 4 new training-run table rows, 1 new Model Comparison row.

## The bug (finding #109, full detail in LEDGER)
`merge_adapter_checkpoint.py`'s generation-config fallback (used whenever training runs with `--no-export-merged`) silently wrote nothing when the base model has no upstream `generation_config.json` (true for `Qwen/Qwen3.5-4B` — confirmed 404 on the hub). `benchmark.py` then derived a default eos_token_id straight from the merged `config.json`'s field — which for this model is 248044 (`<|endoftext|>`, the pad token), not 248046 (`<|im_end|>`, the real chat turn-end token). Generation never hit its stop condition; every example produced the correct answer then ran away hallucinating fake conversation turns for the rest of the 256-token budget (visible as literal `!` filler — token id 0 in this vocab), scoring 0% instead of 96%.

This is a **recurrence** of a documented 2026-03-06 incident (see `agent-logs/2026-03-06_0403_qwen35-checkpoint-benchmark-diagnosis.md`, fixed via commit `ac5b340`) — but that fix only patched the copy-from-final-export happy path. The standalone merge script's actual fallback branch was never exercised again until this session, so the underlying defect survived untouched for four months.

**Fix**: in all 3 scripts, always union the tokenizer's own `eos_token_id` into whatever generation config gets saved/used, rather than trusting `config.json` or `GenerationConfig.from_pretrained()` alone. Added a runaway-generation alarm to `benchmark.py` (warns if >20% of examples hit the token budget without stopping) so this reads as one log line next time, not a burned benchmark cycle.

**Open loose end** (Fable flagged, unresolved): nobody can find the code path that produced the March champion's correct eos list — it may have been hand-patched on the Modal volume rather than fixed at the source, meaning `train_hf.py`'s own export path might have carried the same latent bug on its next real run. The fix was applied there too as a precaution, but this wasn't directly re-verified end to end.

## Results

| | core23 | broad58 | v5-131 |
|---|---|---|---|
| Qwen3.5-4B, existing recipe, seed 42 (March) | 96% | 71% | — |
| Qwen3.5-4B, existing recipe, seed 43 | 96% | 64% | — |
| Qwen3.5-4B, existing recipe, seed 44 | 96% | 67% | — |
| **Qwen3.5-4B + LoRA+ (seed 42)** | **100%** | **76%** | **88%** |
| Gemma 4 E4B champion (for reference) | 95.7% | 82.8% | 88.5% |

## Key Learnings
- **A merge-time bug can look exactly like a training catastrophe.** The tell that separated them: the model produced the *exact correct answer* before running away — weights were fine, only the stop condition was wrong. Always check the raw (untruncated) output before trusting a 0% score.
- **"Fixed once" isn't "fixed"** — a bug patched on one code path (train_hf.py's built-in export) can silently persist on a sibling path (the standalone merge script) that just wasn't exercised again for months. The fix this time was applied to all producer AND consumer sites, plus a loud alarm, specifically so a third occurrence costs one log line instead of a benchmark cycle.
- **LoRA+ transfers across architectures.** It won for Gemma (PLE) and now for Qwen3.5 (DeltaNet-hybrid) at the same base lr=1e-5 — a second, structurally different data point that this is a robust lever, not an artifact of one model family.
- **Don't stack a proven optimizer trick with an independently-plausible-but-untested LR bump without checking the interaction.** LoRA+'s B-matrix LR is a 16x multiplier of the base — bumping the base independently (as arm A would have) risks reproducing a divergence already observed once in this exact project (Gemma's matrix).
- **User cost-consciousness is a legitimate steering input, not corner-cutting to push back on.** Declining further seed-confirmation after one clean signal, and combining remaining arms into one run, was the right call given the result already cleared the noise band by a comfortable margin.

## Context for Future
- Combined arm result is in: `spoke-qwen35-4b-hp-combo-loraplus-dropout0-a4-v2` (seed=42) scored 100/76/80 — ties plain LoRA+ on core23/broad58 but drops v5-131 by 8 points. Stacking more changes on an already-tuned recipe didn't help. **No further Qwen3.5-4B HP arms are planned** — plain LoRA+ (100/76/88) is the recipe going forward.
- The combo run needed 3 launch attempts (2 died to local-network `ConnectionError`/`StreamTerminatedError` mid-training, even with `modal run --detach`) — this is a recurring infra flakiness pattern, not a one-off, and `--detach` did not fully protect against it. Worth a closer look at local network/wifi stability if this keeps happening.
- Remaining ~7-pt broad58 gap to Gemma looks architectural (PLE vs DeltaNet) rather than tunable — the ledger's sober framing from Gemma's own matrix (HP tuning buys ~+1 example once you have one good lever; the model swap was the big jump) applies here too.
- Audit flag from finding #109: any other run merged via `--no-export-merged` against a base model lacking a hub `generation_config.json`, from before this fix, should not be trusted without re-checking its eos_token_id.
- No git commits were made this session — LEDGER.md and the 3 code files (`merge_adapter_checkpoint.py`, `train_hf.py`, `benchmark.py`) are modified but uncommitted as of this log.
