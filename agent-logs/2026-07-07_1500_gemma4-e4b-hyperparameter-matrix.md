# Gemma 4 E4B Hyperparameter Matrix: Research-Informed Sweep + Seed-Confirm

**Date:** 2026-07-07 (afternoon; follows the morning's champion session)
**Agent:** Claude Opus 4.8 (1M context) → handed to Claude Fable 5 mid-session (user switched models)
**Status:** Completed — 6-arm matrix + 3-seed LoRA+ confirm done, best checkpoint has full 3-set line, all committed

## User Intention
After crowning Gemma 4 E4B the accuracy champion (finding #106), the user asked: "did we even explore hyperparameters? maybe we should've done research into how to fine-tune that specific model because it is different from Qwen." Directive: research what Google/DeepMind recommend + what's popular in 2026 for tasks like this (check Hugging Face), then experiment with LR, the learning algorithm, etc. Explicitly OK'd spinning up Sonnet 5 sub-agents to parallelize research while managing them. Later: "compare with a Qwen recipe," then "what's our best checkpoint now" + compare to previous bests.

## What We Accomplished
- **3 parallel Sonnet research agents** (Google/DeepMind official, 2026 LoRA SOTA, HF ecosystem) → synthesized into an evidence-ranked experiment matrix.
- **Enabled two new levers** in `train_hf.py`: `--seed` (was hardcoded 42) + `optimizer=loraplus` (peft `create_loraplus_optimizer`).
- **Ran a 6-arm matrix** (1200 steps each, all benchmarked at step-900 fixed vintage): 2 seed-repeats + α=32 + lr=1e-4 + dropout=0 + LoRA+.
- **3-seed confirm of both** the Gemma champion margin over Qwen AND the LoRA+ front-runner.
- **Completed the best checkpoint's 3-set line** (added the missing core23).
- Logged finding #107, updated the header, 16 bite-sized commits.

## The research (3 agents, key takeaways)
1. **Our recipe IS Google's official Gemma 4 QLoRA recipe** — lr=2e-4, constant_with_warmup, r16/α16, dropout 0.05, seq 512 — almost line-for-line (ai.google.dev QLoRA guide, updated for gemma-4-E2B). The "borrowed from gemma3n" worry dissolved. Unsloth/Axolotl agree (2e-4 standard).
2. **Most LoRA-variant "wins" are under-tuned baselines** (ICML 2026 "Vanilla LoRA May Suffice"; Thinking Machines "LoRA Without Regret"). Expected effect sizes ±1-2% = OUR noise floor → single-arm A/Bs can't separate signal; must seed-confirm.
3. **Skip-list confirmed by our own scars**: DoRA (fabricates on small data), rsLoRA (only helps r≥64), NEFTune (stale evidence), Muon (our Muon-YOLO was already a dead end). **LoRA+** (arXiv 2402.12354) = the one Tier-1-evidence accuracy lever.
4. **Bug checklist — all clear on our run**: `Gemma4ClippableLinear` PEFT crash (our regex sidesteps), `mm_token_type_ids` req, `use_cache=False` KV-corruption (fixed in transformers 5.5.2 which we pinned), Liger crashes (unused), ZeRO-3 (single-GPU). Google's own Vertex script freezes everything not under `model.language_model.*` — identical to our regex.

## The matrix (broad58 / v5-131, all @ step-900, 0 fails unless noted)

| arm | change | broad58 | v5-131 | verdict |
|---|---|---|---|---|
| champion (s42, 2000-step ckpt800) | — | 79.3 | 86.3 | reference |
| R1 seed 43 | seed | 79.3 | 84.0 | seed noise |
| R2 seed 44 | seed | 74.1 | 84.7 | seed noise |
| A: α=32 | α=2r | 79.3 | 83.2 (2 fails) | ✗ no gain, worse eval floor |
| B: lr=1e-4 | half LR | 79.3 | 87.8 | helps v5 |
| C: dropout=0 | no dropout | 82.8 | 84.0 | helps broad |
| **D: LoRA+ base 1e-5** | optim=loraplus | **82.8** | **88.5** | ✅ standout (single seed) |
| D-fail: LoRA+ base 2e-4 | optim=loraplus | DIVERGED | (eval 0.15→0.34) | ✗ lr_B=3.2e-3, 16× champion |

## LoRA+ (what it is + the LR rule)
LoRA learns a patch `W + B·A`; A starts random, B starts at **zero**. Standard LoRA uses one LR for both. LoRA+ gives **B a 16× higher LR** (theory: with a shared LR one matrix is always under-trained). Zero extra params, zero inference cost. **The rule we learned the hard way:** the 16× multiplier applies to the base LR (= lr_A), so center it by where you want lr_B to land, NOT lr_A. base 2e-4 → lr_B=3.2e-3 = divergence (D-fail). base 1e-5 → lr_B=1.6e-4 (proven-good) → the winning arm.

## Seed-confirm verdicts (the real payoff)
**Gemma champion margin over Qwen is REAL.** 3 plain seeds: broad 74-79 (wide!), v5 84-86. 3-seed mean 77.6/85.0 still beats Qwen (74.1/81.7).

**LoRA+ is a SMALL but CONSISTENT real edge — the single-seed standout was noise-inflated.** 3 LoRA+ seeds: s42 82.8/88.5, s43 79.3/84.7, s44 79.3/87.0. Mean 80.5/86.7 vs plain 77.6/85.0 (+2.9 broad / +1.7 v5). **THE METHOD LESSON:** seed 43 alone fell INSIDE the plain band → looked like "no effect" on an unpaired read. But because plain and LoRA+ share seeds 42/43/44, the correct comparison is **PAIRED** (seed-matched), and there LoRA+ ≥ plain on EVERY seed×set cell (5/5 non-negative). LoRA+ never LOST. Comparing to a noise BAND hides small paired effects; seed-matched treatment/control surfaces them.

## Best checkpoint now (full 3-set line)
**LoRA+ base lr=1e-5, seed 42** (`spoke-g4e4b-hp-lplus2-ckpt900`): **95.7 core / 82.8 broad / 88.5 v5, 0 fails on all 231.** Top score on all 3 sets. Core23 (added last) ties the Gemma champion — LoRA+'s broad/v5 gains cost nothing on the easy set.

| checkpoint | core | broad | v5 | note |
|---|---|---|---|---|
| **best (LoRA+ s42)** | 95.7 | 82.8 | 88.5 | lucky seed; recipe mean 80.5/86.7 |
| Gemma champion (2000-step) | 95.7 | 79.3 | 86.3 | prev accuracy best |
| Qwen champion | 100 | 74.1 | 81.7 | prev deployable best |

## The sober framing
**HP tuning barely moved the champion (~+1 ex over the 2000-step Gemma champion) — because our recipe was already Google's official one.** The real jump (+5-8 ex) was the MODEL SWAP (Qwen→Gemma 4). Durable lesson: **model family moves the needle at *set* scale, knob-tuning at *example* scale.** LoRA+ is promotable ONLY because it's free (identical merged weights).

## Ops / incidents
- **Account switch mid-session**: user's `sandy-36852` ran out of credits ($1 left) → killed the in-flight run before it drained, switched to `lolrazh` ($30). Volumes/secrets are per-workspace but `lolrazh` already had them from March (verified train.jsonl byte-identical, sha match).
- **Card-tier discovery**: `lolrazh`'s $30 credits do NOT cover L40S/A100 without a card on file — a GPU probe (T4/L4/A10G ✅, A100/L40S ❌) diagnosed it. User added a card; runs proceeded. Cost ~4 bounced launches.
- **Flaky-network deaths (finding #106 gremlin, twice)**: arm D + arm C's bench chain died mid-run (grpclib StreamTerminated / Errno 8). Both recovered on relaunch; arm C's training had completed so its ckpt survived.
- **Checkpoint rotation**: `save_total_limit=5` rotates away mid-run checkpoints between eval-best and step-900. Step-900 is the earliest that always survives → adopted as the fixed benchmark vintage.

## Protocol fix worth keeping
**Benchmark at fixed training vintage (step 900), NOT the eval-loss minimum.** Seed 43's eval-min landed at 0.77 epochs and benchmarked 8 pts low on v5; re-merged at step 900 (~3 epochs, champion vintage) it matched. Eval-loss TIMING is itself seed noise — compare arms at matched vintage.

## Next steps
- **Untested COMBINATION**: LoRA+ base 1e-5 + dropout=0 (stack the two independent nudges). Do NOT add lr=1e-4 on top — re-overheats lr_B.
- **THE deployment gate (highest value)**: validate gemma4 merged HF → MLX 4-bit conversion + measure real M4 local latency. No amount of +2-ex tuning matters if it can't ship.
- v6 data on this base.

## Commits (bite-sized, chronological)
1. `3627304` train_hf.py: --seed param + LoRA+ optimizer option
2. `e11c4a3` seed-43 arm results (R1, merged eval-min)
3. `b0d6d07` seed-43 step-900 (vintage explains the gap)
4. `4b1d2d2` seed-44 arm merged results (R2)
5. `6519234` seed-44 step-900 (3-seed confirm complete)
6. `d76e780` arm A (α=32): no gain
7. `cc94424` arm B (lr=1e-4): beats seed band on v5
8. `3ac83d3` arm C (dropout=0): tops broad58
9. `fd97618` arm D' (LoRA+ 1e-5): standout single-seed
10. `1eec18f` LEDGER finding #107 + header
11. `edd5426` LoRA+ seed-43 confirm (falls in band)
12. `025376e` LoRA+ seed-44 confirm (paired edge)
13. `2eb5351` LEDGER #107 seed-confirm verdict
14. `5b86388` best-ckpt core23 (full 3-set line)
15. `eda7eff` header: best checkpoint = LoRA+ s42
16. This log
