# Project Restart: Housekeeping + MiniCPM5-1B Bake-off + Qwen3.5-2B Attempt

**Date:** 2026-07-04
**Agent:** Claude Fable 5
**Status:** Completed (Qwen3.5-2B training incomplete — cancelled externally at step 867)

## User Intention
Return to the project after a ~4-month break. Get up to speed via agent logs + ledger, do housekeeping in bite-sized commits, then start a fresh model bake-off with MiniCPM5-1B and Qwen3.5-2B (Gemma 4 deferred). Clarified that there is no v6 data — the last dataset is the v5 stratified split (1,046/131/131) that trained the champion.

## What We Accomplished
- **Read-back:** full ledger + all 49 agent logs (via subagent). Champion confirmed: Qwen3-4B T4-v5split ckpt2000 = 100% core23 / 74% broad58 / 82% v5-131.
- **Housekeeping (5 commits):** committed the untracked June 25 ASR frontend scouting work (Whisper turbo 4-bit control, Granite Speech 4.0/4.1 + full-quant converters, Cohere Transcribe custom 4-bit) + backfilled its agent log; retired stale `NEXT_RUNS.md` (both paths long complete), preserving its data-augmentation ideas as the ledger's **v6-data** queue entry.
- **Bug fix:** the best-broad-ever Qwen3.5-4B run (96%/71%, finding #81) used 12 LoRA target modules incl. DeltaNet projections, but that edit was never committed — `train_hf.py` had only the 7 standard ones. Restored as a `qwen3_5`-gated conditional (`in_proj_qkv/z/a/b`, `out_proj`).
- **Modal workspace migration:** the old `lolrazh` workspace refuses L40S ("add a payment method"); credits live on the newer **`sandy-36852`** workspace (the active profile). Recreated `hf-secret`/`wandb-secret` from local credentials, re-uploaded v5-split + bench sets (verified md5-identical to the volume copy the champion trained on).
- **MiniCPM5-1B full run** (`spoke-minicpm5-1b-v5split-2k-20260704`, wandb tpw3vwnt): champion recipe, 2000 steps. Benchmarked all three suites.
- **Qwen3.5-2B run** (`spoke-qwen35-2b-v5split-2k-20260704`, wandb sg9bpwxr): healthy (loss 0.054 at step 867, eval 0.14) but **cancelled externally at step 867/2000**. Checkpoints 400–800 survive on the volume. (Next-day diagnosis: flaky local network kills attached modal runs — this was almost certainly that.)

## Results

| Model | core23 | broad58 | v5-131 | Notes |
|-------|--------|---------|--------|-------|
| MiniCPM5-1B (champion recipe) | **43%** | **24%** | **38%** | 76/131 partials, only 5 hard fails. Train loss 0.0027 (memorized). |

Failure texture: lowercased camelCase ("navigationbar"), emoji described AND rendered ("Broken heart 💔"), dropped filler words, comma insertions. Learns task shape, lacks copy precision. **Finding #95** — 1B precision ceiling holds even for a 2026 1B-class SOTA; LFM2.5-1.2B (70% core23) remains the small-tier reference.

## Technical Implementation
- MiniCPM5-1B is `model_type: llama` / `LlamaForCausalLM` — zero pipeline changes; existing `enable_thinking=False` probe handles its hybrid thinking mode.
- Files: `spoke/cloud/train_hf.py` (DeltaNet targets), `spoke/LEDGER.md` (MiniCPM5 rows, finding #95, header), `spoke/bench/result_spoke-minicpm5-1b-*` (3 JSONs), `agent-logs/2026-06-25_1300_asr-frontend-scouting.md` (backfill).

## Key Learnings
- **Silent partial-LoRA is the hybrid-architecture footgun**: PEFT matches target modules by name suffix; DeltaNet layers were silently skipped for months in the committed script.
- Modal state (volumes/secrets) is per-workspace; migration = recreate secrets + re-upload data (~10 min).
- **Don't pipe modal run through `tail`** — the output file stays empty until process exit, blinding monitors.

## Context for Future
Qwen3.5-2B never got a full run (only smoke 22%); its checkpoints 400–800 are on the sandy volume if resumption is ever wired up. Gemma 4 has no 1B — smallest are E2B/E4B, successors to Gemma 3n. Modal credits on sandy-36852.
