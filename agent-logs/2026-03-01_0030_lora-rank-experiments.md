# LoRA Rank Experiments & Model Comparison

**Date:** 2026-03-01
**Agent:** Claude Opus 4.6
**Status:** ✅ Completed

## User Intention
User wanted to systematically explore the fine-tuning hyperparameter space to find the optimal deployment configuration for Spoke's ASR post-processing. Key goals: test different LoRA ranks, compare Llama 3.2 1B vs Qwen3-4B, find where overfitting starts for each config, and determine the best quant level. Also wanted research on whether to remove system prompts from training data and how LoRA rank affects quality. Building on `2026-02-28_2330_spoke-experiment-grid.md`.

## What We Accomplished

### Training Runs
- ✅ **Qwen3-4B r=16, 200 iters** (from previous session) - Baseline: 83% spoke, val loss 0.161
- ✅ **Qwen3-4B r=8, 200 iters** - Matches r=16 at 83% spoke with half the params (7.3M vs 14.7M)
- ✅ **Llama 3.2 1B r=16, 200 iters** - 67% spoke, 50% generic (up from 8%/25% zero-shot)
- ✅ **Llama 3.2 1B r=16, 1000 iters** - Overfits at iter 200 (val loss 0.243 best, 0.405 by iter 700). Killed at ~750.
- ✅ **Qwen3-4B r=8, 1000 iters** - Overfitting delayed to iter 500 (val loss 0.146 best). Graceful degradation: 0.219 at iter 1000 vs r=16's 0.284.
- ✅ **r=8 iter400 6-bit benchmarked** - New best deployment config: 83% spoke + 75% generic, 3.1 GB

### Benchmarks
- ✅ **Llama 3.2 1B zero-shot** - 8% generic, 25% spoke, 25% task (hallucinations, refusals, code generation)
- ✅ **Llama 3.2 1B fine-tuned** - 50% generic, 67% spoke (fixed hallucinations entirely)
- ✅ **Qwen r=8 quant sweep** - 4/6/8-bit all benchmarked with generic + spoke prompts
- ✅ **Confirmed latency parity** - r=8 and r=16 have identical inference speed after fusion (thermal throttling caused false alarm)

### Research (Parallel Agents)
- ✅ **Strategy B research (Opus)** - Verdict: Keep system prompt in training. Removing it makes model ignore inference-time prompts, breaking Spoke's dynamic prompt system.
- ✅ **LoRA rank research (Sonnet)** - r=8 is sweet spot for formatting tasks. r=16 is too high. Alpha should scale 2x rank.
- ✅ **Condensed training prompt v2** - Saved to `spoke/bench/TRAINING_PROMPT_V2.md` for future experiment

## Technical Implementation

### Full Comparison Table (All Experiments)

| Model | Rank | Iters | Quant | Size | Generic | Spoke | Latency |
|-------|------|-------|-------|------|---------|-------|---------|
| Qwen3-4B | — | 0 | 4-bit | 2.1 GB | 25% | 58% | 0.69s |
| Qwen3-4B | r=16 | 200 | bf16 | 7.5 GB | 58% | 83% | 1.70s |
| Qwen3-4B | r=16 | 200 | 6-bit | 3.1 GB | 58% | 83% | 0.82s |
| Qwen3-4B | r=16 | 200 | 4-bit | 2.1 GB | 58% | 75% | 0.68s |
| Qwen3-4B | r=8 | 200 | bf16 | 7.5 GB | 58% | 83% | 1.75s |
| Qwen3-4B | r=8 | 200 | 8-bit | 4.0 GB | 58% | 83% | 1.53s* |
| Qwen3-4B | r=8 | 200 | 6-bit | 3.1 GB | 58% | 83% | 0.79s |
| Qwen3-4B | r=8 | 200 | 4-bit | 2.1 GB | 58% | 75% | 0.81s |
| Qwen3-4B | r=16 | 600 | bf16 | 7.5 GB | 75% | 83% | 1.93s |
| Qwen3-4B | r=16 | 600 | 4-bit | 2.1 GB | 67% | 75% | 0.66s |
| Llama 1B | — | 0 | bf16 | 2.4 GB | 8% | 25% | 2.00s |
| Llama 1B | r=16 | 200 | bf16 | 2.4 GB | 50% | 67% | 0.63s |

*\*Thermal throttle inflated; re-measured at 0.79s for 6-bit*

### Val Loss Curves

**Qwen3-4B r=16 (1000 iters, run 1):**
```
Iter:  1    100   200   300   500   600   800   1000
Loss: 3.30  0.20  0.16  0.15  0.15  0.18  0.24  0.28  ← overfits ~500
```

**Qwen3-4B r=8 (1000 iters, complete):**
```
Iter:  1    100   200   300   400   500   600   700   800   900   1000
Loss: 3.30  0.24  0.19  0.17  0.18  0.15  0.15  0.15  0.18  0.19  0.22  ← overfits ~500, graceful
```

**Llama 1B r=16 (1000 iters):**
```
Iter:  1    100   200   300   400   500   600   700
Loss: 3.10  0.31  0.24  0.26  0.30  0.29  0.31  0.41  ← overfits ~200
```

### Key LoRA Research Findings (from Sonnet agent)
- **r=4-8**: Recommended for simple formatting tasks
- **r=8-16**: General instruction following
- **r=32+**: Domain shift / novel knowledge
- **Alpha = 2x rank** is the standard ratio (we follow this)
- **Breadth > depth**: All layers with low rank beats fewer layers with high rank
- **Train loss < 0.2**: Overfitting signal per Unsloth heuristic
- Source: Raschka, Unsloth docs, Lightning AI experiments, QLoRA paper

### Key Strategy B Research Findings (from Opus agent)
- If trained WITHOUT system prompt, model ignores system prompts at inference
- HuggingFace experiment: Llama 3.8B fine-tuned with empty system prompt → ignored all system prompts after
- Universal rule: match training format to inference format
- `mask_prompt: true` already handles this optimally (system prompt = conditioning context, not training target)
- Verdict: **Keep Strategy A** (system prompt in training)

**Files Modified:**
- `spoke/config.yaml` - rank 16→8, iters 200→1000 (for current run)
- `spoke/config-llama.yaml` - Created for Llama 3.2 1B training
- `spoke/bench/run_benchmark.py` - Added Llama 3.2 1B model entries
- `spoke/bench/TRAINING_PROMPT_V2.md` - Condensed prompt proposal for future
- `.gitignore` - Added spoke/model*/ glob pattern
- `spoke/bench/result_*.json` - Multiple benchmark result files
- `spoke/adapters/` - Organized into run1/run2/run3 subdirectories
- `spoke/adapters-llama/` - Llama adapter checkpoints

## Bugs & Issues Encountered
1. **Mac sleep kills training (exit 134/SIGABRT)** - Metal GPU context lost on sleep
   - **Fix:** Always use `caffeinate -i` prefix for training commands
2. **Local model paths treated as HuggingFace repos** - `spoke/model-4bit` tried to fetch from HF
   - **Fix:** Use relative path with `/`: `../../spoke/model-4bit`
3. **Thermal throttling inflates latency** - r=8 benchmarks ran right after training, showed 1.42s instead of 0.79s
   - **Fix:** Re-benchmark on cool chip confirmed identical latency to r=16. Always let GPU cool between heavy operations and benchmarks.
4. **`spoke/model*/` not gitignored** - Original glob `spoke/model/` didn't match `spoke/model-4bit/`
   - **Fix:** Changed to `spoke/model*/` glob pattern

## Key Learnings
- **r=8 matches r=16 at 200 iters** for this simple formatting task. Half the params, same accuracy, more headroom before overfitting. Literature was right.
- **Fused models are architecturally identical regardless of LoRA rank** - inference speed is the same. Only weight values differ.
- **Llama 1B overfits 2.5x faster than Qwen3-4B** (iter 200 vs 500) at the same rank. Smaller model + higher relative adapter capacity (0.91% vs 0.37%) = faster memorization.
- **4-bit quantization is the quality cliff** for Qwen3-4B — drops 8% accuracy regardless of rank or iters. 6-bit preserves full accuracy.
- **Llama 1B fine-tuning completely fixes hallucination/refusal** (8% → 50% generic) but can't match Qwen's 83% spoke accuracy. The 3x parameter advantage is real.
- **Parallel research agents are highly effective** — Opus and Sonnet produced actionable, well-sourced research in ~3-5 min while training ran in background.

## Architecture Decisions
- **r=8 as new default** - Research + experiments confirm r=16 is overkill for formatting tasks. Half the params with same accuracy.
- **Keep system prompt in training** - Removing it would break Spoke's dynamic prompt composition at inference. Strategy A confirmed correct.
- **6-bit as deployment quant level** - Consistent 83% accuracy, 3.1 GB size, ~0.8s latency. 4-bit is 8% accuracy penalty.
- **Llama 1B not viable for deployment** - 16% accuracy gap vs Qwen at similar model sizes (bf16 Llama ≈ 6-bit Qwen in GB).

## Ready for Next Session
- 🔄 **r=8 1000-iter run completing** - Will have checkpoints at 200/400/600/800/1000 with val loss curve
- ✅ **Training prompt v2 ready** at `spoke/bench/TRAINING_PROMPT_V2.md` — swap into JSONL and retrain
- ✅ **All benchmark infrastructure working** - `run_benchmark.py` supports adapters, local models, all prompt modes

## Remaining Task Board
| # | Task | Status |
|---|------|--------|
| 8 | Train without multi-step examples | Pending |
| 11 | Update results.html with full comparison | Pending |
| — | r=8 400-iter sweet spot training | After 1000-iter results |
| — | Training prompt v2 experiment | After rank experiments |
| — | r=4 experiment (if r=8 still overfits) | Pending |

## Context for Future
This session established r=8 as the correct LoRA rank for Spoke's formatting task (confirmed by both literature research and empirical results). The r=8 1000-iter run will reveal the overfitting curve — expected to delay past r=16's iter 500 cliff. Once we know the r=8 sweet spot, the next high-impact experiment is swapping in the condensed training prompt v2 (adds "never answer questions", "preserve profanity", filler word rules from production Spoke app). Best current deployment config: Qwen3-4B, r=8, 200 iters, 6-bit = 83% spoke accuracy, 3.1 GB, ~0.8s latency.
