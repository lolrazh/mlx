# LFM2-T1 Benchmark: 78% & Extended to 2000 Iters

**Date:** 2026-03-01
**Agent:** Claude Opus 4.6
**Status:** 🔄 Ongoing (LFM2-T1b training to 2000 iters in progress)
**Building on:** `2026-03-01_2200_knowledge-consolidation.md`, `2026-03-01_2100_lfm2-baseline-finetune.md`

## User Intention
After the knowledge consolidation session, the user wanted to track the LFM2-T1 training (launched earlier at 800 iters). When it completed with a stunning 78% accuracy — just 5 points below the best Qwen3 model (83%) — the user immediately wanted to push further, extending training to 2000 iters since the model showed no signs of overfitting and val loss was still dropping steeply.

## What We Accomplished
- ✅ **LFM2-T1 training completed** — 800 iters, val loss 1.771 → 0.480, train loss down to 0.328. No overfitting.
- ✅ **Benchmarked LFM2-T1: 78% bf16 accuracy** — 18 exact, 0 semantic, 5 partial, 0 fail. 1.54s avg latency.
- ✅ **Updated LEDGER** — Added LFM2-T1 results, new findings #34-37, updated experiment queue and roadmap.
- 🔄 **LFM2-T1b training launched** — Extended to 2000 iters, resumed from iter 800 checkpoint. ~2 hours estimated.

## Technical Implementation

### LFM2-T1 Final Results

| Metric | LFM2-T1 (2.6B) | T11 Qwen3-4B (best) |
|--------|----------------|---------------------|
| Zero-shot | 9% | 22% |
| Fine-tuned bf16 | **78%** | **83%** |
| Fine-tune gain | **+69 pts** | +61 pts |
| Avg latency | **1.54s** | 2.67s |
| Val loss final | 0.480 @800 | 0.169 @250 |
| Peak memory | 13.3 GB | 14 GB |
| Training speed | 0.17 it/sec | ~2 it/sec |

### Val Loss Curve (Double-Descent Pattern)

```
1.771 → 0.863 → 0.796 → 0.748 → 0.733 → 0.703 → 0.702 → 0.701 (PLATEAU: 250-400)
→ 0.686 → 0.692 → 0.678 → 0.646 → 0.597 → 0.534 → 0.501 → 0.482 → 0.480 (SECOND DROP: 500-800)
```

Phase 1 (iter 1-250): Steep drop, attention layer LoRA adapting.
Phase 2 (iter 250-400): Plateau at ~0.70.
Phase 3 (iter 400-800): Second steep drop to 0.480 — conv layer adapters kicking in.

### What LFM2-T1 Gets Right vs Qwen3

**LFM2 wins:**
- All 3 self-corrections (Qwen3-T11 misses #6 compound: "React and Svelte")
- 0 fails (Qwen3-T11 has 1 fail)

**Qwen3 wins:**
- 83% vs 78% overall
- Spell-replace (3/3 vs 1/3)
- Quote-unquote (3/3 vs 2/3)

**Both miss:**
- Emphasis #21: LFM2 uses CAPS, Qwen3 also struggles here
- Emoji #22: word order issue

### LFM2-T1b Config (Extended Training)
```yaml
model: mlx-community/LFM2-2.6B-Exp-bf16
iters: 2000
resume_adapter_file: ./spoke/adapters-lfm2-t1/adapters.safetensors
# Everything else unchanged from T1
```

**Files Modified:**
- `spoke/config.yaml` — iters 800→2000, added resume_adapter_file
- `spoke/LEDGER.md` — Added LFM2-T1 results, LFM2-T1b run, findings #34-37, updated queue/roadmap
- `spoke/bench/result_lfm2-2.6b-exp+lora_v2.json` — Benchmark results

## Key Learnings
- **Zero-shot is meaningless for predicting fine-tuning ceiling.** LFM2 gained 69 pts from LoRA despite scoring 9% zero-shot. The model that gained MORE had the WORSE baseline. This is now finding #34.
- **LFM2 hybrid architecture has complementary strengths to pure transformers.** Conv layers seem to help with local pattern matching (self-corrections), while attention layers handle global structure (quoting, spelling). This challenges the "decoder-only can't edit" hypothesis — maybe conv+attention is its own category.
- **Double-descent learning curves happen in hybrid architectures.** The plateau at iter 250-400 followed by steep second drop (500-800) suggests different adapter types converge at different rates. More iters = more phases = potentially higher ceiling.
- **Optimizer state reset on resume causes ~50 iter warmup** (finding #20). Expected for LFM2-T1b. Watch for initial val loss bump.

## Architecture Decisions
- **Extended to 2000 iters (not restarting)** — Val loss 0.480 and dropping, no overfitting, train loss 0.328. Strong signal that more training helps. Resume from checkpoint is ~3x faster than restarting from scratch.
- **No config changes other than iters** — Same lr, rank, batch_size. The model is learning well — don't touch what works.

## Ready for Next Session
- 🔄 **LFM2-T1b training** — 2000 total iters. When done: benchmark same way (`--adapter-path spoke/adapters-lfm2-t1`). Compare to 78% (800 iters) and 83% (Qwen3 T11).
- ✅ **All results in LEDGER** — Findings #34-37 capture the key LFM2 insights.
- 🔧 **If LFM2 matches or beats 83%**: Fuse + quantize to 6-bit, test quant robustness. LFM2's smaller size (2.6B vs 4B) means 6-bit could be ~1.5 GB — very deployable.
- 🔧 **Still queued**: Q1 mixed-bit quant on T11, remaining zero-shot baselines (Qwen3-1.7B, Gemma 3 1B, Llama 3.2 3B).

## Context for Future
LFM2-T1 at 78% is the most surprising result of the entire Spoke project. A 2.6B hybrid conv+attention model that scored 9% zero-shot is now 5 points behind the 4B Qwen3 champion — and it's faster (1.54s vs 2.67s), uses less memory (13.3 vs 14 GB), and solves the compound self-correction that Qwen3 can't. If T1b (2000 iters) closes the gap further, LFM2 becomes the deploy candidate: smaller, faster, and possibly more accurate. The user's instinct to "just try it" despite the 9% zero-shot was vindicated.
