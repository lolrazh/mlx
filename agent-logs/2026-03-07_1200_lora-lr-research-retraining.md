# LoRA Learning Rate Research & Retraining

**Date:** 2026-03-07
**Agent:** Claude Opus 4.6
**Status:** 🔄 Ongoing

## User Intention
User wanted to verify whether we used optimal fine-tuning hyperparameters for Llama 3.2 3B (we didn't — LR was 10-20x too low). Also wanted general research into fine-tuning best practices, training duration, and whether Llama still has a shot vs Qwen3/Gemma 3n. Planning to re-run both Llama 3B and Qwen3 4B with corrected hyperparams.

## What We Accomplished
- **Research: Llama 3.2 3B fine-tuning recommendations** — Meta torchtune uses lr=3e-4, r=64. Community consensus: lr=2e-4, r=16-32.
- **Research: LoRA training duration & best practices** — 3-5 epochs for small datasets, constant LR is fine, val loss divergence from accuracy is well-documented (over-memorization).
- **Research: Llama for text editing/ASR tasks** — DRES paper confirms Llama's conservative editing bias is *good* for ASR post-processing. DistilLabs benchmark: Qwen3-4B ranks 1st, Llama 3.2 3B ranks 5th.
- **Key finding: lr=1e-5 is 10-20x too low for LoRA** — Same pattern as Gemma 3n (65% -> 91% after LR fix). All our LoRA runs used full-FT learning rates.

## Technical Implementation

### Our Config vs Recommendations
| Parameter | Our Value | Recommended | Gap |
|-----------|----------|-------------|-----|
| Learning rate | 1e-5 | 2e-4 to 3e-4 | 10-30x too low |
| LoRA rank | 8 | 16-64 | Low end |
| Alpha/scale | 2.0 | 2.0 | Correct |
| Scheduler | Constant | Constant is fine | OK |
| Target modules | All linear (mlx-lm default) | All linear | Correct |
| Epochs (~1201 ex) | ~6.6 | 3-5 | Slightly high |

### Key Papers
- "Learning Rate Matters" (arXiv:2602.04998): Optimal LoRA LR = 3.6e-4 to 6.3e-4
- "LoRA Without Regret" (Thinking Machines Lab): LoRA needs 10-15x full FT LR
- "Unveiling Over-Memorization" (arXiv:2508.04117): Val loss up + accuracy flat is normal, early stopping on val loss is misleading
- DRES (arXiv:2509.20321): Llama under-deletes (good for ASR), fine-tuned 3B gets +23.46 pts
- DistilLabs 12-model benchmark: Qwen3-4B #1, Qwen3-1.7B #4, Llama 3.2 3B #5

### Planned Runs
1. **Llama 3.2 3B** with lr=2e-4, r=16, 1200 steps, v4 data (Google hyperparams recipe)
2. **Qwen3 4B** with lr=2e-4, r=16, 1200 steps, v4 data (same recipe for fair comparison)

## Key Learnings
- **LoRA needs 10-15x higher LR than full fine-tuning** — LoRA updates <1% of params, so gradients need larger steps. 1e-5 is a full-FT LR; LoRA should use 1e-4 to 3e-4.
- **mlx-lm targets all linear layers by default** — No config needed. auto-discovers all Linear/QuantizedLinear modules.
- **Val loss divergence is over-memorization, not catastrophe** — Model becomes overconfident on correct tokens while miscalibrating others. Accuracy stays flat but OOD robustness drops.
- **Llama 3.2 3B's conservative editing is a feature** — DRES paper: under-deletion is better than over-deletion for ASR cleanup.
- **Qwen3-4B is the consensus #1 for fine-tuning** — DistilLabs benchmark, Springer QLoRA comparison, our own experiments all agree.

## Research Sources
- Meta torchtune configs (GitHub pytorch/torchtune)
- Sebastian Raschka's LoRA practical tips
- "Learning Rate Matters" (arXiv:2602.04998)
- "LoRA Without Regret" (Thinking Machines Lab)
- "Unveiling Over-Memorization" (arXiv:2508.04117)
- DistilLabs 12-model fine-tuning benchmark
- DRES (arXiv:2509.20321) — disfluency removal
- AMD ROCm Llama 3.2 3B LoRA tutorial
- rsLoRA paper (arXiv:2312.03732)

## Context for Future
This session establishes that ALL our LoRA runs used suboptimal learning rates (10-20x too low). The Gemma 3n experiments already proved higher LR works (65% -> 91%). Now re-running Llama and Qwen3 with corrected hyperparams to see if the 91% ceiling was an LR ceiling, not a model capacity ceiling. If Llama hits 96%+, it becomes the best deploy option (fastest inference + conservative editing).
