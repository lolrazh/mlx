# AdamW Ablation Conclusion & v3 Data Spec

**Date:** 2026-03-01
**Agent:** Claude Opus 4.6
**Status:** ✅ Completed

## User Intention
User wanted to complete the AdamW ablation (T6b benchmark + T6c 800-iter extension), understand *why* adam beat adamw (especially for 6-bit quantization), evaluate all remaining experiment options (T7/T9/T10/DPO/data expansion), and ultimately plan the next major improvement. After reviewing the full experiment landscape and the Spoke prompt trigger architecture, user decided to create a v3 dataset that only trains on categories with production triggers — removing categories the model will never see at inference time. The data generation work is delegated to a separate agent.

## What We Accomplished
- ✅ **T6b benchmarks** — 70% bf16, 57% 6-bit. 13% quant loss (worse than T4's 9%).
- ✅ **T6c training (800 iters)** — resumed from T6b iter 500, ran +300 more. Val loss bottomed at 0.190 @600 then plateaued ~0.204.
- ✅ **T6c benchmarks** — 74% bf16 (matches T4!), 61% 6-bit. Still 13% quant loss.
- ✅ **AdamW ablation conclusive** — same bf16 accuracy as adam, worse 6-bit (61% vs 65%), 2.7x slower training (800 vs 300 iters). Adam wins for 6-bit deploy.
- ✅ **Deep optimizer analysis** — explained to user why adam produces more quant-friendly weights (spiky distribution with few large weights vs adamw's uniform distribution).
- ✅ **Evaluated all remaining experiments** — T7 (cosine LR), T9 (QLoRA), T10 (mask_prompt), DPO, data expansion. Concluded data quality is the highest-ROI move.
- ✅ **Reviewed Spoke prompt trigger architecture** — 9 triggers (spelling, symbols, casing, quotes, disfluency, emphasis, emoji, camelcase, multi). Identified 4 test categories with NO triggers.
- ✅ **v3 data spec written** — `spoke/DATAGEN_V3.md` with clear instructions for the data agent.
- ✅ **Ledger fully updated** — T6c rows, findings 18-20, experiment queue revised.
- ✅ **Committed** — `e791ef3` T6c results with full AdamW conclusion.

## Technical Implementation

### AdamW vs Adam: Why Adam Wins for 6-bit Deploy
- **Adam** allows a few weights to grow large (spiky distribution). Large weights quantize cleanly because the grid spacing is proportional to magnitude. Most small weights round to zero. Clean.
- **AdamW** weight decay keeps all weights moderate (uniform distribution). Quantization noise hits more weights simultaneously. More damage.
- **Convergence**: Adam commits to weight values aggressively — good for small data (447 examples). AdamW's regularization fights specialization, needing 2.7x more iters.

### Final AdamW Comparison
| Run | bf16 | 6-bit | Quant Loss | Iters |
|-----|------|-------|------------|-------|
| T4 (adam) | **74%** | **65%** | 9% | 300 |
| T6c (adamw) | **74%** | 61% | 13% | 800 |

### v3 Data Architecture Decisions
- **Remove 4 categories without triggers**: formatting-xml, email, code-aware, hard-negative
- **Keep v2 prompt** (static ~80 tokens) — NOT dynamic spoke prompts (variable length 150-400 tokens breaks training)
- **Keep v2 data untouched** as generic fallback at `spoke/data/final/`
- **Target ~500 train / 20 valid / 23 test** — removing ~61 examples then adding ~114 targeted ones
- **Priority categories for new examples**: quote-endquote (period placement), camelcase (underrepresented), self-correction (compound corrections), at-symbol (underrepresented)

### Val Loss Paradox
T6c at val loss 0.190 scored 65% while val loss 0.205 scored 74%. With only 20 validation examples, val loss fluctuations of 0.01-0.02 are noise. Must benchmark accuracy directly.

**Files Modified:**
- `spoke/LEDGER.md` — T6c training row, 4 benchmark rows, findings 18-20, experiment queue updated
- `spoke/config.yaml` — Updated for T6c (iters=300, adapter_path=adapters-t6c)

**Files Created:**
- `spoke/DATAGEN_V3.md` — Complete v3 data spec for data generation agent
- `spoke/bench/result_qwen3-4b-bf16+lora_v2.json` — T6c bf16 benchmark
- `spoke/bench/result_model-t6c-6bit_v2.json` — T6c 6-bit benchmark
- `spoke/model-t6c-fused/` — Fused T6c model (bf16)
- `spoke/model-t6c-6bit/` — T6c 6-bit quantized model
- `spoke/adapters-t6c/` — T6c adapters (100/200/300 checkpoints)
- `spoke/adapters-t6c-best/` — T6c best val loss checkpoint (iter 600)
- `agent-logs/2026-03-01_1000_t6b-benchmark-t6c-launch.md` — Mid-session log (updated with T6c results)
- `agent-logs/2026-03-01_1200_adamw-conclusion-v3-data-spec.md` — This log

## Bugs & Issues Encountered
1. **Benchmark model name error** — Used `qwen3-4b-bf16+lora` but script expects `qwen3-4b-bf16` with separate `--adapter-path`
   - **Fix:** Use short name `qwen3-4b-bf16`, pass adapter directory separately
2. **Adapter checkpoint benchmark** — Script expects a directory with `adapter_config.json`, not a single `.safetensors` file
   - **Fix:** Created `adapters-t6c-best/` directory with config + renamed checkpoint
3. **Optimizer state reset on resume** — `--resume-adapter-file` loads weights but resets momentum/velocity. Caused val loss bump 0.200→0.219 before recovering.
   - **Workaround:** Expect ~50-iter warm-up period after resume. Don't trust val loss during this window.

## Key Learnings
- **AdamW's quant regression is consistent** — 13% loss at both 70% (T6b) and 74% (T6c) accuracy. Not a fluke.
- **Adam's "spiky" weight distribution is quant-friendly** — few large weights that own their quantization grid, many near-zero weights that round cleanly.
- **Val loss with 20 validation examples is unreliable** — 0.015 difference can mean 9% accuracy swing in either direction. Always benchmark.
- **Optimizer state matters for resumed training** — plan for warm-up period, don't judge early checkpoints after resume.
- **Training data ceiling > hyperparameter ceiling** — after exploring rank, optimizer, adapter type, and quantization, the remaining accuracy gap is entirely about what examples the model has seen.

## Architecture Decisions
- **Adam confirmed as production optimizer** — not because it's theoretically superior, but because it produces weights that survive 6-bit quantization better for this specific task size.
- **v3 data = trigger-matched** — only train on categories the model will actually see in production. Training on unreachable categories wastes model capacity.
- **Static v2 prompt for training** — dynamic spoke prompts vary 150-400 tokens per trigger, which would complicate training dynamics. Static prompt keeps things consistent.
- **Hard negatives removed** — if no trigger fires, input never reaches the model. No need to train "don't format" responses.

## Ready for Next Session
- ✅ **v3 data spec ready** — `spoke/DATAGEN_V3.md` has complete instructions for data agent
- ✅ **T4 (adam, iter 300) confirmed as baseline** — 74% bf16, 65% 6-bit, 3.1 GB
- ✅ **Ledger fully current** with 20 key findings and all experiments marked done/planned
- 🔧 **Data agent needs to execute v3 spec** — filter v2, generate ~114 new examples, create test/valid
- 🔧 **After v3 data**: retrain T4 config on v3 data, benchmark, compare

## Context for Future
The hyperparameter exploration phase is effectively complete (rank, optimizer, adapter type, quantization all tested). The ablation confirmed T4's config (LoRA r=8, adam, flat LR 1e-5, ~300 iters) as optimal for 6-bit deploy. The next phase is data-driven: v3 focuses the dataset on trigger-matched categories, adds ~114 targeted examples for weak spots (quote period placement, camelCase, compound corrections), and should push past T4's 74%/65% ceiling. After v3 training, consider DPO for near-miss examples if SFT plateaus again. Building on `2026-03-01_1000_t6b-benchmark-t6c-launch.md`.
