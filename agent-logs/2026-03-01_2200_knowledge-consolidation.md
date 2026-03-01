# Knowledge Consolidation & Disk Cleanup

**Date:** 2026-03-01
**Agent:** Claude Opus 4.6
**Status:** ✅ Completed (consolidation + cleanup done; LFM2-T1 training still running)
**Building on:** `2026-03-01_2100_lfm2-baseline-finetune.md`

## User Intention
After 15 sessions of experiments, the user realized knowledge was scattered across ~8 docs and agent-logs, and disk was bloated at 64 GB with obsolete model weights. The goal was to consolidate all reusable insights into two canonical sources (LEDGER + RESEARCH), delete stale planning docs, and clean up ~48 GB of dead-end model artifacts — leaving a lean, navigable project.

## What We Accomplished
- ✅ **Consolidated all findings into LEDGER.md** — Mined 15 agent-logs for unique insights, added findings #28-33 (LFM2 architecture, few-shot format≠execution, IFBench, MoE memory, data gen insights). LEDGER now has 33 findings total.
- ✅ **Added LFM2 baselines to LEDGER** — B12 (LFM2.5-1.2B, 9%), B13 (LFM2-2.6B v2, 9%), B14 (LFM2-2.6B spoke, 30%). New "Alternative Models" training runs section with LFM2-T1.
- ✅ **Updated experiment queue** — LFM2-T1 marked ACTIVE, B-new updated (LFM2 done, remaining: Qwen3-1.7B, Gemma 3 1B, Llama 3.2 3B), Phase B roadmap updated.
- ✅ **Deleted 7 stale docs (-1,506 lines)** — PLAN.md, FINETUNE.md, DATAGEN.md, DATAGEN_V3.md, DATAGEN_V3_PATCH.md, PROMPT_GAPS.md, DATAGEN_BRIEF.md. All superseded by LEDGER.
- ✅ **Disk cleanup: 59 GB → 11 GB** — Removed 20 obsolete model/adapter directories. Kept only T11 (best model), T12b, LFM2-T1 (training), and essentials.
- ✅ **Explained LEDGER findings #1-27** — User was confused about "Finding #19" referenced in RESEARCH doc. Showed them the full 27-finding table and explained findings grew organically across sessions.
- ✅ **Research synthesis saved** — `spoke/RESEARCH_2025_2026.md` preserved from previous session (EPO, GRPO, LoRA-GA, distillation, mlx-lm-lora, etc.)
- 🔄 **LFM2-T1 training in progress** — At iter 280/800 when last checked. Val loss 0.703 (still dropping).

## Technical Implementation

### Knowledge Architecture (After Consolidation)
Two canonical sources:
- **`spoke/LEDGER.md`** — Our experiments. 33 findings, all training runs, baselines, experiment queue, research-informed roadmap. Single source of truth for everything we've learned hands-on.
- **`spoke/RESEARCH_2025_2026.md`** — External research. EPO, GRPO variants, LoRA-GA, rsLoRA, distillation recipes, mlx-lm-lora, data paradigms. References LEDGER findings by number.

Agent-logs remain as chronological session history (not consolidated — they serve a different purpose as detailed session journals).

### New LEDGER Findings (#28-33)
- **#28:** LFM2 hybrid conv+attention can't parse meta-linguistic commands zero-shot (9% for both 1.2B and 2.6B)
- **#29:** Few-shot teaches output FORMAT but not command EXECUTION (9% → 30%)
- **#30:** IFBench ≠ meta-linguistic instruction following (LFM2 beats DeepSeek R1 on IFBench, scores 9% on our task)
- **#31:** MoE models: full memory cost for partial compute (Qwen3.5-35B-A3B: 3B active, 35B in memory)
- **#32:** At-symbol is the hardest synthetic data category (17% gen failure rate)
- **#33:** Emphasis trigger-to-format mapping must be 1:1 in training data

### Disk Cleanup Details
**Deleted (20 directories, ~48 GB):**
- 3 fused models: `fused/`, `model-t6b-fused/`, `model-t6c-fused/` (22.5 GB) — AdamW dead ends
- 6 quantized models: `model-6bit/`, `model-t4-6bit/`, `model-t6-6bit/`, `model-t6b-6bit/`, `model-t6c-6bit/` (15.5 GB) — superseded by T11
- 3 base models: `model/`, `model-8bit/`, `model-4bit/` (9.2 GB) — re-downloadable from HF
- 10 adapter dirs: `adapters/`, `adapters-llama/`, `adapters-t4/`, `-t5/`, `-t6/`, `-t6b/`, `-t6c/`, `-t6c-best/`, `-t8/`, `-t12/` (~1.7 GB)

**Kept (11 GB):**
- `model-t11-fused/` (7.5 GB) — best model, source for quantization
- `model-t11-6bit/` (3.1 GB) — deploy model
- `adapters-t11/` (113 MB), `adapters-t12b/` (141 MB), `adapters-lfm2-t1/` (187 MB)

**Files Modified:**
- `spoke/LEDGER.md` — Added findings #28-33, baselines B12-B14, LFM2-T1 run, updated queue + roadmap
- `spoke/RESEARCH_2025_2026.md` — Created (previous session), preserved this session
- Deleted: `spoke/PLAN.md`, `spoke/FINETUNE.md`, `spoke/DATAGEN.md`, `spoke/DATAGEN_V3.md`, `spoke/DATAGEN_V3_PATCH.md`, `spoke/bench/PROMPT_GAPS.md`, `spoke/data/DATAGEN_BRIEF.md`
- Deleted: 20 model/adapter directories (~48 GB)

## Bugs & Issues Encountered
1. **Grep pattern for LEDGER findings failed** — `^### Finding #\d+` didn't match because findings use numbered list format (`1. **Finding...**`), not heading format.
   - **Fix:** Read the LEDGER directly from the findings section (line 225+). Always check format before grep.
2. **Git only tracked config/metadata files for models** — The large `.safetensors` weights were in `.gitignore`, so git commit only shows 24 small file deletions despite 48 GB freed on disk.
   - **Note:** Git history still has the small tracked files. A `git filter-repo` could shrink the repo further but isn't urgent.

## Key Learnings
- **Knowledge fragments organically across sessions.** Each session generates insights logged where convenient (agent-log learnings, MEMORY.md, inline comments). Without periodic consolidation, reusable knowledge becomes unfindable. Schedule consolidation every ~10 sessions.
- **Fused models are the biggest disk hogs.** Each fused Qwen3-4B bf16 = 7.5 GB. Four dead-end fused models = 30 GB wasted. Future: only fuse the best model, delete fused immediately after quantizing if you don't need the bf16 version.
- **Agent-log mining is effective for insight extraction.** An Explore agent read all 15 logs and cross-referenced against existing findings in ~35 seconds. Found 6 genuinely new insights out of ~27 candidates (most were already captured). Good ROI for periodic cleanup.

## Architecture Decisions
- **Two-file knowledge system (LEDGER + RESEARCH)** — Keeps our experimental learnings separate from external research. LEDGER is the "lab notebook," RESEARCH is the "literature review." Cross-references by finding number.
- **Agent-logs kept as history, not consolidated** — They serve as detailed session journals. Consolidating their content into LEDGER doesn't mean deleting them — the chronological narrative has value for debugging "what happened when."
- **Aggressive disk cleanup** — Deleted everything except T11 (best) and active experiments. Models are re-downloadable from HF, adapters are cheap to re-train. Storage is a real constraint on consumer hardware.

## Ready for Next Session
- ✅ **LEDGER is canonical** — 33 findings, all runs, all baselines, experiment queue, roadmap. Start here.
- ✅ **RESEARCH is canonical** — External research synthesis with tiered recommendations. Consult for next experiment ideas.
- 🔄 **LFM2-T1 training** — Should be complete by next session. Benchmark with: `python spoke/bench/run_benchmark.py --model lfm2-2.6b-exp --prompt-mode v2 --test-set spoke/bench/test_set_v3.json --adapter-path spoke/adapters-lfm2-t1`
- 🔧 **Remaining baselines** — Qwen3-1.7B, Gemma 3 1B QAT, Llama 3.2 3B still need zero-shot testing
- 🔧 **Q1 mixed-bit quantization** — Still queued, ready to run on model-t11-fused

## Context for Future
This session was primarily about reducing entropy — both in knowledge organization and disk usage. The project went from 8 scattered docs + 64 GB to 2 canonical sources + 11 GB. All experimental knowledge (33 findings) and external research are now instantly findable. The next session should focus on LFM2-T1 results and deciding the next experiment from the LEDGER queue.
