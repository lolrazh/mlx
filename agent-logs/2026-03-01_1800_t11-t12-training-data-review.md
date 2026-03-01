# T11/T12 Training, Cross-Test Analysis, and Data Patch Review

**Date:** 2026-03-01
**Agent:** Claude Opus 4.6
**Status:** 🔄 Ongoing (patch data review in progress)

## User Intention
Push past T4's accuracy ceiling (74% bf16 / 65% 6-bit) by training on the new v3 trigger-matched dataset, then isolate what's actually improving. After T11 succeeded (83% bf16), the user wanted to fix the remaining 4 bf16 failures with targeted data. When the data patch regressed T12, the focus shifted to data quality review.

## What We Accomplished
- ✅ **Established v3 baseline (B11 = 22% zero-shot)** — created test_set_v3.json from v3/test.jsonl
- ✅ **Trained T11 (v3, 492 train) → 83% bf16, 74% 6-bit** — new best on v3 test set
- ✅ **Cross-test analysis revealing T11's improvement is test-set alignment** — T4 and T11 both score 76% on shared kept-category v2 examples
- ✅ **Quantified v3 data churn** — 57% of v2 touched, only 3 wrongly removed, real gap: 1/150 new self-correction examples
- ✅ **Wrote DATAGEN_V3_PATCH.md** — targeted brief for data agent (4 failure categories, 35-50 examples)
- ✅ **Trained T12 (v3+patch, 535 train) → 74% bf16 REGRESSION** — patch fixed 0/4 targets, caused 2 new regressions
- 🔄 **Launched Opus data quality review** — 3 agents reviewing all 43 patch examples

## Technical Implementation

### T11 Training
- Config: adam, r=8, lr=1e-5, 300 iters, v3 data (492 train)
- Val loss curve: 2.843 → 0.169 @250 (plateau)
- bf16: 83% (18 exact + 1 semantic), 6-bit: 74% (16 exact + 1 semantic)
- Quant loss: 9% (same as T4)

### Cross-Test Discovery
On 17 shared kept-category examples from v2 test:
- T4 6-bit: 76% (13/17) — gained #9, #12, lost #6, #14 vs T11
- T11 6-bit: 76% (13/17) — identical score, different failures
- T11's "83%" came from testing only trigger-matched categories (removed 6 impossible examples)

### T12 Training (REGRESSION)
- Config: identical to T11, data: v3+patch (535 train, +43 targeted)
- Val loss: 0.162 @300 (BEST EVER) but accuracy: 74% (WORST since T11)
- New regressions: #2 spell-replace ("Kadai"→"Kdi"), #14 emphasis (single * not **)
- Unfixed targets: #6 self-correction, #8 quote scope, #17 camelCase, #22 emoji

### Data Patch Composition (43 examples)
- 17 self-correction (compound: keep earlier items, replace last)
- 9 emoji (strip "emoji" word after conversion)
- 8 quote-unquote (narrow scope to 1-3 words)
- 8 camelCase (compound identifiers with file extensions)

**Files Modified:**
- `spoke/bench/run_benchmark.py` — added --test-set flag
- `spoke/bench/test_set_v3.json` — created (23 examples, 9 categories)
- `spoke/bench/test_set_v2.json` — restored from git history (data agent had overwritten)
- `spoke/config.yaml` — updated for T11, then T12
- `spoke/LEDGER.md` — T11/T12 results, findings 21-24, cross-test rows, experiment queue
- `spoke/DATAGEN_V3_PATCH.md` — created (brief for data agent)

## Bugs & Issues Encountered
1. **test_set.json overwritten by data agent** — v3 data agent replaced original v2 test set with v3 categories
   - **Fix:** Recovered from git: `git show 0ae7e10~1:spoke/bench/test_set.json > test_set_v2.json`
2. **T12 regression (83% → 74%)** — 43 targeted patch examples made accuracy worse
   - **Root cause:** Suspected data quality issues (patch examples not reviewed before training)
   - **Fix in progress:** 3 Opus subagents reviewing all 43 examples

## Key Learnings
- **Val loss paradox (again):** T12 had best-ever val loss (0.162) but worst accuracy. With 20 val examples, loss improvements don't correlate with accuracy (finding #19, confirmed again).
- **Test-set alignment ≠ model improvement:** T11's +9 pts came from removing impossible-category test examples, not from the model getting better on shared examples (76% = 76%).
- **More data ≠ better data:** 43 targeted examples with suspected quality issues caused regression. Always review generated data before training.
- **Data churn quantification:** v3 agent touched 57% of v2 data. Only 3 wrongly removed out of 105, but distribution was off (1 self-correction out of 150 new).
- **mlx_lm has no distillation/DPO:** Only SFT (LoRA, DoRA, full fine-tune). Data quality is the only lever.

## Architecture Decisions
- **One variable at a time:** User explicitly requested this discipline. Cosine LR queued but deferred until data experiment resolves.
- **bf16 focus:** User decided to forget quantization for now, optimize bf16 accuracy first.
- **Opus review before training:** Learned from T12 that unreviewed data is dangerous. Always run Opus quality review on generated data.

## Ready for Next Session
- ✅ **T11 is current best model** — 83% bf16, 74% 6-bit on v3 test set
- ✅ **LEDGER fully up to date** through T12 + finding #24
- ✅ **Benchmark infra supports v3 test set** — `--test-set spoke/bench/test_set_v3.json`
- 🔧 **Patch data review in progress** — 3 Opus agents reviewing 43 examples for quality issues
- 🔧 **T12b queued** — re-train on cleaned patch data after review

## Context for Future
T11 (83% bf16, 74% 6-bit) is the accuracy ceiling. 4 bf16 failures remain: compound self-correction (#6), quote-unquote scope (#8), camelCase (#17), emoji word stripping (#22). The first attempt at targeted data (T12) regressed — the patch data likely has quality issues. Opus review agents are analyzing all 43 examples. After cleanup, T12b will re-train with vetted data. Cosine LR (T13) is next in queue after data experiments resolve.
