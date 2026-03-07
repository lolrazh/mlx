# V5 Stratified 80:10:10 Split + Qwen3 4B Benchmark

**Date:** 2026-03-07
**Agent:** Claude Opus 4.6
**Status:** ✅ Completed
**Continues from:** `2026-03-07_2100_epo-experiment-qwen3-4b.md` (EPO dead end, same session)

## User Intention
User wanted to create a proper 80:10:10 train/valid/test split from the v5 data, stratified by category so no category is accidentally omitted from any split. The old split (96.8% train / 1.5% valid / 1.7% test) had only 20 val and 23 test examples — making val loss meaningless and test accuracy unreliable. Then train with the same winning Qwen3 4B recipe and benchmark on the new, larger test set to get a real accuracy number.

## What We Accomplished
- ✅ **Built `build_split.py`** — Pools all 1,308 source examples from v3/source/ (556), v4/raw/ (666), v5/raw/ (86) with category labels from filenames. Maps 37 sub-categories to 12 broad categories. Deduplicates. Stratified 80:10:10 split.
- ✅ **New data split**: 1,046 train / 131 valid / 131 test (was 1,287 / 20 / 23)
- ✅ **Generated `test_set_v5.json`** — 131-example benchmark set with category labels, same format as test_set_v3.json
- ✅ **Uploaded to Modal** — Updated upload_data.py to include test_set_v5.json
- ✅ **Trained T4-v5split on Modal** — Qwen3 4B, 1000 steps (~3.8 epochs), same recipe as T3-v5. 4m38s, ~$0.50.
- ✅ **Benchmarked on both test sets**:
  - v3 test (23 ex): **100%** (22 exact + 1 semantic) — matches T3-v5
  - v5 test (131 ex): **79%** (97 exact + 6 semantic + 28 partial + 0 fail)
- ✅ **Updated LEDGER** with T4-v5split entry, benchmark table with category breakdown, finding #87
- ✅ **Key insight**: The model's "true" accuracy is 79%, not 100% — the old 23-example test set was too small to catch the gaps

## Technical Implementation

**Broad category mapping (37 sub-cats -> 12 broad):**
- spell: spell-replace, spell-simple, spell-corrective, spell-compound, spell-casual, spell-alt-phrase, spell-compound-scope (211 total)
- self-correction: self-correction, selfcorr-partial, selfcorr-mid, selfcorr-ambiguous (188)
- hard-negative: hn-disfluency, hn-quote, hn-symbols, hn-casing, hn-spelling (289)
- quote: quote-endquote, quote-unquote (124)
- multi: compound-selfcorr, compound-quote, compound-3plus, multistep-* (103)
- emoji (98), emphasis + emphasis-caps (79), caps (69), at-symbol (51), camelcase (51), disfluency (30), meta + tempting-questions (15)

**Training config (identical to T3-v5):**
- Model: Qwen/Qwen3-4B-Instruct-2507, LoRA r=8, alpha=16, dropout=0.05
- lr=1e-5, adam, constant scheduler, batch=4, max_seq=256
- 1000 steps (scaled from 1200 on 1287 examples to maintain ~3.8 epochs)

**Files Created/Modified:**
- `spoke/data/v5/build_split.py` — New stratified split builder
- `spoke/data/v5/train.jsonl` — Rebuilt (1046 examples, was 1287)
- `spoke/data/v5/valid.jsonl` — Rebuilt (131 examples, was 20)
- `spoke/data/v5/test.jsonl` — Rebuilt (131 examples, was 23)
- `spoke/data/v5/valid_categorized.json` — Categorized valid set for reference
- `spoke/bench/test_set_v5.json` — New 131-example benchmark set
- `spoke/bench/result_spoke-qwen3-4b-v5split-20260307_modal_v2_test_set_v3.json` — v3 results
- `spoke/bench/result_spoke-qwen3-4b-v5split-20260307_modal_v2_test_set_v5.json` — v5 results
- `spoke/cloud/upload_data.py` — Added test_set_v5.json to bench uploads
- `spoke/LEDGER.md` — T4-v5split entry, benchmark table, finding #87

## Bugs & Issues Encountered
1. **Background task IDs lost** — Two benchmark jobs launched with `run_in_background` couldn't be retrieved via TaskOutput (IDs not found). Re-ran them in foreground.
   - **Fix:** Run benchmarks sequentially in foreground instead of background.

2. **Benchmark script needs full path** — `--test-set test_set_v3.json` failed; needs `spoke/bench/test_set_v3.json`.
   - **Fix:** Used full relative path.

## Key Learnings
- **23-example test set was hiding real failures.** The model scored 100% on 23 examples but 79% on 131. The 23 examples happened to fall in the model's strong categories (self-correction, quote, emoji). The weak categories (at-symbol, multi-step, spell edge cases) were underrepresented or absent.
- **131-example val set makes eval_loss meaningful.** For the first time, eval_loss was monotonically decreasing (no noise). Previous 20-example val sets showed random fluctuations that led to wrong conclusions about overfitting.
- **Hard negatives are rock solid (100%).** 29/29 hard negatives passed — the model never over-edits on inputs that shouldn't be changed. This is a direct result of training with 289 hard negative examples.
- **at-symbol (20%) is the weakest category by far.** The model misplaces @ insertions — puts them on wrong words or adds extra @s. This was known as the hardest category for synthetic data generation (v3 datagen log noted 17% failure rate in generated data).
- **19% less training data didn't hurt.** 1,046 vs 1,287 training examples yielded identical results on the comparable v3 test set (100%). The model generalizes from patterns, not memorized examples.

## Architecture Decisions
- **Broad category stratification over sub-category** — Tiny sub-categories (7 examples) get pooled with siblings before splitting. Each broad category gets 80:10:10 representation.
- **1000 steps instead of 1200** — Scaled proportionally to maintain ~3.8 epochs (matching T3-v5's effective epoch count at checkpoint 1200).
- **Kept v3 test set for backward compatibility** — Can still compare against all historical runs on the same 23 examples.

## Ready for Next Session
- ✅ **New baseline established**: 79% on 131 examples with clear category-level failure analysis
- ✅ **Category priorities for v6 data**: at-symbol (20%), multi (50%), spell (67%), emoji (70%)
- ✅ **Val loss is now meaningful** — can use for early stopping / checkpoint selection in future runs

## Context for Future
The 80:10:10 split is a one-time infrastructure investment that pays dividends on every future experiment. Instead of guessing whether "100%" is real, we now have a 131-example test set that reveals actual weak spots. The category breakdown directly informs where to generate v6 training data: at-symbol needs the most help (20%), followed by multi-step (50%) and spell edge cases (67%). The model's strengths (hard-neg 100%, disfluency 100%, camelcase 100%) suggest those categories have enough training data and don't need more.
