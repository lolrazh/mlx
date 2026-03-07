# EPO (Edit-Weighted Loss) Experiment on Qwen3 4B

**Date:** 2026-03-07
**Agent:** Claude Opus 4.6
**Status:** ✅ Completed
**Continues from:** `2026-03-07_1830_gemma3n-e4b-v5-experiments.md` (previous session's experiment queue)

## User Intention
User wanted to test EPO (Edit-wise Preference Optimization) as the next training innovation after rsLoRA and lr=5e-5 both failed to beat the baseline. The hypothesis: upweighting edit tokens in the loss function would help the model focus gradient signal on the ~10% of tokens that actually change, improving generalization. User specifically chose Qwen3 4B (not Qwen3.5) with the "winning recipe" — same hyperparameters, only the loss function changes.

## What We Accomplished
- ✅ **Committed previous session's pending changes** — rsLoRA support, merge EOS fix, lr=5e-5 benchmarks, 8 result JSONs
- ✅ **Implemented EPO loss in `train_hf.py`** — Character-level LCS diff identifies edit vs copy tokens, per-token weighted cross-entropy loss, full diagnostic stats
- ✅ **Trained Qwen3 4B with EPO on Modal** — 1200 steps, edit_weight=3.0, v5 data, v2 prompt. EPO stats: 10.9% edit tokens, 89.1% copy tokens
- ✅ **Benchmarked on both test suites** — Core23: 87% (-13 pts), Broad58: 66% (-3 pts). Clear regression
- ✅ **Diagnosed failure mechanism** — Over-editing: model drops words that should be preserved ("Okay", "really", "React and", "absolutely")
- ✅ **Updated LEDGER with finding #86** — EPO dead end for copy-heavy tasks
- ✅ **Updated MEMORY.md** with EPO lesson

## Technical Implementation

**EPO Loss Implementation** (`spoke/cloud/train_hf.py`):
- `get_edit_char_mask()`: Uses `difflib.SequenceMatcher` for character-level LCS between user input and assistant output. Characters NOT in LCS = edit characters.
- Token mapping: `tokenizer(text, return_offsets_mapping=True)` maps character positions to token positions. If ANY character in a token's span is an edit, the whole token gets `edit_weight`.
- `parity_data_collator`: Extended to pad `token_weights` alongside `input_ids`/`labels`, with proper shift alignment.
- `MLXParityTrainer.compute_loss()`: Overridden to use `reduction='none'` cross-entropy, multiply by per-token weights, normalize by weight sum.

**New CLI parameters**: `--loss-mode` (standard/epo), `--epo-edit-weight` (default 3.0)

**EPO dataset stats**: 2,192 edit tokens / 20,050 active tokens = 10.9% edit rate across 1,287 training examples.

**Files Modified:**
- `spoke/cloud/train_hf.py` — EPO loss implementation (get_edit_char_mask, weighted loss, data collator, diagnostics)
- `spoke/LEDGER.md` — EPO experiment entry, benchmark results, finding #86
- `spoke/bench/result_spoke-qwen3-4b-epo-w3_modal_v2_test_set_v3.json` — Core23 results
- `spoke/bench/result_spoke-qwen3-4b-epo-w3_modal_v2_test_set_evals.json` — Broad58 results

## Bugs & Issues Encountered
1. **Modal CLI not found in shell**
   - Symptom: `command not found: modal` when running `modal run`
   - **Fix:** `pip install modal` into the active `.venv` environment. Should have activated venv first (`source .venv/bin/activate`) per CLAUDE.md instructions.

2. **Duplicate parameter signatures in train/main functions**
   - Symptom: `Edit` tool found 2 matches for `export_merged: bool = True, use_rslora: bool = False`
   - **Fix:** Added more surrounding context to uniquely identify the `train()` function signature vs the `main()` entrypoint.

## Key Learnings
- **EPO (edit-weighted loss) is wrong for copy-heavy tasks.** The fundamental insight: for Spoke, ~90% of output tokens SHOULD be copies. Standard SFT's "boring" equal token weighting naturally allocates 90% of gradient signal to copy tokens — which IS the correct priority. EPO flips this incentive by upweighting edits, teaching the model "editing = more valuable than copying," which manifests as over-deletion of words.
- **The failure mode is qualitatively distinct.** EPO doesn't just make the model "worse" — it creates a systematically different (more aggressive) editing policy. The model drops intensifiers ("really," "absolutely"), discourse markers ("Okay"), and secondary items in lists ("React and"). It learned to trim aggressively.
- **GEC ≠ ASR post-processing.** EPO was SOTA for grammatical error correction where models under-edit. Spoke's task is the opposite — models should default to copying and only change what's explicitly commanded. The literature insight doesn't transfer.
- **10.9% edit ratio is informative.** This confirms the task's extreme copy-heaviness. Any training technique that shifts the loss balance away from copy tokens will likely hurt.

## Architecture Decisions
- **Character-level LCS over word-level alignment** — More robust to tokenization artifacts and handles punctuation changes naturally. `difflib.SequenceMatcher` is efficient enough for short texts (<200 chars).
- **Case-sensitive comparison** — Capitalization changes ARE meaningful edits in Spoke (e.g., "hey" → "Hey"). Case-insensitive would miss these.
- **Override `compute_loss` instead of `compute_loss_func`** — HF Trainer's `compute_loss_func` only receives `(outputs, labels)`, not the full batch dict. Overriding `compute_loss` gives access to `token_weights` from the batch.

## Ready for Next Session
- ✅ **EPO infrastructure is reusable** — The weighted loss code works correctly and can be adapted for future experiments (e.g., inverse EPO: upweight COPY tokens)
- ✅ **All results committed and pushed** — LEDGER updated, finding #86 documented

## Context for Future
Three training innovations have now been tested and failed for Spoke's copy-heavy task: rsLoRA r=16 (-9 core), lr=5e-5 (-5 broad), EPO (-13 core). The common thread: anything that pushes the model toward more aggressive behavior hurts. The baseline recipe (lr=1e-5, r=8, standard SFT, v5 data) at 100% core / 69% broad remains optimal. Next gains will likely come from better data (v6) or fundamentally different approaches (tool-call edit format, distillation from larger teacher), not training algorithm tweaks.
