# Bake-off Round 2: Nanbeige4.1-3B, Ministral 3 3B, Nemotron-3-Nano-4B

**Date:** 2026-07-05
**Agent:** Claude Fable 5
**Status:** Completed (Ministral invalidated, Nemotron parked)

## User Intention
Try three more models on the champion recipe (LoRA r=8, adam, lr=1e-5, 2000 steps, v5-split, v2 prompt): Ministral 3 3B, Nemotron 3 Nano 4B, Nanbeige4.1 3B. Earlier in the session: answered Gemma 4 RAM questions (31B ≈ 17.5–18 GB Q4, 26B A4B ≈ 14.4 GB Q4 — MoE is full memory, finding #31) and confirmed Gemma 3n E4B was already tried (96% core23 / 59% broad, never top-5 on broad).

## What We Accomplished
- **Pipeline extensions:** `mistral3` added to the multimodal text-only allowlist (train/benchmark/merge); `nemotron_h` Mamba LoRA targets (`in_proj`/`out_proj`); generation_config sanitization on export (strip sampling flags when `do_sample=False`); tokenizer 5.x→4.x fallback in the merge script; dedicated **`SPOKE_MAMBA_IMAGE=1`** image for Mamba hybrids.
- **Trained + benchmarked Nanbeige4.1-3B** (`spoke-nanbeige41-3b-v5split-2k-20260705`) — clean run, zero pipeline changes (`model_type: llama`).
- **Trained Ministral 3 3B** (`spoke-ministral3-3b-v5split-2k-20260705`, unsloth bf16 repo — official is FP8) — benchmarks returned 0% all-partials; diagnostic proved the **base model through our text-only path outputs token salad** → loader bug, run INVALID.
- **Nemotron-3-Nano-4B**: ~10 launch attempts to get the environment working, then two full training runs (lr=1e-5 champion recipe; lr=2e-4 Gemma-3n-style probe). Both failed on quality in opposite directions. PARKED.
- Diagnosed the "cancelled by user or a failure" mystery (3 occurrences incl. yesterday's Qwen3.5-2B): **flaky local network kills attached `modal run` sessions**. Retry loops now mandatory for benchmarks.
- Ledger: findings #96, #97; queue entries `ministral-fix`, `nemotron-midlr`; header updated.

## Results

| Model | core23 | broad58 | v5-131 | Verdict |
|-------|--------|---------|--------|---------|
| Nanbeige4.1-3B | **70%** | **40%** | **66%** | 0 hard fails. Arena-Hard-beats-Qwen3-4B pedigree does NOT transfer (finding #96). Below plain Llama 3.2 3B. |
| Ministral 3 3B | — | — | — | INVALID: `config=text_config` + `AutoModelForCausalLM` doesn't remap `language_model.*` VLM prefixes (analog of finding #83). Fix = load full `Mistral3ForConditionalGeneration`. |
| Nemotron lr=1e-5 | 22% | 26% | 31% | **Echoes input verbatim — zero commands executed.** 12 hard fails on core23 (every other FT model: 0–1). |
| Nemotron lr=2e-4 | — | 0% | 0% | **Opposite pole: edits execute correctly, then multilingual repetition collapse** ("oorspronkelijke tekst:" loops). ckpt-1000 not retained (save_total_limit=5 kept 1600–2000, all degenerate). |

## The Nemotron ops saga (finding #97's tax — do not relearn)
1. `mamba_ssm` missing → 2. sdist needs nvcc (`bare_metal_version` NameError) → 3. prebuilt wheel metadata upgrades torch to 2.12/cu13, breaking torchvision → 4./5. both cxx11abi wheel variants fail with undefined `c10::Warning` symbols vs conda AND pip torch 2.6 → 6. same → 7. PyPI sdist missing csrc files → 8. my sed broke Python quoting → 9. source build imports need `triton.set_allocator` (triton≥3.5) → **10. WORKING: debian_slim + pip torch 2.9 + torch2.9-tagged wheels `--no-deps`** — mamba-ssm 2.3.x simply targets the torch-2.9 era; its torch2.6 wheels are broken uploads. Post-training: NVIDIA's `trust_remote_code` needs transformers 4.x (`_tied_weights_keys` list vs 5.x dict) → mamba image pins **transformers 4.53**; checkpoint tokenizers saved under 5.3 stamp `TokenizersBackend` which 4.x can't read → merge falls back to base-repo tokenizer; Nemotron ships `top_p=0.95` with `do_sample=False` which transformers 5.x refuses to save → sanitization added.

## Key Learnings
- **Finding #96:** alignment/reasoning benchmark wins measure eloquent generation; Spoke needs disciplined copying. Extends finding #30.
- **Finding #97:** the champion recipe is attention-transformer-specific. Mamba mixer weights need a bigger kick than lr=1e-5 to escape the copy prior, but lr=2e-4 destabilizes stopping. Viable window (2e-5–5e-5) unproven; keep `save_total_limit≥20` if ever probed.
- Plain-`llama`-architecture models (Nanbeige, MiniCPM5) cost zero ops effort; hybrids cost days. Weight the bake-off queue accordingly.
- Benchmark harness: all-partial 0% = systematic artifact (wrapper/loader/EOS), never model quality. Always run the base model through the same path as a control before believing a 0%.

## Context for Future
Round verdict: five modern models, zero threats to the March champion (100/74/82). Data (v6) remains the only proven lever; Gemma 4 E4B is the one untried model with a real thesis (3n E4B hit 96% core; does a better base close the 59→74 broad gap?). Ministral fix and Nemotron mid-LR probe are queued as low priority. Modal: sandy-36852, retry loops for attached runs.
