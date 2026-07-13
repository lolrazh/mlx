# MoE Expert-Flashing Investigation → Dense-Champion Shrink: 2-bit Embeddings Proven at 3.3 GB

**Date:** 2026-07-12 → 2026-07-13 (overnight)
**Agent:** Claude Opus 4.8 (1M context), main loop, with 5 research/execution subagents (3 Sonnet research, 2 Opus execution)
**Status:** IN PROGRESS. Feasibility of sub-3 GB dense shrink PROVEN (2-bit embeddings + DWQ = 3.3 GB @ broad58 78% local, matches 4-bit baseline's 76%, 0 hard fails, heal loss 0.038). PLD latency lever explored and RULED OUT (finding #113 — a bust on this 262K-vocab model, and not bit-lossless); the real latency win is **persistent system-prompt caching** (~35%, lossless, median ~500 ms), deferred for wiring per user (focus on shrink first). Shrink now EXHAUSTED: g64 (3.1 GB / 3.41 GB peak / 79.3%) is the deploy pick — mixed-body / vocab-prune / E2B-slice all ruled out; under-3 GB at full quality needs a paid E2B fine-tune. Latency: g64 + prompt-cache = median 488 ms (clean, idle). Durable backlog + inference recipe live in `spoke/TODO.md`.

## User Intention
User proposed running a **Mixture-of-Experts model with experts streamed/flashed from SSD** (à la Apple's "LLM in a Flash" + the GLM-on-25GB-RAM repos) to slash memory. Specifically wanted to fine-tune the **Gemma 4 26B-A4B** MoE and flash its experts so a big-capacity "mug-it-up" model could run cheaply. Hard product constraints stated explicitly: **latency <400–800 ms**, **RAM ~2–3 GB**, reasonable size. Disciplined ask: *prove the method is possible before spending any money/mental bandwidth on fine-tuning.* Then, once streaming was ruled out, pivoted to **shrinking the existing dense Gemma 4 E4B champion below 3 GB** without losing its 82.8 broad58. Requested Sonnet subagents for research, Opus subagents for code/execution, an agent-log, and bite-sized git commits.

## The arc

### 1. MoE expert-flashing: researched, then ruled out for THIS use case
- Two Sonnet research agents established: (a) MoE expert-streaming on Apple Silicon is real (SwiftLM, deepseek-v4-flash-mlx, mlx-flash, TurboQuant-MLX run 100B+ MoEs on 16–64 GB Macs at ~1–5 tok/s), but "streaming" in practice = the OS page cache serving hot experts; (b) mlx-lm has NO built-in expert offload (issue #1438 open) and its loaders **fuse** the expert bank at load (`qwen3_moe.py` `mx.stack`, `gpt_oss.py` `mx.contiguous`), forcing it resident — so streaming needs a custom loader.
- **Gemma 4 26B-A4B exists and IS MoE** (25.2B total / 3.8B active, top-8-of-128 + 1 shared, Apache-2.0). The user was right; earlier I'd wrongly implied no Gemma MoE. The dense E4B champion is NOT MoE (MatFormer + PLE).
- **Killer analysis (from the user's own prior work, findings #99–102 + `ifp/`):** Spoke routing on this MoE is DIFFUSE — a single example fires ~76% of experts (prefill-dominated), 58 examples fire 96.3%. Ran a new offline streaming-cache simulation (`ifp/stream_sim.py`) on the saved routing traces: to serve 95% of decode accesses from RAM you must cache ~50% of experts (~6.9 GB); to fit 2–3 GB you cache ~16/128 → ~60% hit → seconds of latency. **Verdict: 2–3 GB RAM AND <800 ms via streaming this MoE is impossible — ~3× over on RAM and ~6× over on latency simultaneously.** Prefill of the detailed prompt alone cold-loads most of the model. The existing dense champion (3.9 GB, 0.79 s) already beats every streaming operating point.

### 2. Pivot: shrink the DENSE E4B champion (the right lever)
- Established that **decode is memory-bandwidth-bound** (~120 GB/s ÷ 4 GB ≈ 18–24 tok/s), so serving tricks floor out ~400–700 ms for a 4 GB model — **shrinking the model is the ONLY real latency lever, and it also hits the RAM target.** The two goals collapse into one.
- **Local teardown of the deployed DWQ model** (`spoke/models/g4e4b-champion-mlx-dwq4-g64`, config: hidden 2560, 42 layers, vocab 262144): MLP 44% (1.86 GB), **PLE per-layer embeddings 39%** (1.63 GB, one tensor `embed_tokens_per_layer` [262144×1344]), token embeddings 9%, attention 8%. Nearly half the model is compressible lookup tables.
- Third Sonnet research agent (compression SOTA): DWQ is the top MLX-native lever; `mlx_lm.convert --quant-predicate mixed_{2,3,4}_6` and `nn.quantize(class_predicate=…)` give per-tensor bit control; mandatory ordering **Prune → Distill → Quantize**; a known trap is that naive low-bit quant of Gemma 4's PLE (learned-scalar `ScaledEmbedding`) produces garbage — but our deployed model already quantizes PLE to 4-bit and works, because DWQ healed it. That de-risked pushing lower.

### 3. Shrink experiment (Opus subagent, driven to completion by main loop)
- Asymmetric quant (`spoke/shrink/asym_quant.py`): `embed_tokens` + `embed_tokens_per_layer` → 2-bit (group 32 and group 64 variants), body stays 4-bit g64, then DWQ-heal (512 calibration samples) against the bf16 teacher.
- **Result (g32 variant): 3.3 GB, broad58 78% (42 exact / 3 semantic / 13 partial / 0 fail), heal val-loss 0.038** — vs the re-run 4-bit baseline 3.9 GB / 76% / 0 fail / loss 0.039. i.e. **2-bit embeddings MATCH 4-bit at 0.6 GB smaller and ~40% faster wall-clock, zero garbage/thinking-runaway.** The g64 raw variant is 3.1 GB (not yet healed). Feasibility of sub-3.2 GB at champion-parity: PROVEN on the first experiment.

### 4. Latency / PLD lever — DONE, verdict: cache the system prompt, skip PLD (finding #113)
Prompt-lookup decoding *should* fit a text-cleaner (output echoes input) but is a **bust** here: best config ~1.05×, aggressive k=8 actually *slower* (0.86×). Cause: the 262K-vocab lm_head makes multi-token verification non-free (forward flat only to a 2-token block: 47.6→45 ms, then steep: block-4 61 ms, block-9 161 ms). Worse, PLD is **not bit-lossless** on this MLX/gemma3n stack — block-forward numerics flip the argmax on near-ties (0.5-logit gap in the 262K head), silently dropping a clause on broad58 id53. **The real lever the sweep surfaced: the fixed v2 system prompt is ~75% of the prompt and its prefill is ~half of total latency**, so persistent `make_prompt_cache()` of it cuts mean 1044→~640 ms / median 928→~505 ms (~35%), fully lossless (bit-identical on all 58). Sub-500 ms is a *median* reality with caching alone. **Decision (user): defer wiring prompt-caching, focus on shrink first.** Tooling committed: `spoke/bench/pld_bench.py`, `pld_sweep.py`, `pld_cache_bench.py`.

### 5. Shrink round 2 + structural cuts + latency — DONE
- **g64 (2-bit emb group-64) = shrink DEPLOY PICK: 3.1 GB disk / 3.41 GB MLX-peak RAM / broad58 79.3%** — beats g32 (3.3 GB/78%) and the 3.9 GB baseline (76%) on both axes.
- **`mixed_3_6` body: DEAD END** — 3.1 GB / broad58 62.1% (3-bit body craters quality ~17 pts, and no size win because embeddings dominate). Body must stay 4-bit.
- **Vocab pruning (English-safe 165K, 63% kept): NOT WORTH IT** — 2.8 GB / 3.01 GB peak / broad58 72% (1 hard fail). Zero-OOV validated via byte-fallback, but -7 pts quality (79→72) for only 0.4 GB peak RAM + an English-lock. Byte re-decomposition of dropped tokens is the quality cost.
- **MatFormer E2B slice: IMPOSSIBLE for free** — Gemma 4 dropped MatFormer; the shipped E2B is a separately-trained model (hidden 1536 vs E4B's 2560), not sliceable from E4B. Under-2 GB requires a paid Modal fine-tune of `gemma-4-E2B`.
- **Real RAM footprint measured (MLX peak):** baseline 4.28 GB → g64 3.41 GB (~20% cut). Under-3 GB at full quality is NOT reachable by quantization/vocab-pruning; needs the E2B fine-tune.
- **Latency (clean, IDLE machine, lossless greedy + persistent system-prompt cache):** g64 = **median 488 ms / mean 628 ms / p90 1207 ms** (vs no-cache median ~928 ms). Prompt-caching ~halves the median and is THE latency lever; PLD stays rejected (unreliable — lossless on g64 but lossy on the baseline). The p90 tail (long outputs) is the only thing over the 800 ms ceiling. Prompt-caching is NOT yet wired — there's no serving shell to wire it into (TODO Track A). Deploy pick: **g64 + prompt-caching**.

## Key results table
| Model | Emb bits | Size | broad58 (local) | Hard fails | Heal loss |
|---|---|---|---|---|---|
| 2-bit emb + DWQ (g32) | 2 | 3.3 GB | 78% | 0 | 0.038 |
| 2-bit emb raw (g64, unhealed) | 2 | 3.1 GB | — | — | — |
| Baseline DWQ (deployed) | 4 | 3.9 GB | 76% | 0 | 0.039 |
| (ref) cloud bf16 champion | 16 | 15.9 GB | 82.8 (cloud grader) | 0 | — |

## Key learnings
- **Expert streaming is the wrong tool for a real-time, low-RAM narrow task.** It trades latency for memory; a hard <800 ms budget is its worst case. Diffuse routing (finding #99) removes the sparsity that would make it pay off. Streaming's niche is running models too big to fit at all, at 1–5 tok/s.
- **For a bandwidth-bound decode, size IS latency.** Shrinking the dense champion hits the RAM target and the latency target with one lever; serving flags (KV-quant, wired-limit, spec-decode) are near-no-ops at 4 GB / 20-token outputs.
- **DWQ defeats the PLE-quantization-garbage problem.** 2-bit embeddings (including the tied output-projection `embed_tokens`) healed to the same loss (0.038) as 4-bit and held broad58. The scariest risk was retired empirically.
- **Local vs cloud grader differ** (local reads 76–78 where cloud reads 82.8). Quant-vs-quant on the same local grader is the valid comparison; absolute champion-parity needs a cloud `benchmark.py` re-grade (TODO).
- **Subagents stall.** Both Opus execution agents ended their turns "waiting for a notification" that never re-wakes them. Fix that worked: the main loop took over via a detached driver script (`scratchpad/drive_shrink2.sh`) polling the heal PID, and stood the subagent down via SendMessage + TaskStop. DWQ heal processes launched detached (parent PID 1) survive TaskStop of the agent — safe to kill the stalled agent without killing the heal.

## Context for future
- **Next shrink lever (parked):** `mixed_3_6` body on top of 2-bit embeddings → target ~2.5 GB. Risk shifts to the capability-critical body; benchmark broad58 at each step. 3-bit embeddings are NOT a useful fallback (only reach ~3.5 GB); if 2-bit ever fails, protect `embed_tokens` (tied output proj) at higher bits while keeping PLE at 2-bit.
- **Validation TODO:** cloud `benchmark.py` re-grade of the 2-bit variant to confirm 82.8-parity in absolute terms; clean isolated latency measurement.
- **Artifacts:** 2-bit models at `spoke/models/g4e4b-emb2g32-dwq` (3.3 GB, gitignored) and `g4e4b-emb2g64-raw` (3.1 GB, unhealed); bf16 teacher `spoke/models/g4e4b-champion-bf16` (15 GB, KEEP for further DWQ); result JSONs `spoke/bench/result_g4e4b-emb2g32-dwq_v2.json`. Tooling: `spoke/shrink/asym_quant.py`, `ifp/stream_sim.py`.
- **llama.cpp is a dead end for us** (MLX 20–90% faster at this size; Gemma 4 PLE support DOES exist now — issue #22243 closed — so the ledger's "blocked" note is stale, but there's no reason to switch).
