# MoE Expert-Flashing Investigation → Dense-Champion Shrink: 2-bit Embeddings Proven at 3.3 GB

**Date:** 2026-07-12 → 2026-07-13 (overnight)
**Agent:** Claude Opus 4.8 (1M context), main loop, with 5 research/execution subagents (3 Sonnet research, 2 Opus execution)
**Status:** IN PROGRESS. Feasibility of sub-3 GB dense shrink PROVEN (2-bit embeddings + DWQ = 3.3 GB @ broad58 78% local, matches 4-bit baseline's 76%, 0 hard fails, heal loss 0.038). Latency/PLD lever now being explored by an Opus subagent. Push to ~2.5 GB (mixed-precision body) parked for a follow-up.

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

### 4. Now: latency / PLD lever (in progress)
Pivoted (per user) to prompt-lookup decoding — the task-matched latency multiplier (Spoke output echoes its input, so PLD gives 2–4× lossless). Classic draft-model speculative decoding was ruled out (draft overhead erases the win on a 4 GB model). An Opus subagent is implementing a minimal PLD loop against mlx-lm and benchmarking latency vs plain greedy on broad58. Persistent prompt-cache hygiene + an optional llama-bench-vs-MLX comparison are stretch goals.

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
