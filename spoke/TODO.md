# Spoke — Inference Recipe & TODO Backlog

Companion to `LEDGER.md`. **LEDGER** = everything we tried + numbered findings. **This doc** = (1) what MUST be applied at inference/serving time, and (2) what's left to build. Keep it current — this is the single place we remember the deployment recipe and the backlog so it doesn't get buried in per-session logs.

---

## 1. Inference-time recipe (apply ALL of these when serving or benchmarking the deployed model)

- **Model:** `spoke/models/g4e4b-champion-mlx-dwq4-g64` (deployed, 3.9 GB) — or the shrink candidate `g4e4b-emb2g32-dwq` (3.3 GB) once cloud-re-graded. Gemma 4 E4B, MatFormer + PLE, dense.
- **Prompt:** v2 system prompt, **greedy** (temperature 0).
- **⚠️ REQUIRED — no-think logit bias:** ban token ids **98 (`<|think|>`) + 100 (`<|channel>`)**, or broad58 drops ~17 pts (the model slips into its hidden thinking channel). Auto-applied by `run_benchmark.py`'s `make_no_think_processors`. Finding #108.
- **Persistent system-prompt cache** *(decided, not yet wired — see backlog)*: cache the fixed v2 system prompt once (`make_prompt_cache`), feed only the per-request user suffix, trim back after each request. ~35% latency cut (median ~500 ms), fully lossless. Proven approach: `spoke/bench/pld_cache_bench.py`. Finding #113.
- **Do NOT use PLD** (prompt-lookup decoding): a bust on this 262K-vocab model and *not* bit-lossless (near-tie argmax flips). Finding #113.
- **generation_config:** always union the tokenizer's own `eos_token_id` into the saved config (guards the recurring runaway-generation bug). Finding #109.

---

## 2. Backlog / next actions (priority order)

### Shrink — hit the 2–3 GB RAM target (local, free)
- [ ] **Confirm g64 variant (3.1 GB):** DWQ-heal `g4e4b-emb2g64-raw` + broad58 bench. Likely the better deploy pick than g32 if quality holds. *(in progress)*
- [ ] **`mixed_3_6` body + 2-bit embeddings → ~2.5 GB:** 3-bit body / 6-bit sensitive layers, on top of 2-bit embeddings, DWQ-heal, bench broad58. Risk shifts to the capability-critical body — watch broad58. *(in progress)*
- [ ] *(stretch)* **Vocab pruning → ~2.3 GB:** Spoke's vocab is narrow, and pruning it shrinks PLE + token tables proportionally. Needs a heal-tune and domain-locks the model. Only if pushing past 2.5 GB. Must come BEFORE quantization (Prune→Distill→Quantize).

### Validation
- [ ] **Cloud `benchmark.py` re-grade of the 2-bit variant** → confirm absolute 82.8-parity. The local grader only reads 76–78; we can't claim true champion-parity without this. **COSTS MODAL $** (small).
- [ ] **Clean isolated latency measurement** of the 3.3 GB variant (the ~40% speedup seen so far was from a noisy benchmark log).

### Latency / serving (deferred by user — do after shrink)
- [ ] **Wire persistent system-prompt caching as the default** in `run_benchmark.py` and the inference/serving path. Proven ~35% (median ~500 ms), lossless. Reference impl: `spoke/bench/pld_cache_bench.py`. Deferred 2026-07-13.

### Closed — no action needed
- **MoE expert-flashing** — infeasible for Spoke's real-time / low-RAM budget (diffuse routing). Finding #111.
- **Fine-tuning Gemma 4 26B-A4B** — moot; only made sense if streaming worked.
- **PLD** — bust; use prompt-caching instead. Finding #113.
- **llama.cpp** — dead end (MLX 20–90% faster at this size; Gemma 4 PLE support exists now, #22243 closed, but irrelevant).
