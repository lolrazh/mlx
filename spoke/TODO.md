# Spoke — Inference Recipe & TODO Backlog

Companion to `LEDGER.md`. **LEDGER** = everything tried + numbered findings. **This doc** = (1) what MUST be applied at inference, and (2) what's left to build. Single durable home so decisions don't get buried in per-session logs.

> **Scope note (2026-07-13):** This now covers the WHOLE product — the Gemma 4 E4B text-cleaner AND the ASR front-end that feeds it AND the serving/app shell. A full audit of all 59 agent-logs + the ledger (4 review agents) found several *decided* production components with **zero tracking**. Items tagged **[audit]** were surfaced by that sweep; verify against the cited source before acting.

---

## 1. Inference-time recipe (apply ALL of these when serving/benchmarking the cleaner)

- **Model:** `spoke/models/g4e4b-champion-mlx-dwq4-g64` (deployed, 3.9 GB) — or the shrink candidate `g4e4b-emb2g32-dwq` (3.3 GB) once cloud-re-graded. Gemma 4 E4B, MatFormer + PLE, dense.
- **Prompt:** v2 system prompt, **greedy** (temperature 0).
- **⚠️ REQUIRED — no-think logit bias:** ban token ids **98 (`<|think|>`) + 100 (`<|channel>`)**, or broad58 drops ~17 pts. Auto-applied by `run_benchmark.py`'s `make_no_think_processors`. Finding #108.
- **Persistent system-prompt cache** *(decided, not yet wired — see Serving track)*: cache the fixed v2 system prompt once, feed only the per-request user suffix, trim back. ~35% latency cut (median ~500 ms), lossless. Ref impl: `spoke/bench/pld_cache_bench.py`. Finding #113. **⚠️ [audit] effect-size unresolved: 2026-07-08 log measured ~15%, 2026-07-13 measured ~35% — reconcile before citing a number.**
- **Do NOT use PLD** (prompt-lookup decoding): a bust on this 262K-vocab model and not bit-lossless. Finding #113.
- **generation_config eos:** union the tokenizer's own `eos_token_id` into the saved config. Finding #109. **⚠️ [audit] never re-verified end-to-end on the Gemma champion's OWN export path (provenance untraced) — audit the deployed artifact's `generation_config.json` before trusting it.**
- **[audit] Not in the recipe, evaluated as low-value:** KV-cache quant (`--kv-bits`, near-no-op for ~20-token outputs); speculative decoding via a Gemma-4-E2B 4-bit draft (shared 262144 vocab, the old "<0.5s lever" #108 — but finding #113's draft-overhead result predicts it's weak on a 4 GB model; test only if worthwhile).

---

## 2. Backlog

### Track A — Serving / app shell  **[BIGGEST GAP: no code exists yet]** [audit]
- [ ] **Build the persistent-process serving wrapper** (load model once, keep resident; "Handy-style `Arc<Mutex<Option<LoadedEngine>>>`" pattern, decided "the right pattern for Spoke" in 2026-03-04 inference-audit). All validation to date is `run_benchmark.py`, NOT a real server — this is what "deployment" actually requires.
- [ ] Wire **persistent system-prompt caching** into that server (the proven ~35% / median-500 ms win).
- [ ] (maybe) warm in-memory KV cache in the server.

### Track B — ASR front-end (decided prod component, zero integration) [audit]
- [ ] **RESOLVE THE DIRECTION (needs user decision):** the 2026-07-10 Nemotron log decided "Nemotron 3.5 ASR 0.6b, 8-bit, 320 ms lookahead" for **streaming**; but `moonshine/BENCHMARKS.md` lists **Cohere Transcribe 4-bit / Parakeet TDT 0.6b** for batch. Which is the pick (or is it streaming=Nemotron, batch=Cohere)?
- [ ] **License check — SHIP-BLOCKER:** Nemotron 8-bit OpenMDW vs NVIDIA Open Model License ambiguity; AND whether Cohere Transcribe's license permits app redistribution. Legal gate before shipping.
- [ ] Mirror the chosen 8-bit checkpoint to HF (`spokedotso/`).
- [ ] Wire the sidecar streaming protocol (start/audio-chunk/finalize + `NemotronEngine.stream()`); partials currently DROPPED at `sidecarEngine.ts:247`.
- [ ] Pill UI: live partial text + finalize animation.
- [ ] Validation: idle-machine 8-bit-vs-bf16 rematch + real short-dictation-audio latency (past benches ran under load / on 8 s LibriSpeech clips) + mic-mode subjective test + LibriSpeech test-other/noisy confirmation.

### Track C — Model shrink (Gemma cleaner) → 2–3 GB RAM (local, free)
- [ ] **Confirm g64 variant (3.1 GB):** DWQ-heal + broad58 bench. *(in progress)*
- [ ] ~~`mixed_3_6` body → 2.5 GB~~ — **[audit-updated] weak lever**: 2-bit embeddings dominate size, so mixed-body only trims ~0.28 GB (lands ~3.1–3.26 GB, not 2.5). Confirming, then likely shelve.
- [ ] **Vocab pruning → ~2.3 GB — now the PRIMARY path to 2.5 GB** (shrinks the 262K-vocab embedding tables proportionally). Needs a heal-tune, domain-locks the model. Prune→Distill→Quantize order.

### Track D — Accuracy / training / data [audit]
- [ ] **v6 training data** targeting weak categories (at-symbol/multi/spell/emoji) — ledger's own HIGH-priority "only remaining accuracy lever" (#90/#93/#107/#110). Never built.
- [ ] **emoji-ablation-confirm** (2–3 seeds) — gates what v6 does with emoji; a "+6 pts" claim was RETRACTED as noise (#105). Do before v6.
- [ ] **Structured edit-format / tool-call output** (SEARCH/REPLACE or JSON) — RESEARCH doc's "highest-conviction" unexplored idea; Phase-D trigger (SFT plateau) arguably met. Never tried.
- [ ] (small) Untested HP combo: LoRA+ (lr=1e-5) + dropout=0 stacked. (small) LoRA-GA gradient-approx init (2–4× convergence, one-time cost). (open) DWQ 1024-sample recalibration of the deployed 4-bit body.

### Track E — Validation
- [ ] **Cloud `benchmark.py` re-grade** of the shrink variant → confirm absolute 82.8-parity (local grader reads 76–78). COSTS MODAL $.
- [ ] **Clean latency measurement on an IDLE machine** — many past benches ran "under load," giving unreliable latency (the exact gauge problem). [audit]
- [ ] (low) expand core test set 23→50+; `speech-wer` for Apple SpeechAnalyzer/SpeechTranscriber.

### Track F — Infra / tech-debt (low priority; cloud path inactive) [audit]
- [ ] **Automate checkpoint-sweep benchmarking** — never trust eval_loss for selection (caused ~6 documented wrong-checkpoint picks). Lesson absorbed into local protocol (#107) but no tooling exists.
- [ ] safetensors integrity check (`safe_open`) + `--force` on `modal volume get` in `download_model.py` — a "lesson learned" (silent-corruption `!!!!` bug) never written back into the script.
- [ ] (dormant) Unsloth `save_pretrained_merged` config-mangling fix — worked-around, not fixed; bites if Unsloth/cloud conversion is revisited.
- [ ] (someday/side-quest) Qwen3-1.7B zero-shot baseline; T9 QLoRA; ministral-fix (`mistral3` text-only load); nemotron-midlr (lr window); user's standalone "MLX pruning library" repo.

### Closed / obsolete — do NOT re-litigate
- MoE expert-flashing + `ifp/` mass-weighted pruning — infeasible for Spoke's real-time/low-RAM budget (#111).
- Fine-tuning Gemma 4 26B-A4B; PLD (#113); llama.cpp (MLX faster).
- **[audit] superseded by the model/ASR swaps:** Qwen3/Qwen3.5 cloud+Unsloth training pipeline, DWQ-on-Qwen3-as-deploy, Qwen3-1.7B "small-fast-base" latency strategy, Moonshine MLX port Phase-3, Whisper mixed-quant sweep, GRPO/DPO RL (RESEARCH doc argues against for small models), Q1 mixed-bit on old Qwen, FA2-on-cloud.
