# Apple Intelligence Exploration: Foundation Models Benchmark, Adapter Path, IFP Idea

**Date:** 2026-07-06
**Agent:** Claude Fable 5
**Status:** Completed (exploration + baselines; adapter probe queued pending toolkit download)

## User Intention
"Doesn't Apple have an on-device LLM? Can we try that?" → benchmark Apple's Foundation Model with all prompt variants, measure RAM and latency, map the Apple Intelligence model lineup, figure out how adapter fine-tuning access works, and evaluate whether the AFM 3 Core Advanced NAND-streaming architecture could be replicated on an open MoE (Gemma 4 26B A4B).

## What We Accomplished
- **Built access tooling:** `spoke/bench/apple_fm_shim.swift` (CLI around `LanguageModelSession`, one fresh session per item, `permissiveContentTransformations` guardrails, JSON-lines stdin/stdout) + `spoke/bench/apple_fm_benchmark.py` (drives shim, reuses `run_benchmark.py` scoring, `--prompt-mode` for all five variants). Compiled against macOS 26.2's system `FoundationModels.framework`.
- **Full prompt sweep** (B28–B34) + **RAM/latency measurement** via inference-daemon RSS and system wired-memory delta.
- Researched: WWDC26 AFM 3 lineup, adapter training toolkit access mechanics, macOS 27 `fm` CLI/Python SDK/provider architecture, SpeechAnalyzer/SpeechTranscriber, IFP paper.

## Results (zero-shot = no fine-tuning; all local, free)

| Prompt | core23 | broad58 (clean) | Refusals |
|--------|--------|-----------------|----------|
| mini | 13% | — | 8/23 |
| generic | 22% | — | 3/23 |
| v2 | 22% | 12% | 3/23, 10/58 |
| v3 | 22% | — | **1/23** |
| spoke-full | ~~30%~~ (leaked 4/23) | **21%** | 2/23, 6/58 |

- **Refusals scale inversely with prompt richness** — rule-heavy prompts reassure the safety layer. 10/58 refusals on realistic dictation (v2) is production-disqualifying zero-shot.
- **RAM (measured):** inference host idles at 82 MB (demand-loaded); process RSS peaks 245 MB (lies — weights mmapped); **true cost = +2.25 GB wired peak** (2.58→4.83 GB), settling to +1.6 GB cached. Same class as DWQ Qwen (2.1 GB) but OS-managed/shared/evictable — marginal cost ≈ 0 if user already runs Apple Intelligence.
- **Latency:** 0.35s avg (v2) / 0.86s (spoke-full ~1k tok) — prompt-length-bound. Fastest latency ever measured in this project.
- Guardrails run in a **separate process** (`GenerativeExperiencesSafetyInferenceProvider`) — weak evidence adapters won't bypass refusals.

## Apple Intelligence lineup (AFM 3, WWDC26)
- **AFM 3 Core** (3B dense, on-device — what macOS 26.2 exposes today, prev gen).
- **AFM 3 Core Advanced** (20B sparse on-device: full weights in NAND, 1–4B active experts loaded to DRAM per prompt via Instruction-Following Pruning; natively multimodal; Apple's evals prefer it for dictation 44.7% vs 17.6%; needs M3+/A19 Pro; **macOS 27 only — untestable on 26.2**; dual-boot beta on a separate APFS volume is the only near-term path).
- **AFM 3 Cloud / Cloud Pro** on Private Cloud Compute — **FREE for Small Business Program devs <2M downloads**.
- Framework opens to any provider (custom MLX models, Anthropic, Google) behind the same `LanguageModelSession` — it's a deployment surface, not a model. macOS 27 adds `fm` CLI + Python SDK.

## Adapter fine-tuning access (verified)
- **The toolkit download CONTAINS the base model weights** (training-only license). No entitlement to train/test; production needs `com.apple.developer.foundation-model-adapter`.
- PyTorch, Python 3.11+, **Linux GPU supported → Modal pipeline works**; JSONL chat format ≈ v5-split as-is; optional draft-model training → speculative decoding; exports ~160 MB `.fmadapter`, ships via Background Assets.
- Version-pinned per OS (toolkit 26.0.0 final for OS 26; retrain each OS cycle — user accepts this).
- **Decisive experiment queued (afm-adapter):** one v5-split adapter answers (a) does 21% broad58 → ~85% 3B-class, (b) do refusals survive fine-tuning.

## SpeechAnalyzer / SpeechTranscriber (native ASR)
New on-device ASR API since macOS 26 (WWDC25); powers Notes/call transcription; developer-accessible and reportedly *better than* consumer Dictation (which hadn't switched over). 2.2x faster than Whisper Large V3 Turbo, quality "comparable" but **no published WER** — queued (speech-wer) to benchmark on LibriSpeech with the moonshine harness next to Granite NAR (1.21%)/Parakeet (1.93%)/shipped Whisper turbo (2.66%). Endgame architecture: SpeechTranscriber → Spoke FM adapter = all-Apple zero-download dictation pipeline.

## The IFP idea (side-project candidate — user excited)
- Technique published: **arXiv 2501.02086** (Apple+UCSB, ICML 2025) — sparsity predictor reads the *prompt*, emits FFN pruning masks, dense sub-model runs whole generation. No code/weights released. Pairs with Apple's "LLM in a Flash".
- **Model-specific, not engine-specific**: needs training for prompt-stable routing. Engine half is nearly free on Apple Silicon (MLX mmap + unified memory = untouched experts never page in).
- Gemma 4 26B A4B is per-token MoE → naive retrofit fails (expert union → full model, finding #31). **BUT Spoke is IFP's best case**: one fixed task, fixed prompt, narrow inputs → routing likely highly concentrated.
- **Effort ladder (queued as ifp-lite):**
  1. (~day) Expert-locality profiling: Gemma 4 A4B 4-bit (~14.4 GB, fits 24 GB M4 batch-1) in MLX, log router decisions over v5 inputs, histogram expert usage. mmap residency itself measures locality.
  2. (~week) Domain pruning: drop cold experts → permanently shrunk Spoke-domain model, validate on broad58.
  3. (research) Trained mask predictor à la the paper — nobody has open-sourced this; genuinely high-value OSS.

## Context for Future
Stopped here by user request. Queue now holds: **afm-adapter (HIGH, blocked on user downloading toolkit from Apple Developer)**, speech-wer, ifp-lite, plus the standing v6-data entry. All Apple FM results in LEDGER B28–B34 + finding #98. Shim binary rebuilds from `spoke/bench/apple_fm_shim.swift` (see header). Terminology reminder: "zero-shot" = no fine-tuning per CLAUDE.md; all sweep runs used system prompts.
