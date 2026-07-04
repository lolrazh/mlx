# ASR Frontend Scouting: Granite Speech + Cohere Transcribe vs Whisper Turbo

**Date:** 2026-06-25
**Agent:** Backfilled 2026-07-04 by Claude Fable 5 (reconstructed from scripts + result JSONs; the original session was never logged or committed)
**Status:** Completed (uncommitted until 2026-07-04)

## User Intention
Scout replacement ASR frontends for the Spoke app. The app ships Whisper large-v3 turbo 4-bit; this session benchmarked IBM Granite Speech 4.0/4.1 and Cohere Transcribe (03-2026) against it on LibriSpeech test-clean (250 samples) using the existing `moonshine/eval` WER pipeline.

## What Was Built
- `whisper_turbo_4bit_baseline.py` — CONTROL. Points mlx_whisper at the turbo 4-bit repo, byte-identical to the spokedotso fork shipped in the app.
- `granite_baseline.py` — Granite 4.0 1B Speech (4-bit, mlx-audio `granite_speech`). Needs mlx-audio >= 0.4.0. Must NOT pass `language=` (that builds a translate prompt); rely on the default transcription prompt.
- `granite_ar_baseline.py` + `granite_ar_convert.py` — Granite Speech 4.1 2B autoregressive. The 4.0 AR loader also loads 4.1-2b. Converter produces TRUE full quantization (encoder + projector + LM all quantized, not just the LM).
- `granite_nar_baseline.py` + `granite_nar_convert.py` — Granite Speech 4.1 2B NAR (Conformer CTC encoder + projector + bidirectional editor, single-pass non-autoregressive). Stock mlx-community 5-bit only quantizes the editor (encoder/projector stay bf16, ~3.3 GB peak); converter dequantizes and re-quantizes ALL Linears uniformly to 4/6-bit to get under 2 GB.
- `cohere_baseline.py` — Cohere Transcribe 03-2026. Requires the DEDICATED `.venv-cohere` (mlx-speech, Python >= 3.13, mlx >= 0.31) — the main `.venv`/mlx-audio silently drops decoder weights and produces garbage. No native-layout 4-bit exists on HF, so the script builds one (dequantize native 8-bit → re-quantize with mlx-speech's own quantizer, group_size 64), cached at `moonshine/models/cohere-transcribe-mlx-4bit`.

## Results (LibriSpeech test-clean, 250 samples, 5,297 words)

| Model | WER | RTF | MLX peak mem | Notes |
|-------|-----|-----|--------------|-------|
| **granite-speech-4.1-2b-nar-5bit (stock)** | **1.21%** | **0.056** | 3.26 GB | Best WER. Editor-only quant, encoder bf16. |
| granite-nar-6bit-full | 1.25% | 0.073 | 2.66 GB | Full quant, near-stock accuracy. |
| **cohere-transcribe-4bit (custom build)** | **1.28%** | **0.050** | **1.89 GB** | Best under 2 GB. Fastest RTF. |
| granite-nar-4bit-full | 1.43% | 0.059 | 2.19 GB | Full 4-bit costs 0.22 pts vs 6-bit. |
| granite-4.1-2b-ar-4bit-full | 1.47% | 0.118 | 5.83 GB | AR 2x slower than NAR, worse WER. |
| granite-4.0-1b-speech-4bit | 1.60% | 0.345 | 6.27 GB | Old 4.0. Slow (LLM decoder). |
| **whisper-turbo-4bit (app control)** | **2.66%** | 0.298 | 1.06 GB | What the Spoke app ships today. |

## Key Learnings
- **Every candidate roughly halves the shipped Whisper Turbo's WER.** Best (Granite NAR 5-bit) is 2.2x better at 5x the speed.
- **NAR > AR for Granite 4.1**: single-pass editor beats autoregressive decoding on both WER (1.21 vs 1.47) and RTF (0.056 vs 0.118).
- **Under the 2 GB app budget, Cohere Transcribe 4-bit wins**: 1.28% WER, RTF 0.050, 1.89 GB peak — but needs the mlx-speech stack (separate venv, Python 3.13).
- **Granite NAR full 6-bit is the compromise** if staying on mlx-audio: 1.25% WER at 2.66 GB.
- mlx-audio vs mlx-speech loader mismatch fails SILENTLY (token salad, no error) — always smoke-test transcripts, not just load success.

## Context for Future
These results position the ASR frontend upgrade for the Spoke app. Open follow-ups: LibriSpeech test-other / noisy-set confirmation, latency on real dictation audio (short utterances, not 8s LibriSpeech clips), and whether Cohere's license permits app redistribution.
