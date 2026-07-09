# Nemotron Streaming ASR: prototype + quant sweep + English-only bake-off

**Date:** 2026-07-09/10
**Agent:** Claude Fable 5
**Status:** DONE for the research phase. Decision: ship **Nemotron 3.5 ASR 0.6b, 8-bit g64, 320 ms lookahead** as Spoke's streaming dictation model. App integration (sidecar streaming protocol + pill partial-text UI) is the next phase, not started.

## Why

Spoke wants true streaming dictation: live partials in the pill while speaking, instant finalize on key-release. Every model shipped so far (Whisper/Cohere/Parakeet) is batch-only; the app transcribes once after key-release. NVIDIA's Nemotron speech models are the first viable cache-aware streaming family with an MLX path.

## The models

- `nvidia/nemotron-3.5-asr-streaming-0.6b`: cache-aware streaming FastConformer-RNNT, 40 language-locales via language-ID prompt (auto-detect supported), native punctuation/caps, runtime lookahead knob `att_context_size` ([56,0]=80 ms, [56,3]=320 ms, [56,6]=560 ms, [56,13]=1120 ms). MLX support in **mlx-audio git main only** (pinned 04151c6 = v0.4.5, Python 3.13 OK); bf16 checkpoint `mlx-community/nemotron-3.5-asr-streaming-0.6b`.
- `nvidia/nemotron-speech-streaming-en-0.6b`: English-only sibling, no prompt layer. NOT loadable by mlx-audio (config predates the 3.5 arch); benchmarked via pip `nemotron-asr-mlx` 0.2.0 (weights `dboris/nemotron-asr-mlx`). This is the exact checkpoint Handy ships as GGUF Q8_0 (`handy-computer/nemotron-speech-streaming-en-0.6b-gguf`).

## Results (canonical harness: LibriSpeech test-clean, 250 samples, seed 42; machine under load)

| Config | WER | RTF | Peak RAM |
|---|---|---|---|
| 3.5 bf16, la1120 | 3.23% | 0.157 | 1.89 GB |
| 3.5 6-bit, la1120 | 3.25% | 0.095 | 0.77 GB |
| 3.5 6-bit, la320 | 3.45% | 0.192 | 0.76 GB |
| **3.5 8-bit, la320 (PICK)** | **3.32%** | 0.231 | **0.89 GB** |
| EN bf16 (nemotron-asr-mlx, greedy) | 2.45% | 0.02 | 3.09 GB |

Streaming prototype (`moonshine/streaming/nemotron_stream_proto.py`, paced 48.7 s clip + `--mic` mode): partials **append-only** in every run (greedy RNNT never retracts, 0 prefix violations = no pill flicker), ~3 partials/s at la320, text lags live speech ~130-200 ms plus up to one 320 ms chunk, **time-to-final 90-215 ms** after audio ends (the "whoosh" number). **la80 cannot sustain real-time** through Python (per-chunk overhead > 80 ms budget; lag drifts unboundedly).

## Findings worth remembering

1. **6-bit silently destroys punctuation/capitalization at la320** while WER stays fine (fine at la1120, fine at bf16 la320). Punctuation tokens are marginal at short lookahead; quant noise tips them. WER harness normalizes punctuation away, so this is invisible in the table. 8-bit is the floor for streaming quants of this model.
2. **The WER is real, not a harness artifact** (user asked): `generate()` runs the same cache-aware streaming path as live streaming (verified char-identical outputs); ref/hyp error analysis found 3/171 error words were normalization artifacts, 0 digit errors. The gap vs Parakeet (1.87%) is architectural (limited lookahead) + multilingual tax, skewed to rare proper nouns.
3. English-only model halves the gap (2.45%, beats shipped Whisper 2.66%) but needs a second inference lib and RAM work (3.09 GB bf16 batch mode). Parked; multilingual+streaming won.
4. mlx-audio's port exposes exactly what an app needs: `stream_encode_chunks` accepts an arbitrary mel-chunk iterator (live feed friendly), `log_mel_spectrogram_frames` gives frame-exact incremental mel, decode is resumable per chunk.

## Artifacts

- `moonshine/streaming/nemotron_stream_proto.py`: streaming feasibility probe (paced file + mic), reports lag/cadence/prefix-stability/time-to-final. Stats JSONs in `moonshine/streaming/`.
- `moonshine/recipes/nemotron/quantize.py`: quant recipe (mlx_lm quantize_model, convert.py-compatible; output loads with stock `mlx_audio.stt.utils.load_model`).
- `moonshine/baselines/nemotron_baseline.py` (`--bits`, `--lookahead`), `moonshine/baselines/nemotron_en_baseline.py`.
- Checkpoints in `moonshine/models/` (gitignored): `nemotron-3.5-asr-streaming-0.6b{,-6bit,-8bit}`. 8-bit = 756 MB disk.
- Venv: `.venv-nemotron` (Python 3.13; mlx-audio @ 04151c6, datasets==3.6.0, librosa, jiwer, nemotron-asr-mlx, sounddevice).
- Result JSONs + ref/hyp pairs in `moonshine/results/nemotron-*`.

## NEXT (Spoke integration, needs its own session)

1. Mirror the 8-bit checkpoint to `spokedotso/` on HF. **Check license first**: 3.5 card shows OpenMDW vs mlx-community README says NVIDIA Open Model License; read the actual LICENSE file.
2. Sidecar: streaming request mode (start/audio-chunk/finalize framing) + `NemotronEngine.stream()`; partials are currently dropped at `sidecarEngine.ts:247`.
3. Pill UI: live partial text + finalize animation (design conversation first; monochrome system, existing pill patterns).
4. Pending rematches: 8-bit vs bf16 on an idle machine; mic-mode subjective testing (`--mic` works, needs terminal mic permission).
