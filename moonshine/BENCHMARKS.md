# ASR Benchmark Results (Spoke model evaluation)

Consolidated results for every ASR model evaluated for Spoke's local dictation.
All canonical runs: **LibriSpeech test-clean, 250 samples, shuffled seed 42, M4 24 GB**.
Raw per-run JSON (including per-sample WER) lives in `moonshine/results/`.

- **WER**: word error rate, lower is better.
- **RTF**: real-time factor, inference seconds per audio second, lower is faster.
- **Peak RAM**: `mlx_peak_memory_gb` measured during inference, after model load (steady-state weights + activations). Older runs marked `n/a` predate this metric; the `peak_memory_gb` field in those JSONs is tracemalloc noise, ignore it.

## Standings (2026-07-10)

| Model | Quant | WER | RTF | Peak RAM | Loader / notes |
|---|---|---|---|---|---|
| Cohere Transcribe 4-bit **(SHIPPED)** | 4-bit g64 full | 1.28% | 0.05 | 1.89 GB | mlx-speech, Python 3.13, 14 languages |
| Granite 4.1-2b NAR 6-bit | 6-bit full | 1.25% | 0.073 | 2.66 GB | stock mlx-audio; most accurate overall |
| Granite 4.1-2b NAR 5-bit | 5-bit decoder-only | 1.21% | 0.056 | 3.26 GB | encoder bf16, hence the RAM |
| Granite 4.1-2b NAR 4-bit | 4-bit full | 1.43% | 0.059 | 2.19 GB | ~1.7 GB on dictation-length clips; 2.19 is one 28 s outlier |
| Granite 4.1-2b AR 4-bit | 4-bit full | 1.47% | 0.118 | 5.83 GB | AR decode activations dominate; dead end for RAM |
| Granite 4.0-1b 4-bit | 4-bit decoder-only | 1.60% | 0.345 | 6.27 GB | first Granite attempt, encoder left bf16; superseded |
| **Parakeet TDT 0.6b v2 6-bit** | 6-bit g64 (Linear/Embedding) | 1.87% | 0.022 | 1.13 GB | matches bf16 WER exactly; fastest run on record; sweet spot |
| Parakeet TDT 0.6b v2 8-bit | 8-bit g64 (Linear/Embedding) | 1.91% | 0.024 | 1.26 GB | parakeet-mlx, Python 3.11, English-only; first shipped config |
| Parakeet TDT 0.6b v2 4-bit | 4-bit g64 (Linear/Embedding) | 2.02% | 0.033 | 1.01 GB | smallest (445 MB) but slower: MLX 4-bit matmul underperforms here |
| Parakeet TDT 0.6b v2 bf16 | none | 1.87% | 0.030 | 2.10 GB | quantization is free down to 6-bit |
| Parakeet TDT 0.6b v3 bf16 | none | 1.93% | 0.034 | n/a | 25 languages; matches official leaderboard exactly |
| Moonshine Medium (MLX port) | fp16 | 2.19% | 0.042 | n/a | our from-scratch port |
| Qwen3-ASR-1.7B 8-bit | 8-bit decoder-only | 2.30% | 0.204 | n/a | mlx-audio; encoder bf16 (hardcoded exclusion) |
| Nemotron Speech EN 0.6b bf16 | none | 2.45% | 0.02 | 3.09 GB | nemotron-asr-mlx port; STREAMING; English-only; RAM is batch mode, unquantized |
| Whisper Turbo 4-bit **(previous default)** | 4-bit | 2.66% | 0.298 | 1.06 GB | mlx-whisper, 99 languages |
| Whisper Large v3 | fp16 | 3.02% | 0.344 | n/a | reference ceiling model |
| Nemotron 3.5 ASR 0.6b bf16 | none | 3.23% | 0.157 | 1.89 GB | mlx-audio git main (0.4.5); STREAMING; 40 languages; lookahead 1120 ms |
| Nemotron 3.5 ASR 0.6b 8-bit **(streaming pick)** | 8-bit g64 (Linear/Embedding) | 3.32% | 0.231 | 0.89 GB | measured at the 320 ms streaming lookahead; punctuation intact; 756 MB disk |
| Nemotron 3.5 ASR 0.6b 6-bit | 6-bit g64 (Linear/Embedding) | 3.25% | 0.095 | 0.77 GB | WER fine but LOSES punctuation/caps at 320 ms lookahead; do not ship |

Whisper mixed-quant sweep results (25-sample exploratory runs) are in `results/whisper-*-mixed-*` and `results/whisper-mixed-quant-sweep-25.json`; none beat the shipped turbo 4-bit meaningfully.

## Streaming (Nemotron, 2026-07-10)

The two Nemotron models are the only true streaming entries (cache-aware FastConformer-RNNT: each frame encoded once, partials emitted per chunk). Everything else in the table is batch-only. WER for them is measured on the same streaming compute path used live; there is no separate "offline mode" inflating the numbers.

Prototype: `moonshine/streaming/nemotron_stream_proto.py` (paced-file and `--mic` modes). Paced-clip results for Nemotron 3.5, 48.7 s LibriSpeech concat, machine under load:

| Lookahead | Partial cadence | Text lag (p50) | Time-to-final | Notes |
|---|---|---|---|---|
| 80 ms | n/a | drifts unboundedly | n/a | per-chunk overhead exceeds budget; not viable via Python |
| **320 ms** | ~3/s | ~130-200 ms + up to one chunk | **90-215 ms** | operating point for dictation |
| 560 ms | ~2/s | ~205 ms | ~195 ms | |
| 1120 ms | ~1/s | ~245 ms | ~145 ms | best WER (3.23% vs 3.45% at 320 ms, 6-bit) |

Key properties: partials are **append-only** (greedy RNNT never retracts; 0 prefix violations in all runs), punctuation/capitalization native. Error analysis (250 ref/hyp pairs, bf16): only 3/171 error words were normalization artifacts, 0 digit errors; the WER gap vs Parakeet is real (limited lookahead context + multilingual tax) and skews to rare proper nouns.

Decision 2026-07-10: **ship Nemotron 3.5 8-bit at 320 ms lookahead** for streaming dictation (multilingual, RAM-lightest, punctuation survives). The English-only model is more accurate (2.45%) but needs a different loader (see row) plus quantization work for RAM; revisit if English accuracy becomes the priority. 8-bit vs bf16 rematch pending on an idle machine.

## Reference numbers (external, for sanity checks)

- Parakeet v2 official Open ASR Leaderboard: test-clean 1.69%, test-other 3.19%, avg 6.05%.
- Parakeet v3 official: test-clean 1.93% (our run reproduced this exactly), avg 6.34%. v2 beats v3 on 6 of 8 English sets; v3 adds 25 languages.
- Higgs Audio v3 STT (bosonai, not yet benchmarked): Boson claims 1.55% test-clean. 2.68B params, MLX port exists only in mlx-audio git main (unreleased as of 2026-07-09), no prequantized checkpoint anywhere, estimated 3-4 GB peak at 4-bit. Parked.

## Lessons that keep repeating

1. **"4-bit" checkpoints usually mean decoder-only.** mlx-audio hardcodes encoder exclusions (Qwen3-ASR `audio_tower`, same for Higgs); the first Granite run peaked at 6.27 GB because of this. Always check the safetensors index for `.scales` on encoder weights, and quantize the encoder yourself when RAM matters (see `baselines/granite_nar_convert.py`).
2. **Peak RAM must be measured, not inferred from disk size.** Activation memory scales with clip length (Granite NAR) or decode strategy (Granite AR).
3. **Quantization is nearly free down to 6-bit.** Parakeet v2 6-bit matches bf16 WER exactly at half the RAM; 8-bit costs +0.04pp; 4-bit costs +0.15pp AND runs slower than 6-bit (MLX 4-bit matmul throughput). Sub-4-bit or aggressive mixed quant was never worth it (Whisper sweep).
4. **RTF ranking: transducer (Parakeet 0.024-0.034) < NAR decoder (Cohere/Granite 0.05-0.07) << AR decoder (Qwen 0.2, Whisper 0.3).**
5. **WER survives quantization; punctuation may not.** Nemotron 3.5 6-bit matched 8-bit/bf16 on WER but silently dropped punctuation/capitalization at the short streaming lookahead (fine at 1120 ms). Punctuation tokens are marginal predictions; always eyeball formatted output at the target operating point, the WER harness normalizes it away.

## Reproducing

```bash
cd ~/Documents/Projects/mlx
.venv/bin/python -m moonshine.baselines.<baseline> --max-samples 250
# Cohere needs the Python 3.13 venv:
.venv-cohere/bin/python -m moonshine.baselines.cohere_baseline --max-samples 250
# Nemotron needs mlx-audio git main (Python 3.13 venv):
.venv-nemotron/bin/python -m moonshine.baselines.nemotron_baseline --bits 8 --lookahead 320 --max-samples 250
```

One baseline file per model in `moonshine/baselines/`. Self-built checkpoints live in `moonshine/models/` (gitignored); conversion recipes in `moonshine/recipes/` and the `*_convert.py` baselines. The Parakeet v2 8-bit checkpoint is rebuilt with `moonshine/recipes/parakeet/quantize_v2_8bit.py`.
