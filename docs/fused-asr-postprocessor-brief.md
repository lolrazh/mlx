# Fused ASR + Post-Processor: Research Brief

**Date:** 2026-03-04
**Status:** Early research / architecture exploration
**Hardware target:** Apple M4, 24 GB unified memory, 120 GB/s bandwidth

---

## 1. The Problem

Current dictation apps have a multi-stage pipeline:

```
Audio → ASR model → raw transcript → LLM post-processor → corrected text
```

Each stage adds latency. On consumer hardware (a MacBook), running a full-size ASR model (Whisper Large v3, ~1.5B params) followed by a full-size LLM (4B params) means **two sequential model inferences per utterance**. Combined latency is 1-3 seconds even with quantization, which breaks the real-time feel of dictation.

Beyond latency, there are deeper problems:

### 1a. Phonetic Ambiguity

ASR models transcribe phonemes to the most statistically likely spelling. When a user says "Groq" (a cloud inference company), Whisper writes "Grok" (the more common spelling). The post-processor can fix this IF given a dictionary — but the dictionary has to be passed as text in the prompt, which wastes context window and doesn't scale past a few hundred entries.

### 1b. In-Speech Meta-Commands

Users naturally disambiguate in speech the same way they would with a human scribe:
- "Groq with a Q" → the post-processor must understand this means "respell Groq using Q instead of K"
- "Sandeep with an H" → insert H into the name
- "I mean Tuesday, not Monday" → self-correction, replace Monday with Tuesday
- "quote ... end quote" → wrap in quotation marks
- "all caps" → convert to uppercase

These are meta-linguistic commands embedded in the speech stream. The ASR model transcribes them literally. A separate post-processor must parse and execute them. This works, but adds a full LLM inference to every utterance.

### 1c. Real-Time Learning

Dictation apps need to adapt to each user's vocabulary over time. When a user corrects "Grok" → "Groq" on their keyboard after transcription, the system should learn this preference. Current approaches (like Wispr Flow) watch keyboard edits and maintain a dictionary. But this dictionary is just a flat word list biased into the ASR prompt — it doesn't deeply integrate with the model's understanding.

### 1d. Screen Context

The user's current screen contains valuable context. If they're writing an email to "Sandeep" and dictate the name, the system should prefer that spelling. Currently this requires: screenshot → OCR → extract words → inject into ASR prompt and/or LLM prompt. Multiple stages, each adding latency and failure modes.

---

## 2. The Application

We are building a **local-first dictation engine for macOS** that runs entirely on-device (Apple Silicon). The requirements are:

| Requirement | Target | Why |
|---|---|---|
| **End-to-end latency** | < 500ms after utterance | Real-time dictation feel |
| **Memory footprint** | < 8 GB total | Must coexist with user's apps on 24 GB machine |
| **Accuracy** | Match or exceed Whisper Large v3 on user's vocabulary | No point if it's worse than existing tools |
| **Extensibility** | Add new vocabulary without retraining | Users add words daily |
| **Learning** | Improve from corrections over time | The Tuna launcher principle — frequency + recency |
| **Privacy** | Fully on-device | No cloud API calls for transcription |

The application handles:
- **General dictation** — everyday speech to text
- **Technical vocabulary** — product names (Groq, Supabase, Svelte), people's names, jargon
- **Formatting commands** — quotes, caps, emphasis, emoji, code-style identifiers (camelCase)
- **Self-corrections** — "I mean X not Y", "actually, change that to Z"
- **Phonetic disambiguation** — "X with a Q", "Y as in Zulu"

---

## 3. What We've Built So Far

### 3a. ASR Baseline (Whisper)

- **Whisper Large v3** running on MLX (Apple's ML framework): 3.02% WER on LibriSpeech test-clean
- **Moonshine Medium** (245M params, PyTorch+MPS): 2.19% WER, 3x faster than Whisper
- Whisper accepts a text prompt for vocabulary bias — we feed it OCR-extracted screen words

### 3b. Post-Processing LLM (Fine-tuned)

We fine-tuned multiple models using LoRA (Low-Rank Adaptation) on ~1,200 training examples covering 9 trigger categories (spelling, self-correction, quotes, caps, emphasis, emoji, camelCase, @-symbols, multi-operation).

**Results on 23-example test set (v3), all using same v4 training data + 2000 iterations:**

| Model | Params | Accuracy | Latency (bf16) | DWQ 4-bit Latency |
|---|---|---|---|---|
| Qwen3-4B | 4.4B | 100% | 1.82s | 0.88s |
| Llama 3.2 3B | 3.2B | 91% | 1.90s | ~0.6-0.7s (est.) |
| Gemma 3 4B | 4.6B | 87%* | 2.52s | ~1.1s (est.) |

*Gemma only tested with older/smaller training data — likely undertested.

**Current deploy model:** Qwen3-4B, DWQ 4-bit quantized, 2.1 GB, 0.88s latency, 96% accuracy.

### 3c. The Gap

The test set is only 23 examples. In real-world usage, the model fails on:
- Multi-word product names ("Whisper Flow" → drops "Flow")
- Phonetic disambiguation without dictionary context
- Novel vocabulary not in training data
- Edge cases around correction scope

100% on 23 examples is meaningless. Real-world accuracy is significantly lower.

---

## 4. The Proposed Architecture

### 4a. Core Insight

Both ASR and post-processing produce text token-by-token using a decoder. The only reason they're separate models is that they were trained separately in different embedding spaces. If we bridge those spaces with a small adapter, we can fuse them into one model.

### 4b. Architecture

```
                    ┌──────────────────┐
Audio Waveform ───→ │  Audio Encoder   │ (frozen, ~100-300M params)
                    │  (Whisper/       │
                    │   Moonshine)     │
                    └────────┬─────────┘
                             │ audio embeddings
                             ▼
                    ┌──────────────────┐
                    │  Tiny Adapter    │ (trainable, ~10-50M params)
                    │  (linear proj   │
                    │   or small MLP)  │
                    └────────┬─────────┘
                             │ projected embeddings
                             ▼
                    ┌──────────────────┐
Screen OCR text ──→ │                  │
                    │   LLM Decoder    │ (LoRA fine-tuned, ~1-2B params)
Dictionary ──────→  │  (Qwen3-1.7B    │
(as soft prompts    │   or Llama 1B)  │
 or text context)   │                  │
                    └────────┬─────────┘
                             │
                             ▼
                      Corrected Text
                    (one forward pass)
```

### 4c. Why This Works

1. **Audio encoder** — The hard part of ASR. Trained on thousands of hours of speech data. We don't train this — we steal a pre-trained one (Whisper encoder or Moonshine encoder). Frozen at inference.

2. **Adapter** — A tiny bridge (~10-50M params) that projects audio embeddings into the LLM's text embedding space. This is the only truly new component to train. Prior work (SLAM-ASR, Whisper-LLaMA) shows a simple linear projection or 2-layer MLP is sufficient.

3. **LLM decoder** — A small language model (1-2B params) fine-tuned with LoRA to:
   - Generate transcription from audio embeddings (replaces Whisper's decoder)
   - Apply formatting commands (what our post-processor already does)
   - Use dictionary context to resolve ambiguity
   - Handle in-speech meta-commands ("with a Q")

   All in one forward pass. No separate ASR → post-processing step.

4. **Dictionary as soft prompts** — Instead of text in the prompt, dictionary entries are stored as learned embedding vectors. During inference, relevant embeddings are injected into the decoder's attention. Scales to thousands of entries with fixed cost.

### 4d. Size Budget

| Component | Params | Size (4-bit) |
|---|---|---|
| Audio encoder (Moonshine) | ~100M | ~50 MB (frozen, can be fp16) |
| Adapter | ~10-50M | ~10-25 MB |
| LLM decoder (1.7B) | ~1.7B | ~1 GB |
| Dictionary embeddings | ~1-5M | ~2-5 MB |
| **Total** | **~1.8-1.9B** | **~1.1-1.3 GB** |

Compare to current pipeline: Whisper Large v3 (~3 GB) + Qwen3-4B DWQ (~2.1 GB) = **5.1 GB**.

The fused model would be **~4x smaller** and **~3-5x faster** (one forward pass vs two sequential inferences).

### 4e. Training Strategy

**Phase 1: Adapter pre-training**
- Freeze audio encoder AND LLM decoder
- Train only the adapter on (audio, transcript) pairs (e.g., LibriSpeech)
- Goal: teach the adapter to project audio embeddings so the LLM can decode them into text
- Data: standard ASR datasets (LibriSpeech, Common Voice, etc.)

**Phase 2: End-to-end fine-tuning**
- Freeze audio encoder
- Train adapter + LoRA on decoder simultaneously
- Data: (audio, corrected_text) pairs — the audio contains raw speech, the target is the FINAL corrected text (not raw transcript)
- This teaches the model to transcribe AND post-process in one shot
- Include dictionary context examples: (audio, dictionary, corrected_text)

**Phase 3: Dictionary learning**
- Train soft prompt embeddings for dictionary entries
- Freeze everything else
- Show the model (audio_with_ambiguous_word, dictionary_entry, correct_spelling) triplets

### 4f. The Learning Loop

```
User dictates → Model transcribes → User sees result
                                         │
                                    User corrects on keyboard?
                                         │ yes
                                         ▼
                                    Add/boost word in dictionary
                                    (frequency + recency weighted)
                                         │
                                         ▼
                                    Next dictation: dictionary
                                    entry biases the model
```

No retraining needed. The dictionary is data, not weights. The model already knows HOW to use dictionary context (trained for it in Phase 2). Memory grows from corrections.

For the Tuna launcher effect: rank dictionary entries by `score = count * recency_decay`. Most-used, most-recent words surface first. If the dictionary exceeds prompt capacity, retrieve top-K most relevant entries using phonetic similarity (Soundex/Metaphone) or embedding distance to the current audio.

---

## 5. Open Questions (Need Research)

1. **What's the smallest LLM decoder that works?** We know 4B → 100%, 3B → 91%, 1.2B → 70% on text-only post-processing. But with audio embeddings directly, the decoder has richer signal — might need fewer parameters.

2. **Can this be trained in MLX?** MLX has Whisper support and LLM LoRA training. But fusing them into one training graph is uncharted territory for mlx-lm.

3. **Adapter architecture** — Linear projection vs MLP vs Q-Former? How to handle the length mismatch (audio encoder outputs thousands of frames, text is much shorter)?

4. **Training data** — Can we train end-to-end on (audio → corrected text) from the start? Or must we stage it (first ASR, then post-processing)?

5. **Does Gemma 3n's modular architecture apply?** Google's Gemma 3n has a modular design where you can swap encoders and decoders. Could we use their framework to prototype this?

6. **Moonshine's encoder vs Whisper's encoder** — Moonshine is 6x faster with better WER (2.19% vs 3.02% on our benchmark). Is its encoder compatible with this fusion approach?

---

## 6. Prior Art

- **SLAM-ASR** — Speech-LLM Adapter Module. Connects Whisper encoder to LLMs via adapter.
- **Whisper-LLaMA** — Whisper encoder + LLaMA decoder with projection layer.
- **Qwen2-Audio** — Whisper encoder + Qwen2 decoder. Production quality but 7B+.
- **Mini-Omni** — Small audio-language model attempting real-time speech understanding.
- **Gemma 3n** — Google's modular multimodal architecture with swappable encoders.
- **Moonshine** — UsefulSensors' tiny ASR model (245M params). Encoder could be reused.

---

## 7. Success Criteria

The fused model is worth building if it achieves:

| Metric | Target | Current (2-model pipeline) |
|---|---|---|
| End-to-end latency | < 500ms | ~1.5-2s (Whisper + LLM) |
| Total model size | < 2 GB | 5.1 GB (Whisper + Qwen3 DWQ) |
| ASR accuracy (WER) | < 5% on LibriSpeech | 3.02% (Whisper alone) |
| Post-processing accuracy | > 90% on expanded test set | 96% (DWQ 4-bit on 23 examples) |
| Dictionary capacity | 1000+ words, no retraining | ~100 words in prompt |
| Learning from corrections | Automatic, no retraining | Not implemented |
| Memory during inference | < 4 GB | ~6 GB (both models loaded) |

If we can get close to these numbers with a ~1.5-2B fused model, it would be a significant improvement over the current two-model pipeline in every dimension: faster, smaller, more capable, and extensible without retraining.
