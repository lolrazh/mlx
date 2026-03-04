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

## 5. Open Questions → Answered

We researched all six open questions. Here's what we found:

### Q1: Smallest LLM decoder that works?

**Answer: Qwen3-1.7B is the sweet spot.**

Our text-only post-processing results (4B → 100%, 3B → 91%, 1.2B → 70%) suggest a steep drop below 2B. But with audio embeddings providing richer signal than text-only input, the decoder needs fewer parameters for the same task — it doesn't have to "re-imagine" the audio from a lossy transcript.

Benchmarks from existing audio-LLM models confirm:
- **Qwen3-1.7B**: Used in Qwen3-ASR-1.7B, achieves **1.63% WER** on standard benchmarks. At 4-bit quantization: ~0.85 GB, sub-200ms for typical utterances on M4.
- **Qwen3-0.6B**: Exists as Qwen3-ASR-0.6B. Much faster but 61.8% instruction reliability drop — brittle on complex commands. Not suitable for post-processing.
- **Estimated post-processing accuracy with audio input**: ~87-91% for 1.7B (extrapolating from our text-only curve, offset by richer audio signal).

**Recommendation:** Start with Qwen3-1.7B. If accuracy is insufficient, try 4B. Don't go below 1B for post-processing tasks.

### Q2: Can this be trained in MLX?

**Answer: Yes. All building blocks exist today.**

The `mlx_audio` package (pip-installable) already contains working implementations of the exact encoder→adapter→LLM decoder pattern:

| Model | Architecture | Status in MLX |
|---|---|---|
| **Qwen3-ASR** | AuT encoder → linear projection → Qwen3 decoder | Inference working via `mlx_audio` |
| **Voxtral** | Whisper encoder → MLP adapter → Mistral decoder | Inference working via `mlx_audio` |
| **VibeVoice ASR** | Encoder → adapter → LLM decoder | Inference working via `mlx_audio` |

For **training**, MLX provides:
- `mlx.nn.LoRALinear` for efficient LoRA on the decoder
- `mlx.nn.value_and_grad` for gradient computation through the full graph (encoder → adapter → decoder)
- `mx.compile` for kernel fusion and speed
- Custom training loops (not `mlx_lm.lora` — that's text-only). We'd write a ~200-line training script.

**The gap:** No off-the-shelf "audio-LLM LoRA trainer" in MLX. We'd need a custom training loop. But the primitives are all there — this is engineering, not research.

### Q3: Adapter architecture?

**Answer: Simple linear projection is sufficient. Frame stacking handles the length mismatch.**

Research from SLAM-ASR (2024) showed this is "embarrassingly simple" — a single linear layer from encoder hidden dim to LLM hidden dim achieves competitive results. No Q-Former needed.

**Practical adapter design:**
```
Linear(encoder_dim, lm_dim) → ReLU → Linear(lm_dim, lm_dim)
```
- Parameters: ~5-21M depending on dims (e.g., 1024→2560 = ~5M for 2-layer MLP)
- **Frame stacking with k=5**: Concatenate every 5 consecutive audio frames before projection. This reduces the sequence length by 5x (e.g., 1500 frames → 300 tokens), solving the length mismatch without a complex pooling mechanism.
- Qwen3-ASR uses a single linear projection layer. Falcon3-Audio uses a 2-layer MLP. Both work.

### Q4: End-to-end training from the start?

**Answer: Yes, but staged training is faster to converge.**

**Whispering-LLaMA** (2024) demonstrated end-to-end training on (audio → corrected text) with only 8M trainable parameters (adapter + LoRA). It worked, but required careful learning rate scheduling.

**Falcon3-Audio** demonstrated simultaneous adapter + LoRA training from scratch — adapter and decoder LoRA weights are trained together in a single pass.

**Recommended approach:**
1. **Phase 1** (2-4 hours): Train adapter only on standard ASR data (LibriSpeech). Freeze encoder + decoder. This teaches the adapter to project audio into the LLM's embedding space. ~1K-5K examples sufficient.
2. **Phase 2** (2-4 hours): Train adapter + LoRA together on (audio, corrected_text) pairs. This is where the post-processing behavior is learned. Our existing 1,200 text examples need to be paired with audio recordings (TTS-generated audio works for this).
3. **Phase 3** (optional): Dictionary-aware fine-tuning with (audio, dictionary_context, corrected_text) triplets.

Total: 1K-5K training examples per phase. Comparable to what we already have.

### Q5: Does Gemma 3n apply?

**Answer: Architecture philosophy yes. The model itself, no.**

Gemma 3n introduces three ideas relevant to us:
1. **MatFormer** — Nested sub-models within the same weight matrices. Multiple model sizes share one set of weights. Interesting for future optimization (serve a tiny model for simple utterances, full model for complex ones).
2. **Per-Layer Embeddings (PLE)** — Offload embedding tables to CPU/SSD, load per-layer on demand. Reduces active memory at the cost of some latency. Relevant for running on smaller devices.
3. **Modular encoders** — Swap vision/audio encoders via tiny projection layers (~17 MB each). This is exactly our adapter concept, validated at Google scale.

**But Gemma 3n's actual audio quality is terrible** — 35-83% WER across benchmarks. Their audio encoder is an afterthought. Don't use Gemma 3n for ASR. Use their architectural patterns as inspiration.

### Q6: Moonshine vs Whisper encoder?

**Answer: Moot — Qwen3-ASR's AuT encoder beats both.**

| Encoder | WER (LibriSpeech) | Size | Speed |
|---|---|---|---|
| Whisper Large v3 | 3.02% | ~1.5B | 0.34x RTF |
| Moonshine Medium | 2.19% | ~100M | 0.11x RTF |
| **Qwen3-ASR AuT** | **1.63%** | **~300M** | **55x real-time** |

Qwen3-ASR's Audio Transformer (AuT) encoder is purpose-built for the encoder→adapter→LLM pipeline and outperforms both Whisper and Moonshine. Since Qwen3-ASR already exists as a complete working model with the exact architecture we proposed, the encoder question is answered: use AuT.

Moonshine's encoder could still be viable if we need an even smaller model, but the accuracy gap is significant.

---

## 6. Prior Art (Expanded)

### Directly Relevant (exact architecture match)

- **Qwen3-ASR-1.7B** (2025) — AuT encoder (300M) → linear projection → Qwen3-1.7B decoder. 1.63% WER on LibriSpeech. **Already ported to MLX** via `mlx_audio`. ~3.4 GB at fp16. 55x faster than real-time on M4. This is essentially our proposed architecture, pre-built.
- **Qwen3-ASR-0.6B** — Smaller variant. Faster but lower accuracy and brittle on complex instructions.
- **SLAM-ASR** (2024) — "Embarrassingly simple" approach. Single linear projection between frozen Whisper encoder and LLM decoder achieves competitive ASR. Proved that complex adapters (Q-Former) are unnecessary.
- **Whispering-LLaMA** (2024) — Whisper encoder + LLaMA decoder. End-to-end training with only 8M trainable parameters (adapter + LoRA). Demonstrated that staged training converges faster but end-to-end works.
- **Falcon3-Audio** (2025) — Whisper encoder + Falcon decoder. Simultaneous adapter + LoRA training. Showed that both components can be trained together from scratch.

### Architecturally Relevant

- **Voxtral** (Mistral, 2025) — Whisper encoder → MLP adapter → Mistral decoder. Production-grade audio-LLM. Working in MLX via `mlx_audio`.
- **Qwen2-Audio** (2024) — Whisper encoder + Qwen2 decoder. Production quality but 7B+. Too large for on-device.
- **Mini-Omni** (2024) — Small audio-language model for real-time speech understanding. Streaming-capable.
- **VibeVoice ASR** — Encoder-adapter-decoder ASR model, working in MLX.

### Architecture Concepts

- **Gemma 3n** (Google, 2025) — Modular multimodal architecture. Key ideas: MatFormer (nested sub-models), PLE (per-layer embeddings for memory reduction), modular encoders via ~17 MB projection layers. Audio quality poor (35-83% WER), but architecture patterns are valuable.
- **Moonshine** (UsefulSensors) — 245M param ASR model. Encoder could be reused but outperformed by Qwen3-ASR's AuT encoder.

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

---

## 8. Research Findings: The Shortcut

### The model we want to build already exists.

**Qwen3-ASR-1.7B** is a pre-trained audio-language model with the exact architecture we proposed independently:

```
Audio Waveform → AuT Encoder (300M, frozen) → Linear Projection → Qwen3-1.7B Decoder → Text
```

It achieves **1.63% WER** on LibriSpeech test-clean — better than both our Whisper baseline (3.02%) and Moonshine baseline (2.19%). It's already ported to MLX via the `mlx_audio` package and runs at **55x real-time** on Apple Silicon.

This changes the project from "build a fused model from scratch" to "fine-tune an existing fused model for our specific task."

### What Qwen3-ASR-1.7B already does:
- Transcribes audio to text with state-of-the-art accuracy
- Runs on Apple Silicon via MLX
- Uses the same Qwen3 decoder family we already fine-tuned for post-processing

### What it doesn't do (yet — our fine-tuning target):
- Execute in-speech meta-commands ("Groq with a Q")
- Apply formatting (quotes, caps, emphasis)
- Handle self-corrections ("I mean X not Y")
- Use dictionary context for disambiguation
- Learn from user corrections

### MLX ecosystem support

The `mlx_audio` package provides ready-made building blocks:

| Component | MLX Status |
|---|---|
| Audio preprocessing (mel spectrograms) | Working (via mlx_audio) |
| Audio encoder (AuT, Whisper) | Working (frozen, pre-trained) |
| Linear projection adapter | Working (simple nn.Linear) |
| LLM decoder (Qwen3-1.7B) | Working (via mlx-lm) |
| LoRA fine-tuning | Working (mlx.nn.LoRALinear) |
| Gradient computation through full graph | Supported (nn.value_and_grad) |
| Kernel fusion | Supported (mx.compile) |

**Gap:** No off-the-shelf audio-LLM LoRA trainer. We write a custom training loop (~200 lines).

---

## 9. Recommended Path Forward

### Phase 0: Validate the base model (1 day)

1. Install `mlx_audio`, load Qwen3-ASR-1.7B
2. Run it on our existing eval pipeline (LibriSpeech test-clean)
3. Verify the 1.63% WER claim on our hardware
4. Measure latency, memory, throughput on M4
5. Test with our ad-hoc dictation samples — see how it handles natural speech

**Exit criteria:** Model runs, WER is competitive, latency < 500ms for typical utterances.

### Phase 1: Text-only decoder fine-tuning (2-3 days)

Before touching audio, verify that Qwen3-1.7B's decoder can learn our post-processing task:

1. Fine-tune Qwen3-1.7B (text-only, no audio encoder) using our existing v4 data (1,201 examples)
2. Use the same LoRA setup: r=8, lr=1e-5, 2000 iters
3. Benchmark against our 23-example test set + expanded real-world tests
4. This tells us the decoder's ceiling for post-processing at 1.7B params

**Exit criteria:** Accuracy > 87% on test set (matching 3B Llama). If < 70%, the 1.7B decoder is too small and we need Qwen3-4B.

### Phase 2: Audio + post-processing fusion (1-2 weeks)

1. Write custom training loop that passes audio through the frozen AuT encoder → adapter → decoder
2. Create training data: pair our text post-processing examples with audio (use macOS TTS or record real speech)
3. Train adapter + LoRA simultaneously:
   - Audio input: speech containing meta-commands
   - Target output: corrected text (not raw transcript)
4. Start with our 1,200 examples, expand to 3-5K as needed

**Exit criteria:** Model transcribes AND post-processes in one forward pass with > 85% accuracy on post-processing tasks.

### Phase 3: Dictionary and learning (1 week)

1. Add dictionary context to the decoder's input (text prefix or learned embeddings)
2. Fine-tune on (audio, dictionary, corrected_text) triplets
3. Implement the learning loop: keyboard corrections → dictionary update → next inference is biased
4. Test with ambiguous vocabulary (Groq vs Grok, etc.)

**Exit criteria:** Dictionary resolves 90%+ of ambiguous words. Learning loop works end-to-end.

### Phase 4: Quantize and deploy (2-3 days)

1. DWQ 4-bit quantization on the decoder (adapter stays fp16)
2. Measure final latency, memory, accuracy
3. Integrate into macOS dictation app

**Target final model:**

| Component | Size | Latency contribution |
|---|---|---|
| AuT encoder (fp16) | ~600 MB | ~50ms |
| Adapter (fp16) | ~20 MB | ~5ms |
| Decoder (4-bit) | ~850 MB | ~150ms |
| **Total** | **~1.5 GB** | **~200ms** |

Compare to current: 5.1 GB, 1.5-2s. That's **3.4x smaller** and **7-10x faster**.

---

## 10. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| 1.7B decoder too small for post-processing | Medium | Fall back to Qwen3-4B (~2.5 GB total, still 2x smaller than current pipeline) |
| TTS-generated training audio doesn't generalize to real speech | Medium | Record real dictation samples; augment with noise/speed variation |
| Custom MLX training loop is complex | Low | All primitives exist in mlx_audio examples; ~200 lines of new code |
| Dictionary soft prompts don't scale | Low | Fall back to text-in-prompt (our current approach works up to ~100 words) |
| Qwen3-ASR encoder quality regresses on dictation-style speech | Low | Test in Phase 0 before committing; swap to Whisper encoder if needed |

### What could go wrong

The biggest unknown is whether a 1.7B decoder can handle BOTH transcription AND post-processing simultaneously. These are two tasks sharing one set of LoRA weights. If they interfere, accuracy on one or both tasks could degrade. Phase 1 (text-only 1.7B) is specifically designed to detect this risk early.

### What makes this different from "just another audio-LLM"

Most audio-LLMs are built for conversational AI (talking to a chatbot). We're building a **scribe** — a model that listens and writes, not one that listens and responds. The post-processing aspect (meta-commands, formatting, self-correction) is unusual and may require task-specific training data that doesn't exist in standard datasets.
