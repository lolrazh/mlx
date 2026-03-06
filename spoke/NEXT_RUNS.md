# Next Training Runs

Temporary doc. Delete after runs are complete.

## Context

Current best: Qwen3-4B DWQ 4-bit = 96% accuracy, 2.1 GB, 0.88s latency.
Problem: 2.1 GB is over <2 GB target. Broad eval only 67%. Llama 3B tops at 87% bf16.
Goal: >90% accuracy at <2 GB deployed size. Ideally much smaller.

## Path A: Gemma 3n E2B Re-run (corrected hyperparams)

Previous run used lr=1e-5 (20x too low) -> 65% accuracy. Google recommends 2e-4.

### Hyperparameters (Google official QLoRA guide)

| Parameter | Old (wrong) | New (correct) |
|-----------|-------------|---------------|
| learning_rate | 1e-5 | **2e-4** |
| lr_scheduler | constant (no warmup) | **constant + warmup** |
| warmup_ratio | 0 | **0.03** |
| max_grad_norm | default (1.0) | **0.3** |
| weight_decay | 0 | **0.01** |
| lora_r | 8 | **16** |
| lora_alpha | 16 | **16** |
| lora_dropout | 0.05 | **0.05** |
| optimizer | adamw | **adamw_torch_fused** |
| batch_size | 4 | 4 (keep) |
| max_steps | 1200 | **2000** |
| data | v5 | **v4** |

### Code changes needed in train_hf.py

1. Add `--lr-scheduler` flag (currently hardcodes `lr_scheduler_type="constant"`)
   - constant with warmup needs: `lr_scheduler_type="constant_with_warmup"`, `warmup_ratio=0.03`
2. Add `--warmup-ratio` flag
3. Add `--max-grad-norm` flag (currently hardcodes `max_grad_norm=1.0`)
4. Add `--weight-decay` flag
5. Add `--lora-r` flag (currently hardcodes `r=8`)

### Run command (after code changes)

```bash
python spoke/cloud/upload_data.py  # ensure v4 data on volume
modal run spoke/cloud/train_hf.py \
  --model-name google/gemma-3n-E2B-it \
  --run-name spoke-gemma3n-e2b-v4-v2 \
  --learning-rate 2e-4 \
  --lr-scheduler constant_with_warmup \
  --warmup-ratio 0.03 \
  --max-grad-norm 0.3 \
  --weight-decay 0.01 \
  --lora-r 16 \
  --max-steps 2000 \
  --data-dir /data/v4
```

### Expected size at 4-bit

E2B effective = ~2B params. At 4-bit: ~1.0 GB. Well under 2 GB target.
E4B effective = ~4B params. At 4-bit: ~2.0 GB. Right at target.

### Architecture notes

- Gemma 3n has built-in low-rank (LAuReL rank=64, PLE gate=256). External LoRA r=16 is sufficient.
- E2B is a nested subset of E4B (MatFormer). Adapters NOT transferable between variants.
- For text-only: use `AutoModelForCausalLM` with text_config extraction (already in train_hf.py).

---

## Path B: T5 / Flan-T5 Proof-of-Concept

Encoder-decoder architecture is ideal for copy-heavy text editing. T5-base = 220M params.
FlanEC paper shows Flan-T5-base reduces ASR WER 11.8% -> 10.3% (13% relative improvement).
Small decoder-only models are 3x worse at grammar correction than encoder-decoder (Jan 2026 paper).

### Model options

| Model | Params | 4-bit size | Notes |
|-------|--------|------------|-------|
| Flan-T5-small | 60M | ~15-20 MB | Might be too small for 9 trigger categories |
| **Flan-T5-base** | 220M | ~55-70 MB | Good starting point |
| **Flan-T5-large** | 770M | ~190-240 MB | Sweet spot if base is insufficient |
| Flan-T5-3B | 3B | ~750 MB | Overkill but guaranteed accuracy |

### Training approach

**Input format** (seq2seq, NOT chat):
```
Correct this transcription: I'm gonna be using Celero VAD for this. Can you spell that as S-I-L-E-R-O?
```
**Output**:
```
I'm gonna be using Silero VAD for this.
```

No system prompt needed. T5 is pretrained with task prefixes. The "Correct this transcription:" prefix tells it what to do.

### Training options

**Option 1: Cloud (Modal) with HuggingFace Seq2SeqTrainer**
- Standard HF pipeline. Seq2SeqTrainer + Seq2SeqTrainingArguments.
- Full fine-tuning possible (only 220M params, fits in <2 GB VRAM).
- LoRA also works with PEFT on T5.
- Cost: ~$0.05-0.10 per run (T5-base is tiny).
- Need new Modal training script (separate from train_hf.py which is decoder-only).

**Option 2: Local with HuggingFace on MPS**
- T5-base bf16 = ~440 MB. Fits trivially on M4 24GB.
- Full fine-tuning feasible locally (no LoRA needed at this size).
- Risk: MPS has known precision bugs. CPU fallback is slow but works.
- HuggingFace Trainer supports `device="mps"`.

**Option 3: Local with MLX custom training loop**
- `mlx-examples/t5/` has inference only (t5.py, hf_t5.py). No training.
- Would need to write encoder-decoder training loop (~200-300 lines).
- Highest effort but most integrated with our toolchain.

**Recommended: Option 1 (cloud)** for reliability, then convert for MLX inference.

### Data format conversion

Current data is chat format (messages array). Need to convert to seq2seq pairs:
```python
# Convert JSONL chat -> seq2seq
for example in data:
    system_msg = example["messages"][0]["content"]  # system prompt
    user_msg = example["messages"][1]["content"]     # input text
    assistant_msg = example["messages"][2]["content"] # output text
    # T5 format:
    input_text = f"Correct this transcription: {user_msg}"
    target_text = assistant_msg
```

### Inference (MLX)

`mlx-examples/t5/` supports inference for T5-small through T5-11B and all Flan-T5 variants:
```bash
python t5.py --model google/flan-t5-base --prompt "Correct this transcription: ..."
```

Also supported via llama.cpp (GGUF conversion, PR #8055). Use Q5_K_M or larger (no imatrix for T5).

### Risk assessment

- T5-base (220M) might not have enough capacity for all 9 trigger categories.
  FlanEC: T5-base gives 13% WER reduction vs 28% for T5-3B.
- Flan-T5-large (770M, ~200 MB) is the fallback if base can't learn everything.
- Even T5-large at 240 MB is 9x smaller than current Qwen3 DWQ (2.1 GB).

---

## Data Augmentation (future, after runs)

### Semantic augmentation (spell categories)
Rephrase the instruction while keeping the transformation:
- "spell that K-A-D-A-I" -> "the spelling is K-A-D-A-I" -> "it should be K-A-D-A-I"
- "Can you spell that as S-I-L-E-R-O?" -> "Actually it's S-I-L-E-R-O"

### Lowercase augmentation (passthrough/hard-ignore categories)
Duplicate examples with lowercased input, same output. Teaches grammar correction as side effect.

### Emphasis scope precision
More examples where "emphasize X" has distractors nearby. Addresses over-emphasis error.

---

## Execution Order

1. Add hyperparameter flags to `train_hf.py` (for Gemma 3n)
2. Launch Gemma 3n E2B v2 run with corrected hyperparams
3. While that trains: build T5 training script (new file, seq2seq)
4. Convert v4 data to seq2seq format
5. Launch T5 Flan-T5-base run
6. Benchmark both
7. If T5-base < 85%, try Flan-T5-large
8. Delete this file
