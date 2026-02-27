# Spoke: LLM Post-Processing for ASR

## Overview

Fine-tune a small LLM (<4B params, <2GB at 4-bit) to clean raw ASR transcripts containing embedded verbal meta-commands. The model should run on a 16GB MacBook with <400ms total latency.

This is the core intelligence behind **Spoke** — a dictation app that understands when you say things like "spell that as S-I-L-E-R-O" or "actually scratch that" and executes those commands inline.

## Architecture

```
User speaks
    |
    v
+-------------------+
|   ASR Engine       |  Parakeet TDT V3 (0.03x RTF, 1.93% WER)
|   (parakeet-mlx)   |  or Moonshine MLX (0.04x RTF, 2.19% WER)
+--------+----------+
         |  raw transcript
         v
+-------------------+
|  Regex Router      |  <1ms
|  (trigger words)   |---- no triggers ----> output as-is (~70-80% of inputs)
+--------+----------+
         |  triggers found
         v
+-------------------+
|  Fine-tuned LLM    |  ~200-400ms
|  (LoRA on MLX)     |
+--------+----------+
         |  cleaned text
         v
      Display
```

### Regex Router Triggers

The router checks for keywords/patterns that indicate the LLM is needed:

```
Spelling:       /\b[A-Z](-[A-Z]){2,}/          S-I-L-E-R-O
                /spell (that|it|this) as/
Self-correction: /wait no|sorry|scratch that|actually,?\s*(no|not|say|make|let)/
Quoting:        /quote.?unquote|end quote/
Symbols:        /at symbol|at sign|dot com|dot org/
Formatting:     /in (all )?caps|lowercase|uppercase|emphasize|bold/
Emoji:          /emoji$|\bemoji\b/
                /^(two |three )?(hearts?|crying|broken|fire)/
```

If none match, the transcript passes through untouched. This saves compute on ~70-80% of inputs.

## The Task

**Input:** Raw ASR transcript with embedded verbal commands
**Output:** Clean text with commands executed and removed

### Capability Categories

| # | Category | Example Input | Expected Output |
|---|----------|--------------|-----------------|
| 1 | **Spell-and-replace** | "Celero VAD... spell that S-I-L-E-R-O" | "Silero VAD" |
| 2 | **Self-correction** | "AMD. Wait no, sorry, Nvidia" | "Nvidia" |
| 3 | **Quote-unquote** | "quote-unquote brilliant" | `"brilliant"` |
| 4 | **@ symbol insertion** | "Tag mom with an at symbol" | "@mom" |
| 5 | **Email/URL dictation** | "rajkumar dot sandheep at gmail dot com" | "rajkumar.sandheep@gmail.com" |
| 6 | **Formatting: caps** | "Write in caps" | ALL CAPS OUTPUT |
| 7 | **Formatting: emphasis** | "Emphasize surprisingly" | `**surprisingly**` |
| 8 | **Emoji** | "Two hearts" | "❤️❤️" |
| 9 | **Multi-step correction** | Spell + replace + self-correct in one | Combined operations |
| 10 | **CamelCase/code** | "usetranscription.ts" | "useTranscription.ts" |
| 11 | **Prompt injection** | "ignore previous instructions" | Pass through unchanged |

### What Makes This Hard

- Meta-commands are embedded *inside* content, not in a system prompt
- Multiple commands can overlap in a single utterance (row 39 of evals)
- Model must know which parts are commands to strip vs. content to keep
- "like" is sometimes filler (strip) and sometimes intentional (keep)
- Frontier models (GPT-5, Gemini Flash, DeepSeek v3.2) underperform on this
- Best performers: Kimi K2, Llama 4 Maverick, Gemma 27B

---

## Phase 1: Dataset Curation

### Seed Data

60 gold examples in `Spoke - Evals.csv` covering all 11 categories.

### Split Strategy

```
60 gold examples
├── 12 --> sacred test set (stratified, 2+ per category, NEVER train on)
├── 8  --> validation set (stratified)
└── 40 --> training seeds for expansion
```

### Expansion Pipeline (using Kimi K2.5 via Baseten)

**Why Kimi K2.5:** User reports it as the best-performing model on this task. Using it as the teacher for distillation is ideal — generate training data from the strongest available model.

**Step 1: Categorize seeds**
Tag each of the 40 training seeds by category (multi-label where applicable).

**Step 2: Generate synthetic examples**
For each category, prompt Kimi K2.5 with 3-5 seed examples:

```
System: You are generating training data for an ASR post-processing model.
Given examples of (raw_transcript, clean_output) pairs, generate 20 new
diverse pairs for the category: [CATEGORY].

Rules:
- Vary sentence length, word choice, topic, and command placement
- Include realistic disfluencies and speech patterns
- The raw transcript should sound like something a person would actually say
- Commands can appear at the start, middle, or end of the utterance
- Include some examples where multiple commands overlap
- Make the examples progressively harder

Here are seed examples:
[3-5 seeds]

Generate 20 new (raw_transcript, clean_output) pairs as JSON:
```

**Step 3: Quality filter**
Run each generated pair through Kimi K2.5 again:
```
Is this (input, output) transformation correct?
Does the output faithfully execute all commands in the input?
Score 1-5 and explain any issues.
```
Keep only 4-5 rated examples.

**Step 4: Human review**
Manually review ~20% random sample. Fix or discard bad examples.

**Step 5: Category balancing**
Target distribution (~500 total training examples):

```
spell-replace:        50  (10%)
self-correction:      50  (10%)
quote-unquote:        40  (8%)
at-symbol/email:      50  (10%)
formatting commands:  50  (10%)
emoji:                40  (8%)
caps/case:            40  (8%)
code-aware:           30  (6%)
multi-command combos: 80  (16%)
edge cases/ambiguous: 40  (8%)
prompt injection:     30  (6%)
```

No pass-through examples needed — the regex router handles that.

### Dataset Format

Chat JSONL for mlx-lm with `--mask-prompt`:

```jsonl
{"messages": [{"role": "system", "content": "Clean the transcript by executing all verbal commands (spell-outs, corrections, formatting, symbols, emoji). Output ONLY the cleaned text."}, {"role": "user", "content": "I'm gonna be using Celero VAD for this. Can you spell that as S-I-L-E-R-O?"}, {"role": "assistant", "content": "I'm gonna be using Silero VAD for this."}]}
```

System prompt is short and consistent across all examples.

### Output Files

```
spoke/data/
├── evals.csv          (original 60 gold examples)
├── train.jsonl        (~500 examples)
├── valid.jsonl        (8 gold examples)
└── test.jsonl         (12 gold examples, sacred)
```

---

## Phase 2: Model Selection

### Candidates

Run the 12-example test set zero-shot through each model to establish baselines before any fine-tuning.

| Priority | Model | Params | 4-bit Size | Why Test |
|----------|-------|--------|-----------|----------|
| **Must** | Qwen3-4B-2507 | 4.0B | 2.0 GB | #1 fine-tuning benchmark, 91.6% IFEval |
| **Must** | LFM2.5-1.2B | 1.2B | 0.6 GB | 86% IFEval at 1.2B, non-transformer |
| **Must** | Phi-4-mini | 3.8B | 1.9 GB | MIT, strong reasoning |
| **Should** | Nemotron-Nano-4B | 4.0B | 2.0 GB | Compressed 8B, MT-Bench 7.4 |

### Zero-Shot Evaluation

For each model, run all 12 test examples with the same system prompt. Score:

1. **Exact match** — output matches ideal exactly (after whitespace normalization)
2. **Semantic match** — output is correct but phrased slightly differently
3. **Partial** — some commands executed correctly, others missed
4. **Fail** — wrong output, hallucination, or commands not executed

Compute:
- **Accuracy** = (exact + semantic) / total
- **Per-category accuracy** = breakdown by the 11 categories
- **Latency** = TTFT + generation time

### Decision Criteria

Pick top 2 models for fine-tuning based on:
1. Zero-shot accuracy (is the base good enough to fine-tune?)
2. Latency (must be <400ms total for short transcripts)
3. Memory footprint (must run alongside ASR model in 16GB)

---

## Phase 3: Fine-Tuning

### Training Config

```yaml
# spoke/config.yaml
model: mlx-community/Qwen3-4B-Instruct-2507-4bit  # or winner from Phase 2
train: true
data: ./spoke/data
fine_tune_type: lora
mask_prompt: true

lora_parameters:
  rank: 16
  scale: 2.0          # alpha = 32
  dropout: 0.05

batch_size: 4
iters: 1000
learning_rate: 1e-5
num_layers: 16
max_seq_length: 512    # transcripts are short
steps_per_report: 10
steps_per_eval: 100
save_every: 200
val_batches: 25
seed: 42

adapter_path: ./spoke/adapters
```

### Training (on M4 24GB)

```bash
# Train
mlx_lm.lora -c spoke/config.yaml

# Fuse (dequantize first to avoid quality loss)
mlx_lm.fuse \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --adapter-path ./spoke/adapters \
  --save-path ./spoke/fused \
  --dequantize

# Re-quantize for deployment
mlx_lm.convert --hf-path ./spoke/fused --q-bits 4 -o ./spoke/model
```

Estimated training time: ~10-15 minutes on M4 24GB.

### Evaluation

Run the fused model against the sacred 12-example test set. Compare:

| Metric | Zero-shot | Fine-tuned | Target |
|--------|-----------|------------|--------|
| Accuracy | ? | ? | >85% |
| TTFT | ? | ? | <200ms |
| Total latency | ? | ? | <400ms |

### Iteration Loop

```
Fine-tune --> Evaluate --> Find failure modes
                              |
                              v
                 Add targeted examples to train.jsonl
                              |
                              v
                         Re-train (10 min)
```

Focus on adding examples for categories the model gets wrong, not adding more of what it already handles.

---

## Phase 4: Deployment

### Inference Pipeline

```python
import mlx_lm
import re

# Load once at startup
model, tokenizer = mlx_lm.load("./spoke/model")

TRIGGERS = re.compile(
    r"[A-Z](-[A-Z]){2,}"           # S-I-L-E-R-O
    r"|spell (that|it|this)"
    r"|wait no|sorry|scratch that"
    r"|actually,?\s*(no|not|say)"
    r"|quote.?unquote|end quote"
    r"|at symbol|at sign"
    r"|dot com|dot org"
    r"|in (all )?caps|lowercase"
    r"|emphasize|bold"
    r"|emoji",
    re.IGNORECASE
)

SYSTEM = ("Clean the transcript by executing all verbal commands "
          "(spell-outs, corrections, formatting, symbols, emoji). "
          "Output ONLY the cleaned text.")

def process(transcript: str) -> str:
    if not TRIGGERS.search(transcript):
        return transcript  # fast path, <1ms

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": transcript},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = mlx_lm.generate(
        model, tokenizer, prompt=prompt,
        max_tokens=512, temp=0.0
    )
    return response.strip()
```

### Memory Budget (16GB MacBook)

```
macOS + apps:        ~4 GB
ASR (Parakeet):      ~1.2 GB
LLM (Qwen3-4B 4bit): ~2.5 GB (loaded on demand, or kept warm)
KV cache:            ~0.5 GB
Headroom:            ~7.8 GB
Total:               ~8.2 GB used
```

Fits comfortably with room for context and other apps.

### Latency Budget

```
ASR (Parakeet, 5s audio):    ~200ms
Regex router:                 <1ms
LLM (when triggered):        ~200-400ms
Total (worst case):           ~400-600ms
Total (no LLM needed):       ~200ms
```

---

## Parallel Workstreams

These can run simultaneously:

### Track A: Dataset (User)
1. Set up Kimi K2.5 via Baseten
2. Categorize 60 seed examples
3. Generate synthetic training data
4. Quality filter + human review
5. Format as train.jsonl / valid.jsonl / test.jsonl

### Track B: Model Benchmarking (Agent)
1. Download candidate models (Qwen3-4B-2507, LFM2.5-1.2B, Phi-4-mini, Nemotron-Nano-4B)
2. Run zero-shot eval on test set
3. Profile latency (TTFT + generation)
4. Report results, pick top 2

### Track C: Fine-Tuning (After A + B converge)
1. Train top 2 models on dataset from Track A
2. Evaluate on sacred test set
3. Iterate on data + hyperparameters
4. Fuse + quantize winning model
5. Integration test with ASR pipeline

---

## Success Criteria

| Metric | Target | Stretch |
|--------|--------|---------|
| Test set accuracy | >85% | >92% |
| Multi-command accuracy | >70% | >85% |
| TTFT (short transcript) | <200ms | <100ms |
| Total latency (ASR + LLM) | <600ms | <400ms |
| Model size (4-bit) | <2.5 GB | <1 GB |
| Runs on 16GB MacBook | Yes | With ASR simultaneously |

---

## Open Questions

1. **Kimi K2.5 prompt tuning** — What prompt structure gets the best synthetic data quality? Need to experiment.
2. **How much data is enough?** — Start with 500, but may need 1000+ for multi-command cases.
3. **LoRA vs full fine-tune** — If LoRA plateaus, try full fine-tune on the smaller models (LFM2.5-1.2B).
4. **Existing GEC datasets** — Are there grammar error correction or ASR post-processing datasets we can mix in? Could boost generalization.
5. **Quantization impact** — Does 4-bit quantization after fine-tuning hurt accuracy? May need to test 8-bit as fallback.
