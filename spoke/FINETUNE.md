# Spoke: Fine-Tuning Pipeline

Companion doc to `DATAGEN.md`. This covers model benchmarking, training,
evaluation, and deployment. Runs after datagen produces `spoke/data/final/`.

## Model Selection

### Why Qwen3-4B-Instruct-2507

Benchmarked 5 candidate models zero-shot on 12 sacred test examples.
Three prompt strategies tested: generic (1 sentence), task-specific
(per-category paragraph), and Spoke-style (production rules + few-shot examples).

| Model | Params | 4-bit Size | Generic | Task | Spoke | Avg Latency |
|-------|--------|-----------|---------|------|-------|-------------|
| **Qwen3-4B-2507** | 4.0B | 2.3 GB | **25%** | **50%** | **58%** | 0.71-1.47s |
| Gemma 3n E4B | 8B eff | 3.9 GB | 25% | — | — | 1.29s |
| Gemma 3 4B QAT | 4.0B | 2.0 GB | 17% | — | — | 0.92s |
| Phi-4-mini | 3.8B | 2.2 GB | 8% | — | — | 0.61s |
| LFM2.5-1.2B | 1.2B | 0.6 GB | 0% | — | — | 0.30s |

Qwen3-4B wins because:
1. Best zero-shot accuracy across all prompt modes
2. \#1 on distillabs fine-tuning benchmark (91.6% IFEval)
3. Understands the task structure — failures are near-misses, not nonsense
4. Natively supports thinking/non-thinking modes (we use non-thinking)
5. ChatML template, well-supported in mlx-lm

### Zero-Shot Failure Analysis (Qwen3-4B, Spoke prompts)

| Category | Score | Issue |
|----------|-------|-------|
| spell-replace | 1/2 exact | Can't map phonetics → spelled letters for novel words |
| self-correction | 1/2 exact | Over-corrects — drops too much context |
| quote-unquote | 1/1 exact | Works with production prompts |
| quote-endquote | partial | Wrong quote scope ("lucky" vs "lucky to be here") |
| at-symbol | fail | Passes through instruction unchanged |
| caps | exact | Already works zero-shot |
| emphasis | exact | Works with task-specific or Spoke prompts |
| emoji | partial | Outputs extra emoji (😢💔 instead of just 💔) |
| multi-step | exact | Works with Spoke prompts (few-shot example helps a lot) |
| camelcase | exact | Works with task-specific prompts |

**High-priority training categories**: spell-replace, self-correction,
quote-endquote, at-symbol. These are where the model struggles most.

Detailed results: `spoke/bench/result_qwen3-4b_*.json`

---

## Precision Strategy

```
Training:    bf16 base (8 GB) + fp16 LoRA adapters (50 MB)
Fusing:      bf16 fused model (7.5 GB)
Deployment:  4-bit quantized (2.1 GB)
```

**Why bf16 for training, not QLoRA on 4-bit?**
- Gradients flow through full-precision weights → better learning signal
- 9 GB peak memory on M4 24GB → fits comfortably
- QLoRA saves memory but loses signal quality through quantized weights
- Only use QLoRA if base model doesn't fit in memory (not our case)

**Why 4-bit for deployment?**
- 2.1 GB fits easily alongside ASR model on 16 GB MacBook
- Post-training quantization preserves fine-tuned knowledge well
- 4.5 bits/weight effective (mixed quantization: embeddings stay higher precision)

---

## Prerequisites

Before training, need from datagen (`DATAGEN.md`):

```
spoke/data/final/
├── train.jsonl    ~500 examples, chat JSONL format
├── valid.jsonl    ~8 examples (gold, stratified by category)
└── test.jsonl     ~12 examples (sacred, NEVER train on)
```

Each line:
```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "<raw>"}, {"role": "assistant", "content": "<cleaned>"}]}
```

System prompt is consistent across all examples:
```
Clean the transcript by executing all verbal commands
(spell-outs, corrections, formatting, symbols, emoji).
Output ONLY the cleaned text.
```

---

## Training Config

```yaml
# spoke/config.yaml
model: mlx-community/Qwen3-4B-Instruct-2507-bf16
train: true
data: ./spoke/data/final
fine_tune_type: lora
mask_prompt: true

lora_parameters:
  rank: 16
  scale: 2.0          # alpha = 32
  dropout: 0.05

batch_size: 4
iters: 1000
learning_rate: 1e-5
num_layers: 16         # top 16 of 36 layers
max_seq_length: 512    # transcripts are short
optimizer: adam

steps_per_report: 10
steps_per_eval: 100
save_every: 200
val_batches: -1        # use full validation set

adapter_path: ./spoke/adapters
seed: 42
```

### Key Hyperparameters

| Param | Value | Why |
|-------|-------|-----|
| `rank` | 16 | Sweet spot for narrow tasks. 8 if underfitting, 32 if dataset grows |
| `scale` | 2.0 | alpha = rank × scale = 32. Controls adapter contribution strength |
| `dropout` | 0.05 | Light regularization. Increase to 0.1 if overfitting |
| `lr` | 1e-5 | Conservative. Standard for LoRA on instruct models |
| `batch_size` | 4 | Fits in memory. Increase for smoother gradients |
| `iters` | 1000 | ~8 epochs over 500 examples at batch_size=4. Adjust based on val loss curve |
| `num_layers` | 16 | Fine-tune top 44% of layers. Bottom layers are general knowledge, top layers are task-specific |
| `mask_prompt` | true | Critical — only train on assistant responses, not the prompt |
| `max_seq_length` | 512 | Our transcripts are short. Saves memory vs default 2048 |

### Expected Training Profile

Based on pipeline test (M4 24GB):

```
Peak memory:     ~9 GB (with batch_size=4, may rise to ~11 GB)
Speed:           ~2 it/sec, ~30 tok/sec
Estimated time:  ~8-10 minutes for 1000 iterations
Val loss:        Should drop from ~2.5 to <0.5
```

---

## Training Procedure

### Step 0: Baseline (before training)

Run zero-shot benchmark to establish the pre-fine-tuning baseline:

```bash
cd /path/to/mlx
source .venv/bin/activate

# Zero-shot with production prompts
python spoke/bench/run_benchmark.py --model qwen3-4b --prompt-mode spoke
```

This gives us the numbers to beat.

### Step 1: Train

```bash
mlx_lm.lora -c spoke/config.yaml
```

Watch for:
- **Val loss curve**: should decrease steadily. If it plateaus early, increase rank or lr.
  If it starts increasing, you're overfitting — reduce iters or increase dropout.
- **Train loss vs val loss gap**: if train loss << val loss, overfitting.
  Add more data or increase dropout.

Adapters saved to `spoke/adapters/adapters.safetensors` (~50-100 MB).

### Step 2: Quick test (before full eval)

```bash
# Test directly with adapters (no fuse needed)
python -c "
import mlx_lm
from mlx_lm.sample_utils import make_sampler

model, tokenizer = mlx_lm.load(
    'mlx-community/Qwen3-4B-Instruct-2507-bf16',
    adapter_path='spoke/adapters'
)
greedy = make_sampler(temp=0.0)

messages = [
    {'role': 'system', 'content': 'Clean the transcript...'},
    {'role': 'user', 'content': 'spell that S-I-L-E-R-O'},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False,
    add_generation_prompt=True, enable_thinking=False
)
print(mlx_lm.generate(model, tokenizer, prompt=prompt,
                       max_tokens=128, sampler=greedy))
"
```

### Step 3: Fuse adapters

```bash
mlx_lm.fuse \
  --model mlx-community/Qwen3-4B-Instruct-2507-bf16 \
  --adapter-path spoke/adapters \
  --save-path spoke/fused
```

Produces a 7.5 GB bf16 model with adapters baked in.

### Step 4: Quantize for deployment

```bash
mlx_lm.convert \
  --hf-path spoke/fused \
  --mlx-path spoke/model \
  -q --q-bits 4
```

Produces a 2.1 GB 4-bit model.

### Step 5: Full evaluation

```bash
# Run benchmark on fine-tuned quantized model
python spoke/bench/run_benchmark.py \
  --model spoke/model \
  --prompt-mode generic
```

Note: after fine-tuning, we test with the **generic** prompt (not Spoke).
The model should have internalized the behavior — it shouldn't need
few-shot examples or detailed rules anymore. If generic accuracy is close
to the Spoke-prompt zero-shot accuracy, fine-tuning worked.

### Step 6: Clean up

```bash
rm -rf spoke/fused  # 7.5 GB, can re-create from adapters
```

Keep: `spoke/adapters/` (112 MB) and `spoke/model/` (2.1 GB).

---

## Evaluation

### Scoring

| Score | Meaning |
|-------|---------|
| **exact** | Output matches ideal after whitespace normalization |
| **semantic** | Correct transformation, minor formatting difference (case, punctuation) |
| **partial** | Some commands executed, others missed. Or close but wrong |
| **fail** | Wrong output, hallucination, or commands not executed |

**Accuracy** = (exact + semantic) / total

### Targets

| Metric | Baseline (zero-shot) | Target | Stretch |
|--------|---------------------|--------|---------|
| Overall accuracy | 25-58% | >85% | >92% |
| Spell-replace | 50% | >80% | >90% |
| Self-correction | 50% | >90% | >95% |
| Multi-command | 0-100%* | >70% | >85% |
| Avg latency (generic prompt) | 0.71s | <0.40s | <0.25s |

*Multi-command jumps from 0% (generic prompt) to 100% (Spoke prompt with
the exact example). True generalization is unknown until tested on novel
multi-command inputs.

### Latency Measurement

The benchmark measures total generation time per example. For deployment:
- **TTFT** (time to first token): Add streaming measurement if needed
- **Total latency**: What matters for our use case (need complete output)
- Test on **short** (5 words) and **long** (50 words) inputs separately

---

## Iteration Loop

```
                ┌──────────────────────────────┐
                │  Fine-tune (10 min)          │
                └──────────┬───────────────────┘
                           │
                           ▼
                ┌──────────────────────────────┐
                │  Evaluate on test set         │
                └──────────┬───────────────────┘
                           │
                    ┌──────┴──────┐
                    │             │
              Accuracy ≥ 85%   Accuracy < 85%
                    │             │
                    ▼             ▼
              Ship it!    Analyze per-category failures
                                  │
                                  ▼
                          Add 20-50 targeted examples
                          for failing categories
                                  │
                                  ▼
                          Re-train (update data path,
                          keep same config)
                                  │
                                  └──────────► loop back
```

### Diagnosing Failures

If accuracy is below target after first training run:

1. **Check per-category breakdown** — which categories are weakest?
2. **Read the actual outputs** — is the model close (formatting issues) or
   completely wrong (doesn't understand the task)?
3. **Check for patterns** — does it fail on long inputs? Novel words? Multiple
   commands? This tells you what to add to the training data.

Common fixes:
| Issue | Fix |
|-------|-----|
| Model is chatty, explains instead of transforming | Add more diverse examples. Increase `mask_prompt` effectiveness by shortening system prompt |
| Works on simple cases, fails on multi-command | Add more multi-command training examples (target: 80) |
| Spell-replace fails on novel words | Add more spell-replace examples with diverse phonetic patterns |
| Model over-corrects (removes too much) | Add examples where the correction is small, preserving most content |
| Val loss still high after 1K iters | Increase iters to 2000, or increase rank to 32 |
| Val loss low but accuracy low | The model learned the distribution but not the exact patterns. Add harder examples |

### Hyperparameter Adjustments

| If this happens... | Try this |
|-------|-----|
| Overfitting (train loss << val loss) | Increase dropout to 0.1, reduce iters |
| Underfitting (val loss plateaus high) | Increase rank to 32, or lr to 2e-5 |
| Memory issues | Reduce batch_size to 2, enable `grad_checkpoint: true` |
| Too slow | Reduce num_layers to 8 (less fine-tuning but faster) |

---

## Prompt Strategy

### Training Data Prompt (in JSONL)

Use the **short generic** system prompt for all training examples:

```
Clean the transcript by executing all verbal commands
(spell-outs, corrections, formatting, symbols, emoji).
Output ONLY the cleaned text.
```

**Why one prompt, not per-category?** The model should learn the behavior
from the (input, output) pairs, not from elaborate instructions. A short,
consistent prompt becomes a "mode selector" — when the model sees this
prompt, it knows to execute verbal commands.

### Production Inference Prompt (after fine-tuning)

Start with the same generic prompt. If accuracy is high enough (>85%),
ship it — simpler is better.

If accuracy needs a boost on specific categories, add the Spoke-style
dynamic prompt with triggered rules (but no few-shot examples — the
model shouldn't need them after fine-tuning):

```
base_instructions + core_rules + triggered_rules
```

The router already detects triggers, so appending 1 focused rule is cheap
(~40 extra tokens, <10ms latency impact).

---

## Deployment Integration

After fine-tuning, the model slots into the Spoke pipeline:

```
ASR (Parakeet/Moonshine)
        │
        ▼
   Regex Router ──── no triggers ──── pass through
        │
    triggers found
        │
        ▼
   mlx_lm.generate(
     model="spoke/model",           # 2.1 GB, 4-bit
     prompt=generic_system_prompt,   # ~25 tokens
     max_tokens=256,
     sampler=greedy,                 # temp=0 for deterministic output
   )
        │
        ▼
   Cleaned text → Display
```

### Memory Budget (16 GB MacBook)

```
macOS + apps:        ~4 GB
ASR (Parakeet):      ~1.2 GB
Spoke LLM (4-bit):   ~2.1 GB
KV cache:            ~0.3 GB
Headroom:            ~8.4 GB free
```

### Files to Ship

```
spoke/model/
├── config.json           # Model architecture config
├── model.safetensors     # 2.1 GB, 4-bit weights
├── special_tokens_map.json
├── tokenizer.json
└── tokenizer_config.json
```

Everything needed for `mlx_lm.load("spoke/model")`.

---

## File Map

```
spoke/
├── PLAN.md              # High-level project plan
├── DATAGEN.md           # Dataset curation pipeline
├── FINETUNE.md          # This doc — training pipeline
├── config.yaml          # mlx_lm.lora training config
├── bench/
│   ├── run_benchmark.py # Evaluation script (generic/task/spoke modes)
│   ├── prompts.py       # Spoke-style dynamic prompt templates
│   ├── test_set.json    # 12 sacred test examples
│   └── result_*.json    # Benchmark results
├── data/
│   ├── evals.csv        # 60 gold seed examples
│   ├── dummy/           # Pipeline test data (10+3+3)
│   ├── prompts/         # Datagen prompt templates (from Track A)
│   ├── raw/             # Raw Kimi output (from Track A)
│   ├── validated/       # Validator output (from Track A)
│   └── final/           # Merged dataset ready for training
│       ├── train.jsonl
│       ├── valid.jsonl
│       └── test.jsonl
├── adapters/            # LoRA weights after training
└── model/               # Quantized 4-bit model for deployment
```
