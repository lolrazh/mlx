#!/usr/bin/env python3
"""Modal cloud training for Spoke using Unsloth + Qwen3.5-4B.

Trains LoRA on Modal L40S GPU, exports merged bf16 model.
No QLoRA for Qwen3.5 — Unsloth recommends bf16 LoRA instead.

Usage:
    modal run spoke/cloud/train.py --run-name spoke-qwen35-t1
    modal run spoke/cloud/train.py --run-name spoke-qwen35-t1 --max-steps 3000 --learning-rate 2e-5
"""

import modal

# ── Modal setup ──────────────────────────────────────────────

app = modal.App("spoke-training")

# Volumes for persistent storage across runs
model_cache = modal.Volume.from_name("spoke-model-cache", create_if_missing=True)
training_data = modal.Volume.from_name("spoke-training-data", create_if_missing=True)
output_vol = modal.Volume.from_name("spoke-output", create_if_missing=True)

# Unsloth image — pin TRL to 0.22.2 (matches ALL official Unsloth notebooks)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("unsloth", "wandb")
    .run_commands("pip install --no-deps trl==0.22.2")
)


# ── Training function ────────────────────────────────────────

@app.function(
    image=image,
    gpu="L40S",
    volumes={
        "/model-cache": model_cache,
        "/data": training_data,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=3600,  # 1 hour max
)
def train(
    run_name: str = "spoke-qwen35-t1",
    model_name: str = "unsloth/Qwen3.5-4B",
    max_steps: int = 2000,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    rank: int = 8,
    lora_alpha: int = 16,
    max_seq_length: int = 512,
):
    import json
    import os

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["WANDB_PROJECT"] = "spoke"

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    # ── Load model with Unsloth ────────────────────────────
    print(f"Loading {model_name} (bf16 LoRA, no QLoRA)...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    print(f"Tokenizer after load: eos={tokenizer.eos_token} type={type(tokenizer).__name__}")

    # Fix EOS token — Unsloth replaces eos with <EOS_TOKEN> placeholder.
    # get_chat_template maps it back correctly (from Qwen3-4B-Instruct notebook).
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="qwen3-instruct",
    )
    print(f"Tokenizer after get_chat_template: eos={tokenizer.eos_token}")

    # ── LoRA config (T2-v4 proven values) ──────────────────
    print(f"Applying LoRA: r={rank}, alpha={lora_alpha}, dropout=0.05")
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        max_seq_length=max_seq_length,
    )

    # ── Load data ────────────────────────────────────────────
    def load_jsonl(path):
        with open(path) as f:
            return [json.loads(line) for line in f]

    train_data = load_jsonl("/data/train.jsonl")
    valid_data = load_jsonl("/data/valid.jsonl")
    print(f"Data loaded: {len(train_data)} train, {len(valid_data)} valid")

    # ── Format with chat template ────────────────────────────
    def format_example(example):
        try:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            # Fallback if chat template doesn't support enable_thinking
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        return {"text": text}

    # Remove original columns — collator can't batch nested dicts
    raw_train = Dataset.from_list(train_data)
    raw_valid = Dataset.from_list(valid_data)
    extra_cols = [c for c in raw_train.column_names if c != "text"]
    train_dataset = raw_train.map(format_example).remove_columns(extra_cols)
    valid_dataset = raw_valid.map(format_example).remove_columns(extra_cols)

    print(f"\n--- Sample formatted input ---")
    print(train_dataset[0]["text"][:500])
    print(f"--- End sample ---\n")

    # ── Trainer (matches Unsloth notebook pattern exactly) ──
    output_dir = f"/output/{run_name}"

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,  # Unsloth notebooks use tokenizer=, not processing_class=
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        args=SFTConfig(
            output_dir=output_dir,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type="constant",
            optim="adamw_torch",  # adamw with wd=0 = plain adam (matches T2-v4)
            weight_decay=0.0,
            bf16=True,
            seed=42,
            dataset_text_field="text",
            dataset_num_proc=1,
            # Logging
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=100,
            # wandb
            report_to="wandb",
            run_name=run_name,
        ),
    )

    # Mask system+user tokens — only train on assistant responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    # ── Verify masking ───────────────────────────────────────
    sample = trainer.train_dataset[0]
    labels = sample["labels"]
    total = len(labels)
    active = sum(1 for l in labels if l != -100)
    print(f"Masking check: {active}/{total} tokens active ({100*active/total:.1f}%)")

    # ── Train ────────────────────────────────────────────────
    print(f"\nStarting training: {max_steps} steps, lr={learning_rate}, batch={batch_size}")
    trainer.train()

    # ── Export merged bf16 ───────────────────────────────────
    merged_path = f"{output_dir}/merged"
    print(f"\nSaving merged bf16 model to {merged_path}...")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

    output_vol.commit()

    print(f"\nTraining complete!")
    print(f"  Model saved to volume 'spoke-output' at /output/{run_name}/merged")
    print(f"  Download with: python spoke/cloud/download_model.py --run-name {run_name}")


# ── CLI entrypoint ───────────────────────────────────────────

@app.local_entrypoint()
def main(
    run_name: str = "spoke-qwen35-t1",
    model_name: str = "unsloth/Qwen3.5-4B",
    max_steps: int = 2000,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    rank: int = 8,
    lora_alpha: int = 16,
    max_seq_length: int = 512,
):
    print(f"Starting cloud training: {run_name}")
    print(f"  Model: {model_name}")
    print(f"  Steps: {max_steps}, LR: {learning_rate}, Batch: {batch_size}")
    print(f"  LoRA: r={rank}, alpha={lora_alpha}")
    print(f"  Max seq length: {max_seq_length}")
    print()

    train.remote(
        run_name=run_name,
        model_name=model_name,
        max_steps=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        rank=rank,
        lora_alpha=lora_alpha,
        max_seq_length=max_seq_length,
    )
