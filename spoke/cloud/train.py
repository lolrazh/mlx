#!/usr/bin/env python3
"""Modal cloud training for Spoke using Unsloth + Qwen3.5-4B.

Trains LoRA on Modal L40S GPU, exports merged bf16 model.
No QLoRA for Qwen3.5 — Unsloth recommends bf16 LoRA instead.

Usage:
    modal run spoke/cloud/train.py --run-name spoke-qwen35-t1
    modal run spoke/cloud/train.py --run-name spoke-qwen35-t2 --max-steps 3000 --learning-rate 2e-5
"""

import modal

# ── Modal setup ──────────────────────────────────────────────

app = modal.App("spoke-training")

# Volumes for persistent storage across runs
model_cache = modal.Volume.from_name("spoke-model-cache", create_if_missing=True)
training_data = modal.Volume.from_name("spoke-training-data", create_if_missing=True)
output_vol = modal.Volume.from_name("spoke-output", create_if_missing=True)

# Unsloth image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "unsloth[cu124-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git",
        "wandb",
    )
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
    from pathlib import Path

    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    # ── Environment ──────────────────────────────────────────
    os.environ["HF_HOME"] = "/model-cache"
    os.environ["WANDB_PROJECT"] = "spoke"

    # ── Load model ───────────────────────────────────────────
    print(f"Loading {model_name} (bf16 LoRA, no QLoRA)...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=False,    # No QLoRA for Qwen3.5!
        load_in_16bit=True,    # bf16 LoRA
        full_finetuning=False,
    )

    # ── LoRA config ──────────────────────────────────────────
    print(f"Applying LoRA: r={rank}, alpha={lora_alpha}, dropout=0.05")
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=lora_alpha,      # scale = alpha/r = 16/8 = 2.0 (matches T2-v4)
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
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
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,  # No CoT for Spoke
        )
        return {"text": text}

    train_dataset = Dataset.from_list(train_data).map(format_example)
    valid_dataset = Dataset.from_list(valid_data).map(format_example)

    # Print a sample to verify formatting
    print(f"\n--- Sample formatted input ---")
    print(train_dataset[0]["text"][:500])
    print(f"--- End sample ---\n")

    # ── Training config ──────────────────────────────────────
    output_dir = f"/output/{run_name}"

    training_args = SFTConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type="constant",     # Flat LR (matches mlx-lm default)
        optim="paged_adam_32bit",          # Plain Adam, not AdamW (AdamW caused quant regression)
        bf16=True,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        seed=42,
        # Logging
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        # wandb
        report_to="wandb",
        run_name=run_name,
        # Performance
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
    )

    # ── Trainer with response-only masking ───────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
    )

    # Mask system+user tokens — only train on assistant responses
    # Uses Qwen chat template markers
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
    print(f"  Expected ~10-20% (assistant responses only)")

    # ── Train ────────────────────────────────────────────────
    print(f"\nStarting training: {max_steps} steps, lr={learning_rate}, batch={batch_size}")
    trainer.train()

    # ── Export merged bf16 ───────────────────────────────────
    merged_path = f"{output_dir}/merged"
    print(f"\nSaving merged bf16 model to {merged_path}...")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

    # Commit volumes so files persist
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
