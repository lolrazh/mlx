#!/usr/bin/env python3
"""Modal cloud training for Spoke using Unsloth + Qwen3.5-4B.

Trains LoRA on Modal L40S GPU, exports merged bf16 model.
No QLoRA for Qwen3.5 — Unsloth recommends bf16 LoRA instead.

Usage:
    modal run spoke/cloud/train.py --run-name spoke-qwen35-t1
    modal run spoke/cloud/train.py --run-name spoke-qwen35-t1 --max-steps 3000 --learning-rate 2e-5
    modal run spoke/cloud/train.py --run-name spoke-qwen35-probe --max-steps 50 --eval-steps 0 --save-steps 0
"""

import modal

# ── Modal setup ──────────────────────────────────────────────

app = modal.App("spoke-training")

# Volumes for persistent storage across runs
model_cache = modal.Volume.from_name("spoke-model-cache", create_if_missing=True)
training_data = modal.Volume.from_name("spoke-training-data", create_if_missing=True)
output_vol = modal.Volume.from_name("spoke-output", create_if_missing=True)

# Clean CUDA build image. Avoid the interactive Unsloth Docker image because it
# boots Jupyter/SSH/Ollama services that are useless on Modal workers.
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install("ninja", "packaging", "wheel")
    .pip_install("unsloth", "wandb")
    .run_commands(
        "MAX_JOBS=4 CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST=8.9 "
        "python -m pip install --no-build-isolation --no-deps flash-attn"
    )
    .pip_install("flash-linear-attention")
    .run_commands(
        "CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST=8.9 "
        "python -m pip install --no-build-isolation --no-deps causal-conv1d"
    )
    .run_commands("python -m pip install --no-deps trl==0.22.2")
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
    timeout=10800,  # 3 hours: room for model load, full run, and merge export
)
def train(
    run_name: str = "spoke-qwen35-t1",
    model_name: str = "unsloth/Qwen3.5-4B",
    max_steps: int = 2000,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    max_seq_length: int = 512,
    packing: bool = True,
    gradient_checkpointing: bool = False,
    eval_steps: int = 200,
    save_steps: int = 500,
):
    import json
    import os
    import wandb

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["WANDB_PROJECT"] = "spoke"

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    eval_enabled = eval_steps > 0
    save_enabled = save_steps > 0

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
    print(
        "Applying LoRA: "
        f"r={rank}, alpha={lora_alpha}, dropout={lora_dropout}, "
        f"grad_ckpt={gradient_checkpointing}"
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth" if gradient_checkpointing else False,
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
        eval_dataset=valid_dataset if eval_enabled else None,
        args=SFTConfig(
            output_dir=output_dir,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            lr_scheduler_type="constant",
            optim="adamw_torch",  # adamw with wd=0 = plain adam (matches T2-v4)
            weight_decay=0.0,
            bf16=True,
            seed=42,
            dataset_text_field="text",
            dataset_num_proc=1,
            packing=packing,
            # Logging
            logging_steps=20,
            eval_strategy="steps" if eval_enabled else "no",
            eval_steps=eval_steps if eval_enabled else 200,
            save_strategy="steps" if save_enabled else "no",
            save_steps=save_steps if save_enabled else 500,
            save_total_limit=2 if save_enabled else None,
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
        num_proc=1,
    )

    # ── Verify masking ───────────────────────────────────────
    sample = trainer.train_dataset[0]
    labels = sample["labels"]
    total = len(labels)
    active = sum(1 for l in labels if l != -100)
    print(f"Masking check: {active}/{total} tokens active ({100*active/total:.1f}%)")

    # ── Train ────────────────────────────────────────────────
    print(
        "\nStarting training: "
        f"{max_steps} steps, lr={learning_rate}, batch={batch_size}, "
        f"accum={gradient_accumulation_steps}, max_seq={max_seq_length}, "
        f"packing={packing}, eval={'off' if not eval_enabled else eval_steps}, "
        f"save={'off' if not save_enabled else save_steps}"
    )
    try:
        trainer.train()

        # ── Export merged bf16 ───────────────────────────────
        merged_path = f"{output_dir}/merged"
        print(f"\nSaving merged bf16 model to {merged_path}...")
        model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

        output_vol.commit()

        print(f"\nTraining complete!")
        print(f"  Model saved to volume 'spoke-output' at /output/{run_name}/merged")
        print(f"  Download with: python spoke/cloud/download_model.py --run-name {run_name}")
    finally:
        if wandb.run is not None:
            print("\nFinalizing Weights & Biases run...")
            wandb.finish()


# ── CLI entrypoint ───────────────────────────────────────────

@app.local_entrypoint()
def main(
    run_name: str = "spoke-qwen35-t1",
    model_name: str = "unsloth/Qwen3.5-4B",
    max_steps: int = 2000,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    max_seq_length: int = 512,
    packing: bool = True,
    gradient_checkpointing: bool = False,
    eval_steps: int = 200,
    save_steps: int = 500,
):
    print(f"Starting cloud training: {run_name}")
    print(f"  Model: {model_name}")
    print(f"  Steps: {max_steps}, LR: {learning_rate}, Batch: {batch_size}")
    print(
        "  LoRA: "
        f"r={rank}, alpha={lora_alpha}, dropout={lora_dropout}, "
        f"grad_ckpt={gradient_checkpointing}"
    )
    print(f"  Grad accum: {gradient_accumulation_steps}")
    print(f"  Max seq length: {max_seq_length}")
    print(f"  Packing: {packing}")
    print(f"  Eval every: {'off' if eval_steps <= 0 else eval_steps} steps")
    print(f"  Save every: {'off' if save_steps <= 0 else save_steps} steps")
    print()

    train.remote(
        run_name=run_name,
        model_name=model_name,
        max_steps=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        rank=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        max_seq_length=max_seq_length,
        packing=packing,
        gradient_checkpointing=gradient_checkpointing,
        eval_steps=eval_steps,
        save_steps=save_steps,
    )
