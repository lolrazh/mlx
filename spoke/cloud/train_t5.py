#!/usr/bin/env python3
"""Modal cloud training for Spoke encoder-decoder models (T5, Flan-T5, T5Gemma 2).

Seq2seq format (no chat templates):
  Input:  "Correct this transcription: {asr_text}"
  Output: "{cleaned_text}"

Usage:
    # T5Gemma 2 1B-1B (FlanEC recipe: lr=5e-5, AdamW, linear+warmup):
    modal run spoke/cloud/train_t5.py --model-name google/t5gemma-2-1b-1b --run-name spoke-t5gemma2-1b-v1

    # Flan-T5-large (full fine-tuning):
    modal run spoke/cloud/train_t5.py --model-name google/flan-t5-large --run-name spoke-flan-t5-large-v1

    # Quick probe:
    modal run spoke/cloud/train_t5.py --run-name spoke-t5gemma2-probe --max-steps 50 --eval-steps 0 --save-steps 0 --no-export-merged
"""

from __future__ import annotations

import modal

app = modal.App("spoke-training-t5")

model_cache = modal.Volume.from_name("spoke-model-cache", create_if_missing=True)
training_data = modal.Volume.from_name("spoke-training-data", create_if_missing=True)
output_vol = modal.Volume.from_name("spoke-output", create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
    .apt_install("git")
    .pip_install(
        "transformers==5.3.0",
        "accelerate==1.4.0",
        "datasets==3.2.0",
        "peft==0.14.0",
        "sentencepiece",
        "protobuf",
        "safetensors",
        "Pillow",
        "wandb",
    )
)

TASK_PREFIX = "Correct this transcription: "


@app.function(
    image=image,
    gpu="L40S",
    volumes={
        "/model-cache": model_cache,
        "/data": training_data,
        "/output": output_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=7200,
)
def train(
    run_name: str = "spoke-t5gemma2-1b-v1",
    model_name: str = "google/t5gemma-2-1b-1b",
    max_steps: int = 2000,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    max_source_length: int = 256,
    max_target_length: int = 256,
    use_lora: bool = False,
    rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.0,
    eval_steps: int = 100,
    save_steps: int = 200,
    save_total_limit: int = 3,
    data_dir: str = "/data/v4",
    export_merged: bool = True,
):
    import json
    import os

    import torch
    import wandb
    from datasets import Dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["WANDB_PROJECT"] = "spoke"
    if os.getenv("HF_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    torch.backends.cuda.matmul.allow_tf32 = True

    eval_enabled = eval_steps > 0
    save_enabled = save_steps > 0

    print(f"Loading model: {model_name}")
    # T5Gemma 2 needs AutoProcessor; Flan-T5 uses AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except (OSError, ValueError):
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(model_name)
        tokenizer = processor.tokenizer
        print("Using AutoProcessor.tokenizer (T5Gemma 2 path)")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
            target_modules="all-linear",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("Mode: LoRA fine-tuning")
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"Mode: Full fine-tuning — {trainable:,} / {total_params:,} params "
            f"({100 * trainable / total_params:.1f}%)"
        )

    # --- Data loading ---
    def load_jsonl(path: str):
        with open(path) as f:
            return [json.loads(line) for line in f]

    train_raw = load_jsonl(f"{data_dir}/train.jsonl")
    valid_raw = load_jsonl(f"{data_dir}/valid.jsonl")
    print(f"Data loaded: {len(train_raw)} train, {len(valid_raw)} valid")

    # Convert chat messages -> seq2seq pairs
    def chat_to_seq2seq(example):
        messages = example["messages"]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assistant_msg = next(m["content"] for m in messages if m["role"] == "assistant")
        return {
            "input_text": f"{TASK_PREFIX}{user_msg}",
            "target_text": assistant_msg,
        }

    train_pairs = [chat_to_seq2seq(ex) for ex in train_raw]
    valid_pairs = [chat_to_seq2seq(ex) for ex in valid_raw]

    # Tokenize
    eos_id = tokenizer.eos_token_id

    def tokenize_fn(examples):
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=max_source_length,
            truncation=True,
            padding=False,
        )
        labels = tokenizer(
            text_target=examples["target_text"],
            max_length=max_target_length,
            truncation=True,
            padding=False,
        )
        # Gemma tokenizer doesn't append EOS — add it so the model learns to stop
        if eos_id is not None:
            for i, ids in enumerate(labels["input_ids"]):
                if not ids or ids[-1] != eos_id:
                    labels["input_ids"][i] = ids + [eos_id]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = Dataset.from_list(train_pairs).map(
        tokenize_fn, batched=True, remove_columns=["input_text", "target_text"]
    )
    valid_dataset = (
        Dataset.from_list(valid_pairs).map(
            tokenize_fn, batched=True, remove_columns=["input_text", "target_text"]
        )
        if eval_enabled
        else None
    )

    # Token length stats
    input_lens = [len(ex["input_ids"]) for ex in train_dataset]
    label_lens = [len(ex["labels"]) for ex in train_dataset]
    print(
        f"Token lengths — input: median={sorted(input_lens)[len(input_lens)//2]}, "
        f"max={max(input_lens)} | target: median={sorted(label_lens)[len(label_lens)//2]}, "
        f"max={max(label_lens)}"
    )

    # Print samples
    for i in range(min(3, len(train_pairs))):
        print(f"\n--- Sample {i} ---")
        print(f"  In:  {train_pairs[i]['input_text'][:120]}")
        print(f"  Out: {train_pairs[i]['target_text'][:120]}")
    print()

    # Compute epochs for logging
    steps_per_epoch = len(train_dataset) / (batch_size * gradient_accumulation_steps)
    total_epochs = max_steps / steps_per_epoch
    print(f"Steps per epoch: {steps_per_epoch:.1f}, total epochs: {total_epochs:.1f}")

    # Data collator handles dynamic padding for seq2seq
    # Don't pass model= for T5Gemma 2 (incompatible prepare_decoder_input_ids_from_labels)
    is_t5gemma = "t5gemma" in model_name.lower()
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=None if is_t5gemma else model,
        padding=True,
        pad_to_multiple_of=8,
    )

    output_dir = f"/output/{run_name}"
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        bf16=True,
        seed=42,
        data_seed=42,
        logging_steps=10,
        eval_strategy="steps" if eval_enabled else "no",
        eval_steps=eval_steps if eval_enabled else None,
        save_strategy="steps" if save_enabled else "no",
        save_steps=save_steps if save_enabled else None,
        save_total_limit=save_total_limit if save_enabled else None,
        load_best_model_at_end=eval_enabled and save_enabled,
        metric_for_best_model="eval_loss" if eval_enabled and save_enabled else None,
        greater_is_better=False if eval_enabled and save_enabled else None,
        report_to="wandb",
        run_name=run_name,
        predict_with_generate=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print(
        f"\nStarting T5 training: {max_steps} steps, lr={learning_rate}, "
        f"batch={batch_size}, accum={gradient_accumulation_steps}, "
        f"mode={'lora' if use_lora else 'full'}, "
        f"scheduler={lr_scheduler_type}"
    )

    try:
        trainer.train()

        if trainer.state.best_model_checkpoint is not None:
            print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")

        if export_merged:
            merged_path = f"{output_dir}/merged"
            print(f"\nSaving model to {merged_path}")

            if use_lora:
                merged_model = trainer.model.merge_and_unload()
                merged_model.save_pretrained(merged_path, safe_serialization=True)
            else:
                trainer.model.save_pretrained(merged_path, safe_serialization=True)
            tokenizer.save_pretrained(merged_path)

            output_vol.commit()
            print(f"\nTraining complete! Model at /output/{run_name}/merged")
        else:
            print("\nTraining complete! Skipped export.")
    finally:
        if wandb.run is not None:
            wandb.finish()


@app.local_entrypoint()
def main(
    run_name: str = "spoke-t5gemma2-1b-v1",
    model_name: str = "google/t5gemma-2-1b-1b",
    max_steps: int = 2000,
    learning_rate: float = 5e-5,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    max_source_length: int = 256,
    max_target_length: int = 256,
    use_lora: bool = False,
    rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lr_scheduler_type: str = "linear",
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.0,
    eval_steps: int = 100,
    save_steps: int = 200,
    save_total_limit: int = 3,
    data_dir: str = "/data/v4",
    export_merged: bool = True,
):
    print(f"Starting encoder-decoder cloud training: {run_name}")
    print(f"  Model: {model_name}")
    print(f"  Steps: {max_steps}, LR: {learning_rate}, Batch: {batch_size}")
    print(f"  Mode: {'LoRA (r=' + str(rank) + ')' if use_lora else 'Full fine-tuning'}")
    print(f"  Max source: {max_source_length}, Max target: {max_target_length}")
    print(f"  Scheduler: {lr_scheduler_type}, Warmup: {warmup_ratio}")
    print(f"  Data: {data_dir}")
    print(f"  Eval: {'off' if eval_steps <= 0 else f'every {eval_steps} steps'}")
    print(f"  Save: {'off' if save_steps <= 0 else f'every {save_steps} steps'}")
    print(f"  Export: {'on' if export_merged else 'off'}")
    print()

    train.remote(
        run_name=run_name,
        model_name=model_name,
        max_steps=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        use_lora=use_lora,
        rank=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        data_dir=data_dir,
        export_merged=export_merged,
    )
