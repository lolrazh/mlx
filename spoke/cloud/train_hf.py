#!/usr/bin/env python3
"""Modal cloud training for Spoke using pure HF + PEFT (no Unsloth).

Goal: stricter parity against local MLX training by removing Unsloth patches and
training on the official Qwen base model.

Usage:
    modal run spoke/cloud/train_hf.py --run-name spoke-qwen3-hf-parity-smoke --max-steps 50 --eval-steps 0 --save-steps 0 --no-export-merged
    modal run spoke/cloud/train_hf.py --run-name spoke-qwen3-hf-v5-overfitguard-v1
"""

from __future__ import annotations

import modal

app = modal.App("spoke-training-hf")

model_cache = modal.Volume.from_name("spoke-model-cache", create_if_missing=True)
training_data = modal.Volume.from_name("spoke-training-data", create_if_missing=True)
output_vol = modal.Volume.from_name("spoke-output", create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
    .apt_install("git", "build-essential")
    .pip_install(
        "transformers==4.51.3",
        "accelerate==1.4.0",
        "datasets==3.2.0",
        "peft==0.14.0",
        "sentencepiece",
        "safetensors",
        "wandb",
    )
)

V2_SYSTEM_PROMPT = (
    "You are a verbatim ASR cleaner. Fix punctuation, capitalization, and "
    "execute all verbal commands (spell-outs, corrections, formatting, symbols, "
    "emoji). Rules: Output ONLY the cleaned text. Never answer questions — "
    "transcribe them. Every output word must be in the input or produced by an "
    "explicit directive. Preserve profanity. Remove \"um\", \"uh\", \"ah\" but keep "
    "other filler words."
)

V3_SYSTEM_PROMPT = (
    "You are a verbatim ASR cleaner. Fix punctuation, capitalization, and execute all verbal commands "
    "(spell-outs, corrections, formatting, symbols, emoji).\n"
    "Output ONLY the cleaned text. Never answer questions — transcribe them. Every output word must be "
    "in the input or produced by an explicit directive. Preserve profanity. Remove \"um\", \"uh\", \"ah\" "
    "but keep other filler words.\n"
    "Self-corrections (\"sorry\", \"scratch that\", \"actually\"): drop the wrong part, keep the correction.\n"
    "Spell commands: letters combine into a word replacing the closest phonetic match; drop directive words.\n"
    "Quote-unquote wraps nearest word(s). Quote...end quote wraps everything between.\n"
    "CamelCase: split unless a known brand. At-symbol: insert @, drop instruction. Emphasis/bold: ALL CAPS.\n"
    "Multiple directives in one input: execute all of them. Apply corrections and spelling first, then "
    "formatting. Last conflicting directive wins."
)


@app.function(
    image=image,
    gpu="L40S",
    volumes={
        "/model-cache": model_cache,
        "/data": training_data,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
    timeout=10800,
)
def train(
    run_name: str = "spoke-qwen3-hf-parity",
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    max_steps: int = 1200,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    max_seq_length: int = 512,
    optimizer: str = "adam",
    max_grad_norm: float = 1.0,
    eval_steps: int = 100,
    save_steps: int = 100,
    system_prompt_mode: str = "as_is",
    export_merged: bool = True,
):
    import json
    import os
    import numpy as np
    import torch
    import wandb
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["WANDB_PROJECT"] = "spoke"
    torch.backends.cuda.matmul.allow_tf32 = True

    eval_enabled = eval_steps > 0
    save_enabled = save_steps > 0
    if eval_enabled and save_enabled and (save_steps % eval_steps != 0):
        raise ValueError(
            f"save_steps ({save_steps}) must be a multiple of eval_steps ({eval_steps}) "
            "when selecting best checkpoint by eval_loss."
        )

    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    template = tokenizer.chat_template or ""
    if "<think>" in template:
        raise RuntimeError(
            f"Tokenizer template for {model_name} contains <think>; expected no-thinking template."
        )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print("Gradient checkpointing: enabled")
    else:
        print("Gradient checkpointing: disabled")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def load_jsonl(path: str):
        with open(path) as f:
            return [json.loads(line) for line in f]

    system_prompt_overrides = {
        "as_is": None,
        "v2": V2_SYSTEM_PROMPT,
        "v3": V3_SYSTEM_PROMPT,
    }
    if system_prompt_mode not in system_prompt_overrides:
        raise ValueError(
            f"system_prompt_mode must be one of {sorted(system_prompt_overrides)}, got: {system_prompt_mode}"
        )
    forced_system_prompt = system_prompt_overrides[system_prompt_mode]

    def maybe_override_system_prompt(example):
        if forced_system_prompt is None:
            return example
        messages = example.get("messages")
        if not messages or messages[0].get("role") != "system":
            raise ValueError("Expected first message to be role=system in chat-formatted dataset.")
        patched_messages = [dict(m) for m in messages]
        patched_messages[0]["content"] = forced_system_prompt
        return {"messages": patched_messages}

    train_data = load_jsonl("/data/train.jsonl")
    valid_data = load_jsonl("/data/valid.jsonl")
    if forced_system_prompt is not None:
        train_data = [maybe_override_system_prompt(ex) for ex in train_data]
        valid_data = [maybe_override_system_prompt(ex) for ex in valid_data]
    print(f"Data loaded: {len(train_data)} train, {len(valid_data)} valid")
    print(f"System prompt mode: {system_prompt_mode}")

    def render_chat_text_from_messages(messages, add_generation_prompt=False):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if "<think>" in text:
            raise RuntimeError("Rendered training text contains <think> unexpectedly.")
        return text

    def tokenize_chat(messages, add_generation_prompt=False):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            return_dict=False,
        )

    def build_mlx_style_example(example):
        messages = example["messages"]
        input_ids = tokenize_chat(messages)
        prompt_ids = tokenize_chat(
            messages[:-1],
            add_generation_prompt=messages[-1].get("role") == "assistant",
        )
        input_ids = input_ids[:max_seq_length]
        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {
            "input_ids": input_ids,
            "labels": labels,
            "length": len(input_ids),
        }

    train_dataset = Dataset.from_list([build_mlx_style_example(ex) for ex in train_data])
    valid_dataset = (
        Dataset.from_list([build_mlx_style_example(ex) for ex in valid_data])
        if eval_enabled
        else None
    )

    sample_formatted = render_chat_text_from_messages(train_data[0]["messages"])
    print("\n--- Sample formatted input ---")
    print(sample_formatted[:500])
    print("--- End sample ---\n")

    def parity_data_collator(features):
        lengths = [len(feature["input_ids"]) for feature in features]
        max_length_in_batch = 1 + 32 * ((max(lengths) + 32 - 1) // 32)
        max_length_in_batch = min(max_length_in_batch, max_seq_length)

        batch_input_ids = torch.zeros((len(features), max_length_in_batch), dtype=torch.long)
        batch_labels = torch.full((len(features), max_length_in_batch), -100, dtype=torch.long)

        for row, feature in enumerate(features):
            input_ids = feature["input_ids"][:max_length_in_batch]
            labels = feature["labels"][:max_length_in_batch]
            length = len(input_ids)
            batch_input_ids[row, :length] = torch.tensor(input_ids, dtype=torch.long)
            batch_labels[row, :length] = torch.tensor(labels, dtype=torch.long)
            # Match MLX default_loss mask semantics for padded rows.
            if length < max_length_in_batch:
                batch_labels[row, length] = 0

        return {
            "input_ids": batch_input_ids[:, :-1],
            "labels": batch_labels[:, 1:],
        }

    def mlx_style_loss(outputs, labels, num_items_in_batch=None):
        logits = outputs.logits
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
        )

    class MLXBatchOrderSampler(torch.utils.data.Sampler):
        def __init__(self, dataset, batch_size, seed):
            self._batch_size = batch_size
            self._rng = np.random.RandomState(seed)
            sorted_indices = sorted(range(len(dataset)), key=lambda idx: dataset[idx]["length"])
            full_batches = len(sorted_indices) // batch_size
            self._batches = [
                sorted_indices[i * batch_size : (i + 1) * batch_size]
                for i in range(full_batches)
            ]

        def __iter__(self):
            batch_order = self._rng.permutation(len(self._batches))
            flattened = [
                idx
                for batch_idx in batch_order
                for idx in self._batches[batch_idx]
            ]
            return iter(flattened)

        def __len__(self):
            return len(self._batches) * self._batch_size

    class MLXLengthSortedSampler(torch.utils.data.Sampler):
        def __init__(self, dataset):
            self._indices = sorted(range(len(dataset)), key=lambda idx: dataset[idx]["length"])

        def __iter__(self):
            return iter(self._indices)

        def __len__(self):
            return len(self._indices)

    class MLXParityTrainer(Trainer):
        def __init__(self, *args, use_true_adam=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False
            self._use_true_adam = use_true_adam

        def create_optimizer(self):
            if self.optimizer is None and self._use_true_adam:
                print("Using torch.optim.Adam for parity.")
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                self.optimizer = torch.optim.Adam(
                    trainable_params,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
                return self.optimizer
            return super().create_optimizer()

        def _get_train_sampler(self, train_dataset=None):
            if train_dataset is None:
                train_dataset = self.train_dataset
            return MLXBatchOrderSampler(
                train_dataset,
                self.args.per_device_train_batch_size,
                self.args.data_seed or self.args.seed,
            )

        def _get_eval_sampler(self, eval_dataset):
            if eval_dataset is None:
                eval_dataset = self.eval_dataset
            return MLXLengthSortedSampler(eval_dataset)

    output_dir = f"/output/{run_name}"
    trainer = MLXParityTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=parity_data_collator,
        compute_loss_func=mlx_style_loss,
        use_true_adam=(optimizer == "adam"),
        args=TrainingArguments(
            output_dir=output_dir,
            max_steps=max_steps,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            lr_scheduler_type="constant",
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            weight_decay=0.0,
            max_grad_norm=max_grad_norm,
            bf16=True,
            gradient_checkpointing=gradient_checkpointing,
            seed=42,
            data_seed=42,
            remove_unused_columns=False,
            label_names=["labels"],
            logging_steps=10,
            eval_strategy="steps" if eval_enabled else "no",
            eval_steps=eval_steps if eval_enabled else None,
            save_strategy="steps" if save_enabled else "no",
            save_steps=save_steps if save_enabled else 100,
            save_total_limit=2 if save_enabled else None,
            load_best_model_at_end=eval_enabled and save_enabled,
            metric_for_best_model="eval_loss" if eval_enabled and save_enabled else None,
            greater_is_better=False if eval_enabled and save_enabled else None,
            report_to="wandb",
            run_name=run_name,
        ),
    )

    sample = trainer.train_dataset[0]
    labels = sample["labels"]
    total = len(labels)
    active = sum(1 for l in labels if l != -100)
    print(f"Masking check: {active}/{total} tokens active ({100*active/total:.1f}%)")

    print(
        "\nStarting HF+PEFT training: "
        f"{max_steps} steps, lr={learning_rate}, batch={batch_size}, "
        f"accum={gradient_accumulation_steps}, max_seq={max_seq_length}, "
        f"optim={optimizer}, max_grad_norm={max_grad_norm}, "
        f"system_prompt={system_prompt_mode}, "
        f"grad_ckpt={'on' if gradient_checkpointing else 'off'}, "
        f"eval={'off' if not eval_enabled else eval_steps}, "
        f"save={'off' if not save_enabled else save_steps}, "
        f"best_ckpt={'on' if eval_enabled and save_enabled else 'off'}, "
        f"export={'on' if export_merged else 'off'}"
    )

    try:
        trainer.train()
        if trainer.state.best_model_checkpoint is not None:
            print(f"Best checkpoint by eval_loss: {trainer.state.best_model_checkpoint}")

        if export_merged:
            adapter_path = f"{output_dir}/adapter"
            merged_path = f"{output_dir}/merged"

            print(f"\nSaving adapter to {adapter_path}")
            trainer.model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)

            print(f"Saving merged bf16 model to {merged_path}")
            merged_model = trainer.model.merge_and_unload()
            merged_model.save_pretrained(merged_path, safe_serialization=True)
            tokenizer.save_pretrained(merged_path)

            output_vol.commit()
            print("\nTraining complete!")
            print(f"  Adapter: /output/{run_name}/adapter")
            print(f"  Merged model: /output/{run_name}/merged")
        else:
            print("\nTraining complete! Skipped merged export for faster probing.")
    finally:
        if wandb.run is not None:
            print("\nFinalizing Weights & Biases run...")
            wandb.finish()


@app.local_entrypoint()
def main(
    run_name: str = "spoke-qwen3-hf-parity",
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    max_steps: int = 1200,
    learning_rate: float = 1e-5,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    max_seq_length: int = 512,
    optimizer: str = "adam",
    max_grad_norm: float = 1.0,
    eval_steps: int = 100,
    save_steps: int = 100,
    system_prompt_mode: str = "as_is",
    export_merged: bool = True,
):
    print(f"Starting pure HF cloud training: {run_name}")
    print(f"  Model: {model_name}")
    print(f"  Steps: {max_steps}, LR: {learning_rate}, Batch: {batch_size}")
    print(
        "  LoRA: "
        f"r={rank}, alpha={lora_alpha}, dropout={lora_dropout}"
    )
    print(f"  Optimizer: {optimizer}")
    print(f"  Max grad norm: {max_grad_norm}")
    print(f"  Grad accum: {gradient_accumulation_steps}")
    print(f"  Grad checkpointing: {'on' if gradient_checkpointing else 'off'}")
    print(f"  Max seq length: {max_seq_length}")
    print(f"  System prompt mode: {system_prompt_mode}")
    print(f"  Eval every: {'off' if eval_steps <= 0 else eval_steps} steps")
    print(f"  Save every: {'off' if save_steps <= 0 else save_steps} steps")
    print(f"  Best checkpoint by eval_loss: {'on' if eval_steps > 0 and save_steps > 0 else 'off'}")
    print(f"  Export merged: {'on' if export_merged else 'off'}")
    print()

    train.remote(
        run_name=run_name,
        model_name=model_name,
        max_steps=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        rank=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        max_seq_length=max_seq_length,
        optimizer=optimizer,
        max_grad_norm=max_grad_norm,
        eval_steps=eval_steps,
        save_steps=save_steps,
        system_prompt_mode=system_prompt_mode,
        export_merged=export_merged,
    )
