#!/usr/bin/env python3
"""Modal cloud training for Spoke using Unsloth + Qwen3-4B.

Trains LoRA on Modal L40S GPU, exports merged bf16 model.
No QLoRA here — use bf16 LoRA for parity with the best local Qwen3 run.

Usage:
    modal run spoke/cloud/train.py --run-name spoke-qwen3-probe --max-steps 50 --batch-size 8 --eval-steps 0 --save-steps 0 --no-export-merged
    modal run spoke/cloud/train.py --run-name spoke-qwen3-t2-cloud --best-local-parity
    modal run spoke/cloud/train.py --run-name spoke-qwen3-unsloth --unsloth-recommended
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
    # Do not source-build flash-attn in the hot path. It makes short Modal
    # probes unusably slow to start. Revisit with a prebuilt wheel or cached
    # image once the faster trainer settings are dialed in.
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
    run_name: str = "spoke-qwen3-cloud",
    model_name: str = "unsloth/Qwen3-4B-Instruct-2507",
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
    optimizer: str = "adamw_torch",
    max_grad_norm: float = 1.0,
    eval_steps: int = 200,
    save_steps: int = 500,
    export_merged: bool = True,
    best_local_parity: bool = False,
    ultra_quality: bool = False,
    unsloth_recommended: bool = False,
    unsloth_epochs: float = 2.0,
    use_rslora: bool = False,
):
    import math
    import json
    import os
    import numpy as np
    import torch
    import wandb

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["WANDB_PROJECT"] = "spoke"

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    from datasets import Dataset
    from transformers import AutoTokenizer, Trainer, TrainingArguments
    from trl import SFTTrainer, SFTConfig

    selected_profiles = [best_local_parity, ultra_quality, unsloth_recommended]
    if sum(bool(x) for x in selected_profiles) > 1:
        raise ValueError("Choose only one profile: --best-local-parity OR --ultra-quality OR --unsloth-recommended.")

    if best_local_parity:
        # Match the successful local T2-v4 training recipe more closely for
        # quality comparisons instead of probe throughput.
        if run_name == "spoke-qwen3-cloud":
            run_name = "spoke-qwen3-t2-cloud"
        lora_dropout = 0.05
        packing = False
        optimizer = "adam"
        # Match mlx_lm default behavior more closely: no gradient clipping.
        max_grad_norm = 0.0
        eval_steps = 50
        save_steps = 100
        print("Applying best-local parity profile (T2-v4-like settings).")
    elif ultra_quality:
        # Higher-quality Unsloth profile:
        # - keep MLX-parity data path (no packing, adam, prompt-mask labels)
        # - increase adapter capacity with rsLoRA
        # - run longer by default
        if run_name == "spoke-qwen3-cloud":
            run_name = "spoke-qwen3-ultra-quality"
        if max_steps == 2000:
            max_steps = 2500
        learning_rate = 8e-6
        rank = 32
        lora_alpha = 64
        lora_dropout = 0.05
        packing = False
        optimizer = "adam"
        max_grad_norm = 0.0
        eval_steps = 50
        save_steps = 100
        use_rslora = True
        print(
            "Applying ultra-quality profile "
            f"(steps={max_steps}, lr={learning_rate}, r={rank}, "
            f"alpha={lora_alpha}, dropout={lora_dropout}, rsLoRA={use_rslora})."
        )
    elif unsloth_recommended:
        # Follow Unsloth guidance instead of MLX parity:
        # - higher LR for LoRA
        # - no LoRA dropout fast path
        # - packing enabled
        # - avoid hard-coding huge step counts; use epoch-targeted steps
        if run_name == "spoke-qwen3-cloud":
            run_name = "spoke-qwen3-unsloth"
        learning_rate = 2e-4
        rank = 16
        lora_alpha = 16
        lora_dropout = 0.0
        packing = True
        optimizer = "adamw_torch"
        max_grad_norm = 1.0
        eval_steps = 50
        save_steps = 100
        print(
            "Applying Unsloth-recommended profile "
            f"(lr={learning_rate}, r={rank}, dropout={lora_dropout}, packing={packing})."
        )

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

    def enforce_no_thinking_chat_template(tok, source_model_name: str):
        template = tok.chat_template or ""
        if "<think>" not in template:
            print("Chat template is already no-thinking.")
            return tok

        fallback_template_model = None
        if "qwen3-4b-instruct-2507" in source_model_name.lower():
            # Match the local MLX tokenizer template used by the 100% run.
            fallback_template_model = "mlx-community/Qwen3-4B-Instruct-2507-bf16"

        if not fallback_template_model:
            raise RuntimeError(
                "Tokenizer chat template contains <think> and no no-thinking fallback "
                f"is configured for model '{source_model_name}'."
            )

        print(
            "Tokenizer chat template includes <think>. "
            f"Replacing with no-thinking template from {fallback_template_model}."
        )
        ref_tok = AutoTokenizer.from_pretrained(
            fallback_template_model,
            trust_remote_code=True,
        )
        ref_template = ref_tok.chat_template or ""
        if "<think>" in ref_template:
            raise RuntimeError(
                f"Fallback tokenizer {fallback_template_model} still contains <think>; aborting."
            )
        tok.chat_template = ref_template

        if "<think>" in (tok.chat_template or ""):
            raise RuntimeError("Failed to apply no-thinking chat template.")
        print("No-thinking chat template applied successfully.")
        return tok

    tokenizer = enforce_no_thinking_chat_template(tokenizer, model_name)

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
        use_rslora=use_rslora,
    )

    # ── Load data ────────────────────────────────────────────
    def load_jsonl(path):
        with open(path) as f:
            return [json.loads(line) for line in f]

    train_data = load_jsonl("/data/train.jsonl")
    valid_data = load_jsonl("/data/valid.jsonl")
    print(f"Data loaded: {len(train_data)} train, {len(valid_data)} valid")
    if unsloth_recommended and max_steps == 2000:
        effective_batch = max(1, batch_size * gradient_accumulation_steps)
        max_steps = max(1, math.ceil((len(train_data) / effective_batch) * unsloth_epochs))
        print(
            "Unsloth profile auto-steps: "
            f"{unsloth_epochs:.2f} epochs × {len(train_data)} rows / "
            f"(batch {batch_size} × accum {gradient_accumulation_steps}) -> {max_steps} steps"
        )

    # ── Format with chat template ────────────────────────────
    def render_chat_text(example):
        messages = example["messages"]
        return render_chat_text_from_messages(messages)

    def render_chat_text_from_messages(messages, add_generation_prompt=False):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if "<think>" in text:
            raise RuntimeError(
                "Thinking tags detected in rendered prompt; no-thinking template enforcement failed."
            )
        return text

    def tokenize_chat(messages, add_generation_prompt=False):
        # Force no-thinking tokenization path for parity mode.
        rendered = render_chat_text_from_messages(
            messages,
            add_generation_prompt=add_generation_prompt,
        )
        return tokenizer(rendered, add_special_tokens=False)["input_ids"]

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

    if best_local_parity or ultra_quality:
        train_dataset = Dataset.from_list([build_mlx_style_example(ex) for ex in train_data])
        valid_dataset = (
            Dataset.from_list([build_mlx_style_example(ex) for ex in valid_data])
            if eval_enabled
            else None
        )
    else:
        def format_example(example):
            return {"text": render_chat_text(example)}

        # Remove original columns — collator can't batch nested dicts
        raw_train = Dataset.from_list(train_data)
        raw_valid = Dataset.from_list(valid_data)
        extra_cols = [c for c in raw_train.column_names if c != "text"]
        train_dataset = raw_train.map(format_example).remove_columns(extra_cols)
        valid_dataset = raw_valid.map(format_example).remove_columns(extra_cols)

    sample_formatted = render_chat_text(train_data[0])
    if "<think>" in sample_formatted:
        raise RuntimeError("Thinking tags detected in formatted training text. Aborting to avoid mismatch.")
    print(f"\n--- Sample formatted input ---")
    print(sample_formatted[:500])
    print(f"--- End sample ---\n")
    # ── Trainer (matches Unsloth notebook pattern exactly) ──
    output_dir = f"/output/{run_name}"

    if best_local_parity or ultra_quality:
        print("Using mlx-style parity trainer: pretokenized dataset + mask_prompt labels.")
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
                    print("Using true torch.optim.Adam for strict parity mode.")
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
                # HF TrainingArguments has no plain "adam" option in this stack.
                # When requested, MLXParityTrainer.create_optimizer() injects
                # torch.optim.Adam directly.
                optim="adamw_torch",
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1e-8,
                weight_decay=0.0,
                max_grad_norm=max_grad_norm,
                bf16=True,
                seed=42,
                data_seed=42,
                remove_unused_columns=False,
                label_names=["labels"],
                logging_steps=10,
                eval_strategy="steps" if eval_enabled else "no",
                eval_steps=eval_steps if eval_enabled else None,
                save_strategy="steps" if save_enabled else "no",
                save_steps=save_steps if save_enabled else 500,
                save_total_limit=2 if save_enabled else None,
                report_to="wandb",
                run_name=run_name,
            ),
        )
    else:
        if optimizer == "adam":
            print("Requested optimizer=adam is only supported in --best-local-parity mode; using adamw_torch here.")
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
                optim="adamw_torch",
                weight_decay=0.0,
                max_grad_norm=max_grad_norm,
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
        f"optim={optimizer}, "
        f"max_grad_norm={max_grad_norm}, "
        f"rsLoRA={use_rslora}, "
        f"packing={packing}, eval={'off' if not eval_enabled else eval_steps}, "
        f"save={'off' if not save_enabled else save_steps}, "
        f"export={'on' if export_merged else 'off'}"
    )
    try:
        trainer.train()

        if export_merged:
            # ── Export merged bf16 ───────────────────────────
            merged_path = f"{output_dir}/merged"
            print(f"\nSaving merged bf16 model to {merged_path}...")
            model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")

            output_vol.commit()

            print(f"\nTraining complete!")
            print(f"  Model saved to volume 'spoke-output' at /output/{run_name}/merged")
            print(f"  Download with: python spoke/cloud/download_model.py --run-name {run_name}")
        else:
            print("\nTraining complete! Skipped merged export for faster probing.")
    finally:
        if wandb.run is not None:
            print("\nFinalizing Weights & Biases run...")
            wandb.finish()


# ── CLI entrypoint ───────────────────────────────────────────

@app.local_entrypoint()
def main(
    run_name: str = "spoke-qwen3-cloud",
    model_name: str = "unsloth/Qwen3-4B-Instruct-2507",
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
    optimizer: str = "adamw_torch",
    max_grad_norm: float = 1.0,
    eval_steps: int = 200,
    save_steps: int = 500,
    export_merged: bool = True,
    best_local_parity: bool = False,
    ultra_quality: bool = False,
    unsloth_recommended: bool = False,
    unsloth_epochs: float = 2.0,
    use_rslora: bool = False,
):
    selected_profiles = [best_local_parity, ultra_quality, unsloth_recommended]
    if sum(bool(x) for x in selected_profiles) > 1:
        raise ValueError("Choose only one profile: --best-local-parity OR --ultra-quality OR --unsloth-recommended.")

    if best_local_parity:
        if run_name == "spoke-qwen3-cloud":
            run_name = "spoke-qwen3-t2-cloud"
        lora_dropout = 0.05
        packing = False
        optimizer = "adam"
        max_grad_norm = 0.0
        eval_steps = 50
        save_steps = 100
    elif ultra_quality:
        if run_name == "spoke-qwen3-cloud":
            run_name = "spoke-qwen3-ultra-quality"
        if max_steps == 2000:
            max_steps = 2500
        learning_rate = 8e-6
        rank = 32
        lora_alpha = 64
        lora_dropout = 0.05
        packing = False
        optimizer = "adam"
        max_grad_norm = 0.0
        eval_steps = 50
        save_steps = 100
        use_rslora = True
    elif unsloth_recommended:
        if run_name == "spoke-qwen3-cloud":
            run_name = "spoke-qwen3-unsloth"
        learning_rate = 2e-4
        rank = 16
        lora_alpha = 16
        lora_dropout = 0.0
        packing = True
        optimizer = "adamw_torch"
        max_grad_norm = 1.0
        eval_steps = 50
        save_steps = 100

    print(f"Starting cloud training: {run_name}")
    print(f"  Model: {model_name}")
    print(f"  Steps: {max_steps}, LR: {learning_rate}, Batch: {batch_size}")
    print(
        "  LoRA: "
        f"r={rank}, alpha={lora_alpha}, dropout={lora_dropout}, "
        f"grad_ckpt={gradient_checkpointing}, rsLoRA={use_rslora}"
    )
    print(f"  Optimizer: {optimizer}")
    print(f"  Max grad norm: {max_grad_norm}")
    print(f"  Grad accum: {gradient_accumulation_steps}")
    print(f"  Max seq length: {max_seq_length}")
    print(f"  Packing: {packing}")
    print(f"  Eval every: {'off' if eval_steps <= 0 else eval_steps} steps")
    print(f"  Save every: {'off' if save_steps <= 0 else save_steps} steps")
    print(f"  Export merged: {'on' if export_merged else 'off'}")
    print(f"  Best-local parity: {'on' if best_local_parity else 'off'}")
    print(f"  Ultra-quality: {'on' if ultra_quality else 'off'}")
    print(f"  Unsloth-recommended: {'on' if unsloth_recommended else 'off'}")
    if unsloth_recommended:
        print(f"  Unsloth epochs target: {unsloth_epochs:.2f}")
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
        optimizer=optimizer,
        max_grad_norm=max_grad_norm,
        eval_steps=eval_steps,
        save_steps=save_steps,
        export_merged=export_merged,
        best_local_parity=best_local_parity,
        ultra_quality=ultra_quality,
        unsloth_recommended=unsloth_recommended,
        unsloth_epochs=unsloth_epochs,
        use_rslora=use_rslora,
    )
