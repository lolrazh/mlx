#!/usr/bin/env python3
"""Modal cloud training for Spoke using pure HF + PEFT (no Unsloth).

Goal: stricter parity against local MLX training by removing Unsloth patches and
training on the official Qwen base model.

Usage:
    modal run spoke/cloud/train_hf.py --run-name spoke-qwen3-hf-parity-smoke --max-steps 50 --eval-steps 0 --save-steps 0 --no-export-merged
    modal run spoke/cloud/train_hf.py --run-name spoke-qwen3-hf-v5-overfitguard-v1
"""

from __future__ import annotations

import re
import modal

app = modal.App("spoke-training-hf")

model_cache = modal.Volume.from_name("spoke-model-cache", create_if_missing=True)
training_data = modal.Volume.from_name("spoke-training-data", create_if_missing=True)
output_vol = modal.Volume.from_name("spoke-output", create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
    .apt_install("git", "build-essential")
    .pip_install(
        "transformers==5.3.0",
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
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
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
    max_seq_length: int = 256,
    max_target_length: int = 256,
    optimizer: str = "adam",
    max_grad_norm: float = 1.0,
    lr_scheduler_type: str = "constant",
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.0,
    eval_steps: int = 100,
    save_steps: int = 100,
    save_total_limit: int = 5,
    data_dir: str = "/data",
    system_prompt_mode: str = "as_is",
    export_merged: bool = True,
    use_rslora: bool = False,
    loss_mode: str = "standard",
    epo_edit_weight: float = 3.0,
):
    import difflib
    import json
    import os
    import numpy as np
    import torch
    import wandb
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    os.environ["HF_HOME"] = "/model-cache"
    os.environ["WANDB_PROJECT"] = "spoke"
    if os.getenv("HF_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    torch.backends.cuda.matmul.allow_tf32 = True

    eval_enabled = eval_steps > 0
    save_enabled = save_steps > 0
    if eval_enabled and save_enabled and (save_steps % eval_steps != 0):
        raise ValueError(
            f"save_steps ({save_steps}) must be a multiple of eval_steps ({eval_steps}) "
            "when selecting best checkpoint by eval_loss."
        )

    print(f"Loading base model: {model_name}")
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    _mtype = getattr(model_config, "model_type", "")
    _has_text_config = getattr(model_config, "text_config", None) is not None
    # Multimodal models with a usable text-only decoder path
    MULTIMODAL_TEXT_ONLY_TYPES = {"qwen3_5", "gemma3n"}
    is_multimodal_text_only = bool(_mtype in MULTIMODAL_TEXT_ONLY_TYPES and _has_text_config)
    effective_model_config = model_config.text_config if is_multimodal_text_only else model_config
    is_encoder_decoder = bool(getattr(effective_model_config, "is_encoder_decoder", False))
    if is_multimodal_text_only:
        print(f"Model family: {_mtype} multimodal -> forcing text-only causal LM path")
    else:
        print(f"Model family: {'encoder-decoder (seq2seq)' if is_encoder_decoder else 'decoder-only (causal)'}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    empty_think_re = re.compile(r"<think>\s*</think>\s*", re.DOTALL)

    def has_disallowed_think_markers(text: str) -> bool:
        cleaned = empty_think_re.sub("", text)
        return "<think>" in cleaned or "</think>" in cleaned

    system_probe_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "test"},
    ]
    user_probe_messages = [{"role": "user", "content": "test"}]
    chat_template_kwargs: dict[str, bool] = {}
    supports_enable_thinking = False
    try:
        tokenizer.apply_chat_template(
            user_probe_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        supports_enable_thinking = True
        chat_template_kwargs["enable_thinking"] = False
        print("Tokenizer supports enable_thinking=False; forcing no-thinking chat templates.")
    except TypeError:
        supports_enable_thinking = False

    system_role_supported = True
    try:
        probe_text = tokenizer.apply_chat_template(
            system_probe_messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs,
        )
        if has_disallowed_think_markers(probe_text):
            raise RuntimeError(
                f"Tokenizer for {model_name} renders <think> unexpectedly."
            )
    except Exception as exc:
        if "System role not supported" in str(exc):
            system_role_supported = False
            probe_text = tokenizer.apply_chat_template(
                user_probe_messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_template_kwargs,
            )
            if has_disallowed_think_markers(probe_text):
                raise RuntimeError(
                    f"Tokenizer template for {model_name} renders <think>; no-thinking unsupported."
                )
            print("Tokenizer does not support system role; folding system prompt into first user turn.")
        else:
            raise

    def normalize_messages_for_template(messages):
        normalized = [dict(m) for m in messages]
        if system_role_supported:
            return normalized
        if normalized and normalized[0].get("role") == "system":
            system_text = normalized.pop(0).get("content", "").strip()
            user_idx = next(
                (idx for idx, msg in enumerate(normalized) if msg.get("role") == "user"),
                None,
            )
            if user_idx is None:
                normalized.insert(0, {"role": "user", "content": system_text})
            else:
                original_user = normalized[user_idx].get("content", "")
                merged_user = f"{system_text}\n\n{original_user}".strip()
                normalized[user_idx]["content"] = merged_user
        return normalized

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|endoftext|>"})

    if is_encoder_decoder:
        model_cls = AutoModelForSeq2SeqLM
    else:
        model_cls = AutoModelForCausalLM
    model_load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if is_multimodal_text_only:
        model_load_kwargs["config"] = effective_model_config
    if "t5gemma" in model_name.lower():
        model_load_kwargs["attn_implementation"] = "eager"
    model = model_cls.from_pretrained(
        model_name,
        **model_load_kwargs,
    )
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print("Gradient checkpointing: enabled")
    else:
        print("Gradient checkpointing: disabled")

    peft_task_type = "SEQ_2_SEQ_LM" if is_encoder_decoder else "CAUSAL_LM"
    if is_encoder_decoder:
        target_modules = "all-linear"
    else:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    print(f"LoRA task type: {peft_task_type}")
    print(f"LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=peft_task_type,
        target_modules=target_modules,
        use_rslora=use_rslora,
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

    train_data = load_jsonl(f"{data_dir}/train.jsonl")
    valid_data = load_jsonl(f"{data_dir}/valid.jsonl")
    if forced_system_prompt is not None:
        train_data = [maybe_override_system_prompt(ex) for ex in train_data]
        valid_data = [maybe_override_system_prompt(ex) for ex in valid_data]
    print(f"Data loaded: {len(train_data)} train, {len(valid_data)} valid")
    print(f"System prompt mode: {system_prompt_mode}")

    def render_chat_text_from_messages(messages, add_generation_prompt=False):
        normalized_messages = normalize_messages_for_template(messages)
        try:
            text = tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **chat_template_kwargs,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                **chat_template_kwargs,
            )
        if has_disallowed_think_markers(text):
            raise RuntimeError("Rendered training text contains <think> unexpectedly.")
        return text

    def tokenize_chat(messages, add_generation_prompt=False):
        normalized_messages = normalize_messages_for_template(messages)
        try:
            return tokenizer.apply_chat_template(
                normalized_messages,
                add_generation_prompt=add_generation_prompt,
                return_dict=False,
                **chat_template_kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                normalized_messages,
                return_dict=False,
                **chat_template_kwargs,
            )

    stop_token_ids_raw = getattr(model.generation_config, "eos_token_id", None)
    if stop_token_ids_raw is None:
        stop_token_ids_raw = tokenizer.eos_token_id
    if stop_token_ids_raw is None:
        stop_token_ids = []
    elif isinstance(stop_token_ids_raw, int):
        stop_token_ids = [int(stop_token_ids_raw)]
    else:
        stop_token_ids = [int(token_id) for token_id in stop_token_ids_raw]
    preferred_stop_token_id = stop_token_ids[-1] if stop_token_ids else None

    use_epo = loss_mode == "epo"
    if use_epo:
        print(f"EPO mode: edit_weight={epo_edit_weight}")

    def get_edit_char_mask(source_text: str, target_text: str) -> list[bool]:
        """Per-character edit mask for target text. True = edit (not in LCS)."""
        sm = difflib.SequenceMatcher(None, source_text, target_text)
        is_edit = [True] * len(target_text)
        for match in sm.get_matching_blocks():
            for k in range(match.size):
                is_edit[match.b + k] = False
        return is_edit

    def build_causal_example(example):
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
        result = {
            "input_ids": input_ids,
            "labels": labels,
            "length": len(input_ids),
        }
        if use_epo:
            token_weights = [1.0] * len(input_ids)
            user_text = next(
                (m["content"] for m in messages if m["role"] == "user"), ""
            )
            assistant_text = messages[-1]["content"]
            char_edit_mask = get_edit_char_mask(user_text, assistant_text)
            assistant_enc = tokenizer(
                assistant_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            for i, (start, end) in enumerate(assistant_enc["offset_mapping"]):
                token_idx = prompt_len + i
                if token_idx >= len(token_weights):
                    break
                if any(char_edit_mask[start:end]):
                    token_weights[token_idx] = epo_edit_weight
            result["token_weights"] = token_weights
        return result

    def build_seq2seq_example(example):
        messages = example["messages"]
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError("Expected final message role=assistant for seq2seq example.")
        source_messages = messages[:-1]
        source_ids = tokenize_chat(source_messages, add_generation_prompt=True)
        full_ids = tokenize_chat(messages, add_generation_prompt=False)
        input_ids = source_ids[:max_seq_length]
        if full_ids[: len(source_ids)] == source_ids:
            label_ids = full_ids[len(source_ids):]
        else:
            target_text = messages[-1]["content"]
            label_ids = tokenizer(
                target_text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_target_length,
            )["input_ids"]
        if preferred_stop_token_id is not None:
            if len(label_ids) >= max_target_length:
                label_ids = label_ids[:max_target_length]
                label_ids[-1] = preferred_stop_token_id
            elif preferred_stop_token_id not in label_ids:
                label_ids = label_ids + [preferred_stop_token_id]
        else:
            label_ids = label_ids[:max_target_length]
        if not label_ids:
            raise RuntimeError("Seq2seq label construction produced empty labels.")
        return {
            "input_ids": input_ids,
            "labels": label_ids,
            "length": len(input_ids),
        }

    build_example = build_seq2seq_example if is_encoder_decoder else build_causal_example
    train_dataset = Dataset.from_list([build_example(ex) for ex in train_data])
    valid_dataset = (
        Dataset.from_list([build_example(ex) for ex in valid_data])
        if eval_enabled
        else None
    )
    if is_encoder_decoder and stop_token_ids:
        probe_count = min(64, len(train_dataset))
        missing_stop = 0
        for idx in range(probe_count):
            labels = train_dataset[idx]["labels"]
            if not any(token_id in labels for token_id in stop_token_ids):
                missing_stop += 1
        if missing_stop:
            raise RuntimeError(
                "Seq2seq stop-token preflight failed: "
                f"{missing_stop}/{probe_count} examples missing stop tokens {stop_token_ids}."
            )

    if is_encoder_decoder:
        sample_messages = train_data[0]["messages"]
        sample_source = render_chat_text_from_messages(
            sample_messages[:-1],
            add_generation_prompt=True,
        )
        sample_target = sample_messages[-1]["content"]
        print("\n--- Sample formatted source ---")
        print(sample_source[:500])
        print("--- Sample target ---")
        print(sample_target[:300])
        sample_labels = build_seq2seq_example(train_data[0])["labels"]
        sample_has_stop = (
            any(token_id in sample_labels for token_id in stop_token_ids)
            if stop_token_ids
            else False
        )
        print("--- Seq2seq stop token ids ---")
        print(stop_token_ids)
        print("--- Sample label tail ---")
        print(sample_labels[-24:])
        print("--- Sample labels include stop token ---")
        print(sample_has_stop)
        print("--- End sample ---\n")
    else:
        sample_formatted = render_chat_text_from_messages(train_data[0]["messages"])
        print("\n--- Sample formatted input ---")
        print(sample_formatted[:500])
        print("--- End sample ---\n")

    def parity_data_collator(features):
        lengths = [len(feature["input_ids"]) for feature in features]
        max_length_in_batch = 1 + 32 * ((max(lengths) + 32 - 1) // 32)
        max_length_in_batch = min(max_length_in_batch, max_seq_length)

        has_weights = "token_weights" in features[0]
        batch_input_ids = torch.zeros((len(features), max_length_in_batch), dtype=torch.long)
        batch_labels = torch.full((len(features), max_length_in_batch), -100, dtype=torch.long)
        if has_weights:
            batch_weights = torch.ones((len(features), max_length_in_batch), dtype=torch.float32)

        for row, feature in enumerate(features):
            input_ids = feature["input_ids"][:max_length_in_batch]
            labels = feature["labels"][:max_length_in_batch]
            length = len(input_ids)
            batch_input_ids[row, :length] = torch.tensor(input_ids, dtype=torch.long)
            batch_labels[row, :length] = torch.tensor(labels, dtype=torch.long)
            # Match MLX default_loss mask semantics for padded rows.
            if length < max_length_in_batch:
                batch_labels[row, length] = 0
            if has_weights:
                weights = feature["token_weights"][:max_length_in_batch]
                batch_weights[row, :length] = torch.tensor(weights, dtype=torch.float32)

        result = {
            "input_ids": batch_input_ids[:, :-1],
            "labels": batch_labels[:, 1:],
        }
        if has_weights:
            result["token_weights"] = batch_weights[:, 1:]
        return result

    def seq2seq_data_collator(features):
        input_lengths = [len(feature["input_ids"]) for feature in features]
        label_lengths = [len(feature["labels"]) for feature in features]

        max_input_length = 1 + 32 * ((max(input_lengths) + 32 - 1) // 32)
        max_input_length = min(max_input_length, max_seq_length)
        max_label_length = 1 + 32 * ((max(label_lengths) + 32 - 1) // 32)
        max_label_length = min(max_label_length, max_target_length)

        batch_input_ids = torch.full(
            (len(features), max_input_length),
            tokenizer.pad_token_id,
            dtype=torch.long,
        )
        batch_attention_mask = torch.zeros((len(features), max_input_length), dtype=torch.long)
        batch_labels = torch.full((len(features), max_label_length), -100, dtype=torch.long)

        for row, feature in enumerate(features):
            input_ids = feature["input_ids"][:max_input_length]
            labels = feature["labels"][:max_label_length]
            input_length = len(input_ids)
            label_length = len(labels)
            batch_input_ids[row, :input_length] = torch.tensor(input_ids, dtype=torch.long)
            batch_attention_mask[row, :input_length] = 1
            batch_labels[row, :label_length] = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
        }

    data_collator = seq2seq_data_collator if is_encoder_decoder else parity_data_collator

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
        def __init__(self, *args, use_true_adam=False, epo_mode=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_accepts_loss_kwargs = False
            self._use_true_adam = use_true_adam
            self._epo_mode = epo_mode

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            token_weights = inputs.pop("token_weights", None)
            outputs = model(**inputs)
            logits = outputs.logits
            if self._epo_mode and token_weights is not None:
                loss_per_token = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                mask = (labels.reshape(-1) != -100).float()
                weights = token_weights.reshape(-1) * mask
                loss = (loss_per_token * weights).sum() / weights.sum().clamp(min=1.0)
            else:
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
            return (loss, outputs) if return_outputs else loss

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
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=weight_decay,
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
        save_total_limit=save_total_limit if save_enabled else None,
        load_best_model_at_end=eval_enabled and save_enabled,
        metric_for_best_model="eval_loss" if eval_enabled and save_enabled else None,
        greater_is_better=False if eval_enabled and save_enabled else None,
        report_to="wandb",
        run_name=run_name,
    )
    trainer = MLXParityTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        use_true_adam=(optimizer == "adam"),
        epo_mode=use_epo,
        args=training_args,
    )

    sample = trainer.train_dataset[0]
    labels = sample["labels"]
    total = len(labels)
    active = sum(1 for l in labels if l != -100)
    print(f"Masking check: {active}/{total} tokens active ({100*active/total:.1f}%)")

    if use_epo:
        total_edit = 0
        total_copy = 0
        for ex in trainer.train_dataset:
            weights = ex.get("token_weights", [])
            labs = ex["labels"]
            for w, l in zip(weights, labs):
                if l != -100:
                    if w > 1.0:
                        total_edit += 1
                    else:
                        total_copy += 1
        total_active = total_edit + total_copy
        print(
            f"EPO stats: {total_edit}/{total_active} edit tokens "
            f"({100*total_edit/max(total_active,1):.1f}%), "
            f"{total_copy}/{total_active} copy tokens "
            f"({100*total_copy/max(total_active,1):.1f}%), "
            f"edit_weight={epo_edit_weight}"
        )

    print(
        "\nStarting HF+PEFT training: "
        f"{max_steps} steps, lr={learning_rate}, batch={batch_size}, "
        f"accum={gradient_accumulation_steps}, max_seq={max_seq_length}, "
        f"max_target={max_target_length}, "
        f"optim={optimizer}, max_grad_norm={max_grad_norm}, "
        f"lr_sched={lr_scheduler_type}, warmup={warmup_ratio}, wd={weight_decay}, "
        f"system_prompt={system_prompt_mode}, "
        f"arch={'seq2seq' if is_encoder_decoder else 'causal'}, "
        f"grad_ckpt={'on' if gradient_checkpointing else 'off'}, "
        f"eval={'off' if not eval_enabled else eval_steps}, "
        f"save={'off' if not save_enabled else save_steps}, "
        f"save_total_limit={save_total_limit if save_enabled else 'off'}, "
        f"best_ckpt={'on' if eval_enabled and save_enabled else 'off'}, "
        f"export={'on' if export_merged else 'off'}, "
        f"loss={loss_mode}"
        + (f" (edit_weight={epo_edit_weight})" if use_epo else "")
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
    max_seq_length: int = 256,
    max_target_length: int = 256,
    optimizer: str = "adam",
    max_grad_norm: float = 1.0,
    lr_scheduler_type: str = "constant",
    warmup_ratio: float = 0.0,
    weight_decay: float = 0.0,
    eval_steps: int = 100,
    save_steps: int = 100,
    save_total_limit: int = 5,
    data_dir: str = "/data",
    system_prompt_mode: str = "as_is",
    export_merged: bool = True,
    use_rslora: bool = False,
    loss_mode: str = "standard",
    epo_edit_weight: float = 3.0,
):
    print(f"Starting pure HF cloud training: {run_name}")
    print(f"  Model: {model_name}")
    print(f"  Steps: {max_steps}, LR: {learning_rate}, Batch: {batch_size}")
    print(
        "  LoRA: "
        f"r={rank}, alpha={lora_alpha}, dropout={lora_dropout}, rsLoRA={'on' if use_rslora else 'off'}"
    )
    print(f"  Optimizer: {optimizer}")
    print(f"  Max grad norm: {max_grad_norm}")
    print(f"  LR scheduler: {lr_scheduler_type}")
    print(f"  Warmup ratio: {warmup_ratio}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Grad accum: {gradient_accumulation_steps}")
    print(f"  Grad checkpointing: {'on' if gradient_checkpointing else 'off'}")
    print(f"  Max seq length: {max_seq_length}")
    print(f"  Max target length: {max_target_length}")
    print(f"  Data dir: {data_dir}")
    print(f"  System prompt mode: {system_prompt_mode}")
    print(f"  Eval every: {'off' if eval_steps <= 0 else eval_steps} steps")
    print(f"  Save every: {'off' if save_steps <= 0 else save_steps} steps")
    print(f"  Save total limit: {save_total_limit if save_steps > 0 else 'off'}")
    print(f"  Best checkpoint by eval_loss: {'on' if eval_steps > 0 and save_steps > 0 else 'off'}")
    print(f"  Export merged: {'on' if export_merged else 'off'}")
    print(f"  Loss mode: {loss_mode}")
    if loss_mode == "epo":
        print(f"  EPO edit weight: {epo_edit_weight}")
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
        max_target_length=max_target_length,
        optimizer=optimizer,
        max_grad_norm=max_grad_norm,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        data_dir=data_dir,
        system_prompt_mode=system_prompt_mode,
        export_merged=export_merged,
        use_rslora=use_rslora,
        loss_mode=loss_mode,
        epo_edit_weight=epo_edit_weight,
    )
