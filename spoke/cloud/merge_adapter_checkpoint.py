#!/usr/bin/env python3
"""Merge a saved HF+PEFT adapter checkpoint into a standalone bf16 model.

Usage:
    modal run spoke/cloud/merge_adapter_checkpoint.py \
      --run-name spoke-qwen3-hf-v5-v3prompt-v2-20260305-2010 \
      --checkpoint-step 1200
"""

from __future__ import annotations

import os

import modal

app = modal.App("spoke-merge-hf-checkpoint")

model_cache = modal.Volume.from_name("spoke-model-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("spoke-output", create_if_missing=True)

standard_image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
    .pip_install(
        "transformers==5.3.0",
        "peft==0.14.0",
        "sentencepiece",
        "safetensors",
    )
)

# Modern-stack image for Mamba-hybrid models (Nemotron H) — see train_hf.py.
# Select with SPOKE_MAMBA_IMAGE=1.
mamba_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.9.*", "torchvision")
    .pip_install(
        "transformers==4.53.0",
        "peft==0.14.0",
        "sentencepiece",
        "safetensors",
        "packaging",
        "einops",
    )
    .pip_install(
        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.9cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
        "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.9cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
        extra_options="--no-deps",
    )
)

# Gemma 4 needs transformers>=5.5.2 + torch>=2.7 + peft>=0.19 and the FULL
# Gemma4ForConditionalGeneration (nested language_model.* keys). See train_hf.py.
# Select with SPOKE_GEMMA4_IMAGE=1.
gemma4_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch==2.8.*", "torchvision")
    .pip_install(
        "transformers==5.5.2",
        "peft==0.19.0",
        "accelerate",
        "sentencepiece",
        "safetensors",
    )
)

if os.environ.get("SPOKE_MAMBA_IMAGE") == "1":
    image = mamba_image
elif os.environ.get("SPOKE_GEMMA4_IMAGE") == "1":
    image = gemma4_image
else:
    image = standard_image


@app.function(
    image=image,
    gpu="L40S",
    volumes={
        "/model-cache": model_cache,
        "/output": output_vol,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=3600,
)
def merge_checkpoint(
    run_name: str,
    checkpoint_step: int,
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    output_run_name: str | None = None,
):
    import os
    from pathlib import Path

    import torch
    from peft import PeftModel
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        GenerationConfig,
    )

    os.environ["HF_HOME"] = "/model-cache"
    if os.getenv("HF_TOKEN"):
        os.environ["HUGGINGFACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    torch.backends.cuda.matmul.allow_tf32 = True

    adapter_dir = Path(f"/output/{run_name}/checkpoint-{checkpoint_step}")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Checkpoint adapter not found: {adapter_dir}")

    if output_run_name is None:
        output_run_name = f"{run_name}-ckpt{checkpoint_step}"
    merged_dir = Path(f"/output/{output_run_name}/merged")
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {model_name}")
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    _mtype = getattr(model_config, "model_type", "")
    _has_text_config = getattr(model_config, "text_config", None) is not None
    MULTIMODAL_TEXT_ONLY_TYPES = {"qwen3_5", "gemma3n", "mistral3"}
    is_multimodal_text_only = bool(_mtype in MULTIMODAL_TEXT_ONLY_TYPES and _has_text_config)
    is_gemma4 = bool(_mtype == "gemma4")
    effective_model_config = model_config.text_config if is_multimodal_text_only else model_config
    is_encoder_decoder = bool(getattr(effective_model_config, "is_encoder_decoder", False))
    if is_encoder_decoder:
        model_cls = AutoModelForSeq2SeqLM
    elif is_gemma4:
        # Full multimodal load so language_model.* decoder weights map (the
        # adapter was trained on this same full model). See train_hf.py #106.
        from transformers import Gemma4ForConditionalGeneration
        model_cls = Gemma4ForConditionalGeneration
    else:
        model_cls = AutoModelForCausalLM
    print(f"Detected architecture: {_mtype} ({'multimodal text-only' if is_multimodal_text_only else 'seq2seq' if is_encoder_decoder else 'causal'})")
    model_load_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    if is_multimodal_text_only:
        print(f"Detected {_mtype} multimodal config; forcing text-only merge path.")
        model_load_kwargs["config"] = effective_model_config
    if "t5gemma" in str(getattr(model_config, "model_type", "")).lower():
        model_load_kwargs["attn_implementation"] = "eager"
    base_model = model_cls.from_pretrained(
        model_name,
        **model_load_kwargs,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(adapter_dir),
            trust_remote_code=True,
        )
    except Exception as exc:
        # Checkpoints saved under transformers 5.x stamp tokenizer_class
        # "TokenizersBackend", which 4.x can't resolve. The tokenizer is
        # unchanged from the base model, so load it from there instead.
        print(f"Adapter tokenizer unreadable ({exc}); loading base tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading adapter: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    print("Merging adapter into base model...")
    merged = peft_model.merge_and_unload()

    print(f"Saving merged model to: {merged_dir}")
    # Strip sampling flags when do_sample is off (Nemotron H ships top_p=0.95
    # with greedy decoding; transformers 5.x refuses to save that combination).
    gen_config = getattr(merged, "generation_config", None)
    if gen_config is not None and not getattr(gen_config, "do_sample", False):
        gen_config.temperature = None
        gen_config.top_p = None
        gen_config.top_k = None
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    # Copy generation_config from the training run's final merged model if available,
    # otherwise from the base model. This ensures correct EOS/pad token IDs.
    final_merged_gen_config = Path(f"/output/{run_name}/merged/generation_config.json")
    if final_merged_gen_config.exists():
        import shutil
        shutil.copy(str(final_merged_gen_config), str(merged_dir / "generation_config.json"))
        print(f"  Copied generation_config from {final_merged_gen_config}")
    else:
        try:
            gen_config = GenerationConfig.from_pretrained(model_name)
            gen_config.save_pretrained(str(merged_dir))
            print(f"  Saved generation_config from base model: eos={gen_config.eos_token_id}")
        except Exception as e:
            print(f"  Warning: could not save generation_config: {e}")

    output_vol.commit()
    print("Merge complete.")
    print(f"  Source checkpoint: {adapter_dir}")
    print(f"  Output model: {merged_dir}")


@app.local_entrypoint()
def main(
    run_name: str,
    checkpoint_step: int,
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    output_run_name: str = "",
):
    merge_checkpoint.remote(
        run_name=run_name,
        checkpoint_step=checkpoint_step,
        model_name=model_name,
        output_run_name=output_run_name or None,
    )
