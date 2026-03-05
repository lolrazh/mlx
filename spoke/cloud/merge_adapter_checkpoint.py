#!/usr/bin/env python3
"""Merge a saved HF+PEFT adapter checkpoint into a standalone bf16 model.

Usage:
    modal run spoke/cloud/merge_adapter_checkpoint.py \
      --run-name spoke-qwen3-hf-v5-v3prompt-v2-20260305-2010 \
      --checkpoint-step 1200
"""

from __future__ import annotations

import modal

app = modal.App("spoke-merge-hf-checkpoint")

model_cache = modal.Volume.from_name("spoke-model-cache", create_if_missing=True)
output_vol = modal.Volume.from_name("spoke-output", create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
    .pip_install(
        "transformers==4.53.0",
        "peft==0.14.0",
        "sentencepiece",
        "safetensors",
    )
)


@app.function(
    image=image,
    gpu="L40S",
    volumes={
        "/model-cache": model_cache,
        "/output": output_vol,
    },
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.environ["HF_HOME"] = "/model-cache"
    torch.backends.cuda.matmul.allow_tf32 = True

    adapter_dir = Path(f"/output/{run_name}/checkpoint-{checkpoint_step}")
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Checkpoint adapter not found: {adapter_dir}")

    if output_run_name is None:
        output_run_name = f"{run_name}-ckpt{checkpoint_step}"
    merged_dir = Path(f"/output/{output_run_name}/merged")
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(adapter_dir),
        trust_remote_code=True,
    )

    print(f"Loading adapter: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    print("Merging adapter into base model...")
    merged = peft_model.merge_and_unload()

    print(f"Saving merged model to: {merged_dir}")
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

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
