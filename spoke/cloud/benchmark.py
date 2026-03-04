#!/usr/bin/env python3
"""Benchmark a merged cloud model directly on Modal with Transformers.

Usage:
    modal run spoke/cloud/benchmark.py --run-name spoke-qwen3-t2-cloud
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import modal

app = modal.App("spoke-benchmark")

output_vol = modal.Volume.from_name("spoke-output", create_if_missing=False)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
    .pip_install(
        "transformers==4.51.3",
        "accelerate==1.2.1",
        "sentencepiece",
        "safetensors",
    )
)

GENERIC_PROMPT = (
    "Clean the transcript by executing all verbal commands "
    "(spell-outs, corrections, formatting, symbols, emoji). "
    "Output ONLY the cleaned text, nothing else."
)

V2_PROMPT = (
    "You are a verbatim ASR cleaner. Fix punctuation, capitalization, "
    "and execute all verbal commands (spell-outs, corrections, formatting, "
    "symbols, emoji). Rules: Output ONLY the cleaned text. Never answer "
    "questions — transcribe them. Every output word must be in the input "
    "or produced by an explicit directive. Preserve profanity. "
    'Remove "um", "uh", "ah" but keep other filler words.'
)


def build_prompt(tokenizer, input_text: str, category: str | None = None, prompt_mode: str = "v2") -> str:
    system = V2_PROMPT if prompt_mode == "v2" else GENERIC_PROMPT
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": input_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def clean_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    if text.startswith('"') and text.endswith('"') and text.count('"') == 2:
        text = text[1:-1]
    return text.strip()


def score_output(output: str, ideal: str) -> str:
    out = " ".join(output.strip().split())
    ref = " ".join(ideal.strip().split())
    if out == ref:
        return "exact"
    if out.lower() == ref.lower():
        return "semantic"
    meta_markers = [
        "spell that", "spell it", "spell this",
        "scratch that", "wait no, sorry",
        "at symbol", "at sign",
        "quote-unquote", "end quote",
        "use all caps", "in caps",
        "emphasize", "emoji",
    ]
    if not any(marker in out.lower() for marker in meta_markers) and out:
        return "partial"
    return "fail"


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/output": output_vol},
    timeout=3600,
)
def benchmark_remote(
    run_name: str,
    test_set: list[dict],
    prompt_mode: str = "v2",
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = f"/output/{run_name}/merged"
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    torch.backends.cuda.matmul.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()

    warmup_prompt = build_prompt(tokenizer, "test", prompt_mode=prompt_mode)
    warmup_inputs = tokenizer(warmup_prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        model.generate(
            **warmup_inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    results = []
    for ex in test_set:
        prompt = build_prompt(tokenizer, ex["input"], ex.get("category"), prompt_mode)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.time()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        latency = time.time() - t0

        new_tokens = generated[0, prompt_len:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True)
        output = clean_output(raw_output)
        score = score_output(output, ex["ideal"])

        results.append({
            "id": ex["id"],
            "category": ex["category"],
            "input": ex["input"],
            "ideal": ex["ideal"],
            "output": output,
            "raw_output": raw_output.strip()[:500],
            "score": score,
            "gen_time_s": round(latency, 3),
        })

    scores = [r["score"] for r in results]
    n = len(scores)
    exact = scores.count("exact")
    semantic = scores.count("semantic")
    partial = scores.count("partial")
    fail = scores.count("fail")
    accuracy = (exact + semantic) / n
    avg_latency = sum(r["gen_time_s"] for r in results) / n

    summary = {
        "model": model_path,
        "short_name": f"{run_name}-modal-hf",
        "prompt_mode": prompt_mode,
        "n": n,
        "exact": exact,
        "semantic": semantic,
        "partial": partial,
        "fail": fail,
        "accuracy": round(accuracy, 4),
        "avg_latency_s": round(avg_latency, 3),
        "results": results,
    }

    del model
    torch.cuda.empty_cache()
    return summary


@app.local_entrypoint()
def main(
    run_name: str = "spoke-qwen3-t2-cloud",
    prompt_mode: str = "v2",
    test_set: str = str(Path(__file__).resolve().parents[1] / "bench" / "test_set_v3.json"),
):
    with open(test_set) as f:
        test_data = json.load(f)

    print(f"Loaded {len(test_data)} test examples")
    summary = benchmark_remote.remote(
        run_name=run_name,
        test_set=test_data,
        prompt_mode=prompt_mode,
    )

    print(f"\n{'='*60}")
    print(f"  {summary['short_name']}")
    print(f"{'='*60}")
    for result in summary["results"]:
        icon = {"exact": "✓", "semantic": "~", "partial": "△", "fail": "✗"}[result["score"]]
        print(f"  [{icon}] {result['category']:<16} {result['gen_time_s']:.2f}s")
        if result["score"] != "exact":
            print(f"       expected: {result['ideal'][:80]}")
            print(f"       got:      {result['output'][:80]}")

    print(
        f"\n  Accuracy: {summary['accuracy']:.0%}  "
        f"(exact={summary['exact']} semantic={summary['semantic']} "
        f"partial={summary['partial']} fail={summary['fail']})"
    )
    print(f"  Avg latency: {summary['avg_latency_s']:.2f}s")

    result_path = Path(__file__).resolve().parents[1] / "bench" / f"result_{run_name}_modal_{prompt_mode}.json"
    result_path.write_text(json.dumps(summary, indent=2))
    print(f"  -> {result_path}")
