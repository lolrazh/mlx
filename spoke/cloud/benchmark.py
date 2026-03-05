#!/usr/bin/env python3
"""Benchmark a merged cloud model directly on Modal with Transformers.

Usage:
    modal run spoke/cloud/benchmark.py --run-name spoke-qwen3-t2-cloud
    modal run spoke/cloud/benchmark.py --run-name spoke-qwen3-t2-cloud --suite broad58
    modal run spoke/cloud/benchmark.py --run-name spoke-qwen3-t2-cloud --suite core23
"""

from __future__ import annotations

import json
import re
import time
import hashlib
from pathlib import Path

import modal

app = modal.App("spoke-benchmark")

output_vol = modal.Volume.from_name("spoke-output", create_if_missing=False)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime")
    .pip_install(
        "transformers==4.53.0",
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

V3_PROMPT = (
    "You are a verbatim ASR cleaner. Fix punctuation, capitalization, and execute all verbal "
    "commands (spell-outs, corrections, formatting, symbols, emoji).\n"
    "Output ONLY the cleaned text. Never answer questions — transcribe them. Every output "
    "word must be in the input or produced by an explicit directive. Preserve profanity. "
    'Remove "um", "uh", "ah" but keep other filler words.\n'
    'Self-corrections ("sorry", "scratch that", "actually"): drop the wrong part, keep the correction.\n'
    "Spell commands: letters combine into a word replacing the closest phonetic match; drop directive words.\n"
    "Quote-unquote wraps nearest word(s). Quote...end quote wraps everything between.\n"
    "CamelCase: split unless a known brand. At-symbol: insert @, drop instruction. Emphasis/bold: ALL CAPS.\n"
    "Multiple directives in one input: execute all of them. Apply corrections and spelling first, then formatting. "
    "Last conflicting directive wins."
)


EMPTY_THINK_RE = re.compile(r"<think>\s*</think>\s*", flags=re.DOTALL)


def has_disallowed_think_markers(text: str) -> bool:
    cleaned = EMPTY_THINK_RE.sub("", text)
    return "<think>" in cleaned or "</think>" in cleaned


def build_prompt(tokenizer, input_text: str, category: str | None = None, prompt_mode: str = "v2") -> str:
    if prompt_mode == "v2":
        system = V2_PROMPT
    elif prompt_mode == "v3":
        system = V3_PROMPT
    else:
        system = GENERIC_PROMPT
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": input_text},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    if has_disallowed_think_markers(prompt):
        raise RuntimeError(
            "Prompt contains <think>; no-thinking template enforcement failed."
        )
    return prompt


def clean_output(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    if text.startswith('"') and text.endswith('"') and text.count('"') == 2:
        text = text[1:-1]
    return text.strip()


def enforce_no_thinking_chat_template(tokenizer, model_id_hint: str):
    template = tokenizer.chat_template or ""
    probe_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "test"},
    ]
    try:
        probe = tokenizer.apply_chat_template(
            probe_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not has_disallowed_think_markers(probe):
            print("Tokenizer supports no-thinking via enable_thinking=False.")
            return tokenizer
    except TypeError:
        pass

    if "<think>" not in template:
        print("Chat template is already no-thinking.")
        return tokenizer

    fallback_template_model = None
    if "qwen3-4b-instruct-2507" in model_id_hint.lower():
        fallback_template_model = "mlx-community/Qwen3-4B-Instruct-2507-bf16"

    if not fallback_template_model:
        raise RuntimeError(
            "Tokenizer chat template contains <think> and no fallback is configured "
            f"for model '{model_id_hint}'."
        )

    print(
        "Tokenizer chat template includes <think>. "
        f"Replacing with no-thinking template from {fallback_template_model}."
    )
    from transformers import AutoTokenizer as HFTokenizer

    ref_tok = HFTokenizer.from_pretrained(
        fallback_template_model,
        trust_remote_code=True,
    )
    ref_template = ref_tok.chat_template or ""
    if has_disallowed_think_markers(ref_template):
        raise RuntimeError(
            f"Fallback tokenizer {fallback_template_model} still contains <think>; aborting."
        )

    tokenizer.chat_template = ref_template
    if has_disallowed_think_markers(tokenizer.chat_template or ""):
        raise RuntimeError("Failed to apply no-thinking chat template.")
    print("No-thinking chat template applied successfully.")
    return tokenizer


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
    tokenizer = enforce_no_thinking_chat_template(tokenizer, model_path)
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
    test_set: str = "",
    suite: str = "core23",
):
    bench_dir = Path(__file__).resolve().parents[1] / "bench"
    suite_map = {
        "core23": bench_dir / "test_set_v3.json",
        "broad58": bench_dir / "test_set_evals.json",
        "legacy12": bench_dir / "test_set.json",
    }
    if suite not in suite_map:
        raise ValueError(f"suite must be one of {sorted(suite_map)}, got: {suite}")

    resolved_test_set = Path(test_set) if test_set else suite_map[suite]
    with open(resolved_test_set) as f:
        test_data = json.load(f)

    test_set_sha256 = hashlib.sha256(
        resolved_test_set.read_bytes()
    ).hexdigest()
    print(f"Loaded {len(test_data)} test examples from {resolved_test_set}")
    print(f"Test set SHA256: {test_set_sha256}")

    summary = benchmark_remote.remote(
        run_name=run_name,
        test_set=test_data,
        prompt_mode=prompt_mode,
    )
    summary["test_set_path"] = str(resolved_test_set)
    summary["test_set_name"] = resolved_test_set.name
    summary["test_set_sha256"] = test_set_sha256
    summary["suite"] = suite

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

    result_path = (
        Path(__file__).resolve().parents[1]
        / "bench"
        / f"result_{run_name}_modal_{prompt_mode}_{resolved_test_set.stem}.json"
    )
    result_path.write_text(json.dumps(summary, indent=2))
    print(f"  -> {result_path}")
