#!/usr/bin/env python3
"""
Zero-shot benchmark for ASR post-processing LLMs.
Tests candidate models against the sacred test set.

Usage:
    python spoke/bench/run_benchmark.py --model qwen3-4b
    python spoke/bench/run_benchmark.py --model mlx-community/Qwen3-4B-Instruct-2507-4bit
    python spoke/bench/run_benchmark.py --all
"""

import json
import re
import time
import argparse
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

BENCH_DIR = Path(__file__).parent
GREEDY = make_sampler(temp=0.0)

# ── Models ──────────────────────────────────────────────────
MODELS = {
    "qwen3-4b": "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "qwen3-4b-bf16": "mlx-community/Qwen3-4B-Instruct-2507-bf16",
    "lfm2.5-1.2b": "lmstudio-community/LFM2.5-1.2B-Instruct-MLX-4bit",
    "lfm2.5-1.2b-bf16": "LiquidAI/LFM2.5-1.2B-Instruct-MLX-bf16",
    "llama3.2-3b": "mlx-community/Llama-3.2-3B-Instruct-bf16",
    "phi4-mini": "mlx-community/Phi-4-mini-instruct-4bit",
    "gemma3n-e4b": "mlx-community/gemma-3n-E4B-it-lm-4bit",
    "gemma3-4b-qat": "mlx-community/gemma-3-4b-it-qat-4bit",
    "llama3.2-1b": "mlx-community/Llama-3.2-1B-Instruct-bf16",
    "llama3.2-1b-8bit": "mlx-community/Llama-3.2-1B-Instruct-8bit",
    "llama3.2-1b-4bit": "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "lfm2-2.6b-exp": "mlx-community/LFM2-2.6B-Exp-bf16",
    "gemma3-4b-bf16": "mlx-community/gemma-3-4b-it-bf16",
    "gemma3-1b-bf16": "mlx-community/gemma-3-1b-it-bf16",
    "qwen3-1.7b-bf16": "Qwen/Qwen3-1.7B-MLX-bf16",
    "qwen3-1.7b-4bit": "Qwen/Qwen3-1.7B-MLX-4bit",
}

# ── Prompts ─────────────────────────────────────────────────
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

TASK_PROMPTS = {
    "spell-replace": (
        "The transcript contains a misspelled word followed by a spelled-out correction. "
        "Replace the misspelled word with the letters spelled out. Remove the spelling instruction. "
        "Output ONLY the cleaned text."
    ),
    "self-correction": (
        "The speaker corrected themselves mid-sentence. "
        "Apply the correction: keep the corrected version, remove the original and the correction phrase. "
        "Output ONLY the cleaned text."
    ),
    "quote-unquote": (
        "Wrap the quoted content in actual quotation marks. "
        "Remove 'quote-unquote' or 'quote...end quote' markers. "
        "Output ONLY the cleaned text."
    ),
    "quote-endquote": (
        "Wrap content between 'quote' and 'end quote' in quotation marks. "
        "Remove the markers. Output ONLY the cleaned text."
    ),
    "at-symbol": (
        "Insert @ symbols where directed. Remove the instruction phrase. "
        "Output ONLY the cleaned text."
    ),
    "caps": (
        "Convert the text to ALL CAPS as requested. Remove the formatting instruction. "
        "Output ONLY the cleaned text."
    ),
    "emphasis": (
        "Apply bold formatting (**word**) to the emphasized word. "
        "Remove the instruction. Output ONLY the cleaned text."
    ),
    "emoji": (
        "Convert the emoji description to the actual emoji character(s). "
        "Output ONLY the emoji, nothing else."
    ),
    "multi-step": (
        "Execute ALL verbal commands in the transcript in order: "
        "spelling corrections, self-corrections, @-symbol insertions, "
        "quoting, formatting. Output ONLY the final cleaned text."
    ),
    "camelcase": (
        "Apply correct camelCase to code identifiers in the transcript. "
        "Output ONLY the cleaned text."
    ),
}


def load_test_set():
    with open(BENCH_DIR / "test_set.json") as f:
        return json.load(f)


def build_prompt(tokenizer, input_text, model_path, category=None, prompt_mode="generic"):
    """Build chat prompt for a model."""
    if prompt_mode == "spoke" and category:
        from prompts import compose_system_prompt
        system = compose_system_prompt(category)
    elif prompt_mode == "task" and category in TASK_PROMPTS:
        system = TASK_PROMPTS[category]
    elif prompt_mode == "v2":
        system = V2_PROMPT
    else:
        system = GENERIC_PROMPT

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": input_text},
    ]

    # Qwen3: disable thinking mode for speed
    kwargs = {}
    if "qwen3" in model_path.lower():
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            pass  # tokenizer doesn't support enable_thinking

    return tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )


def clean_output(text, model_path):
    """Strip model-specific artifacts from output."""
    text = text.strip()
    # Strip Qwen3 thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Strip common wrappers
    text = re.sub(r"^```\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    # Strip leading/trailing quotes if the whole thing is wrapped
    if text.startswith('"') and text.endswith('"') and text.count('"') == 2:
        text = text[1:-1]
    return text.strip()


def score_output(output, ideal):
    """Score model output against the ideal.

    Returns: exact | semantic | partial | fail
    """
    out = " ".join(output.strip().split())
    ref = " ".join(ideal.strip().split())

    if out == ref:
        return "exact"

    # Case-insensitive match (ok for non-caps examples)
    if out.lower() == ref.lower():
        return "semantic"

    # Check if meta-command words were removed (basic sanity check)
    meta_markers = [
        "spell that", "spell it", "spell this",
        "scratch that", "wait no, sorry",
        "at symbol", "at sign",
        "quote-unquote", "end quote",
        "use all caps", "in caps",
        "emphasize", "emoji",
    ]
    has_meta = any(m in out.lower() for m in meta_markers)

    if not has_meta and len(out) > 0:
        return "partial"

    return "fail"


def benchmark_model(model_path, test_set, prompt_mode="generic", verbose=True,
                    adapter_path=None, kv_bits=None, use_prompt_cache=False):
    """Run benchmark on a single model."""
    import mlx_lm
    from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache, load_prompt_cache
    from mlx_lm.generate import generate_step

    short_name = next((k for k, v in MODELS.items() if v == model_path), model_path.split("/")[-1])
    if adapter_path:
        adapter_label = Path(adapter_path).name
        short_name = f"{short_name}+lora"

    opts = []
    if kv_bits:
        opts.append(f"kv_bits={kv_bits}")
    if use_prompt_cache:
        opts.append("prompt_cache")
    opts_str = f"  opts: {', '.join(opts)}" if opts else ""

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {short_name}  ({model_path})")
        if adapter_path:
            print(f"  adapters: {adapter_path}")
        print(f"  prompt: {prompt_mode}")
        if opts_str:
            print(opts_str)
        print(f"{'='*60}")

    # Load
    t0 = time.time()
    try:
        load_kwargs = {}
        if adapter_path:
            load_kwargs["adapter_path"] = adapter_path
        model, tokenizer = mlx_lm.load(model_path, **load_kwargs)
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return None
    load_time = time.time() - t0
    if verbose:
        print(f"  Loaded in {load_time:.1f}s")

    # Warmup (compile kernels)
    warmup = build_prompt(tokenizer, "test", model_path, prompt_mode=prompt_mode)
    mlx_lm.generate(model, tokenizer, prompt=warmup, max_tokens=8, sampler=GREEDY)

    # Build system prompt cache if enabled
    cache_file = None
    system_prefix_len = 0
    if use_prompt_cache:
        # Get the system prompt for this mode
        if prompt_mode == "v2":
            sys_prompt = V2_PROMPT
        else:
            sys_prompt = GENERIC_PROMPT

        # Tokenize just the system message portion of the chat template
        sys_messages = [{"role": "system", "content": sys_prompt}]
        kwargs = {}
        if "qwen3" in model_path.lower():
            try:
                sys_text = tokenizer.apply_chat_template(
                    sys_messages, tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            except TypeError:
                sys_text = tokenizer.apply_chat_template(
                    sys_messages, tokenize=False,
                    add_generation_prompt=False,
                )
        else:
            sys_text = tokenizer.apply_chat_template(
                sys_messages, tokenize=False,
                add_generation_prompt=False,
            )
        sys_tokens = tokenizer.encode(sys_text)
        system_prefix_len = len(sys_tokens)

        # Fill cache with system prompt
        base_cache = make_prompt_cache(model)
        for _ in generate_step(
            mx.array(sys_tokens), model,
            max_tokens=0, prompt_cache=base_cache,
        ):
            pass
        mx.eval([c.state for c in base_cache])

        # Save to temp file for reloading per-example
        cache_file = tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False).name
        save_prompt_cache(cache_file, base_cache)
        del base_cache

        if verbose:
            print(f"  Cached {system_prefix_len} system prompt tokens -> {cache_file}")

    # Extra kwargs for generate
    gen_kwargs = {}
    if kv_bits:
        gen_kwargs["kv_bits"] = kv_bits
        gen_kwargs["kv_group_size"] = 64

    results = []
    for ex in test_set:
        prompt = build_prompt(
            tokenizer, ex["input"], model_path,
            category=ex["category"], prompt_mode=prompt_mode,
        )

        t_start = time.time()
        if use_prompt_cache and cache_file:
            # Load fresh cache copy, pass only the suffix tokens
            ex_cache = load_prompt_cache(cache_file)
            full_tokens = tokenizer.encode(prompt)
            suffix_tokens = full_tokens[system_prefix_len:]
            raw_output = mlx_lm.generate(
                model, tokenizer, prompt=suffix_tokens,
                max_tokens=256, sampler=GREEDY,
                prompt_cache=ex_cache, **gen_kwargs,
            )
        else:
            raw_output = mlx_lm.generate(
                model, tokenizer, prompt=prompt,
                max_tokens=256, sampler=GREEDY,
                **gen_kwargs,
            )
        gen_time = time.time() - t_start

        output = clean_output(raw_output, model_path)
        score = score_output(output, ex["ideal"])

        results.append({
            "id": ex["id"],
            "category": ex["category"],
            "input": ex["input"],
            "ideal": ex["ideal"],
            "output": output,
            "raw_output": raw_output.strip()[:500],
            "score": score,
            "gen_time_s": round(gen_time, 3),
        })

        if verbose:
            icon = {"exact": "✓", "semantic": "~", "partial": "△", "fail": "✗"}[score]
            print(f"  [{icon}] {ex['category']:<16} {gen_time:.2f}s")
            if score != "exact":
                print(f"       expected: {ex['ideal'][:80]}")
                print(f"       got:      {output[:80]}")

    # Aggregate
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
        "short_name": short_name,
        "prompt_mode": prompt_mode,
        "load_time_s": round(load_time, 1),
        "n": n,
        "exact": exact,
        "semantic": semantic,
        "partial": partial,
        "fail": fail,
        "accuracy": round(accuracy, 4),
        "avg_latency_s": round(avg_latency, 3),
        "results": results,
    }

    if verbose:
        print(f"\n  Accuracy: {accuracy:.0%}  (exact={exact} semantic={semantic} partial={partial} fail={fail})")
        print(f"  Avg latency: {avg_latency:.2f}s")

    # Cleanup
    if cache_file:
        import os
        os.unlink(cache_file)
    del model, tokenizer
    mx.clear_cache()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Zero-shot LLM benchmark")
    parser.add_argument("--model", type=str, help="Short name or HF path")
    parser.add_argument("--all", action="store_true", help="Run all models")
    parser.add_argument("--adapter-path", type=str, default=None,
                        help="Path to LoRA adapter weights (loads on top of base model)")
    parser.add_argument("--prompt-mode", choices=["generic", "task", "spoke", "v2"], default="generic",
                        help="Prompt strategy: generic | task | spoke | v2 (condensed training prompt)")
    parser.add_argument("--test-set", type=str, default=None,
                        help="Path to test set JSON (default: test_set.json in bench dir)")
    parser.add_argument("--kv-bits", type=int, default=None,
                        help="Quantize KV cache to N bits (e.g. 4, 8)")
    parser.add_argument("--prompt-cache", action="store_true",
                        help="Cache system prompt KV across examples")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    if args.test_set:
        with open(args.test_set) as f:
            test_set = json.load(f)
    else:
        test_set = load_test_set()
    print(f"Loaded {len(test_set)} test examples")

    if args.all:
        paths = list(MODELS.values())
    elif args.model:
        paths = [MODELS.get(args.model, args.model)]
    else:
        parser.error("Specify --model <name> or --all")

    all_summaries = []
    for path in paths:
        summary = benchmark_model(path, test_set, prompt_mode=args.prompt_mode,
                                   verbose=args.verbose, adapter_path=args.adapter_path,
                                   kv_bits=args.kv_bits, use_prompt_cache=args.prompt_cache)
        if summary is None:
            continue
        all_summaries.append(summary)

        # Save per-model result
        out_name = f"result_{summary['short_name']}_{args.prompt_mode}.json"
        out_path = BENCH_DIR / out_name
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  -> {out_path}")

    # Comparison table
    if len(all_summaries) > 1:
        print(f"\n{'='*60}")
        print("  COMPARISON  (sorted by accuracy)")
        print(f"{'='*60}")
        header = f"  {'Model':<18} {'Acc':>6} {'Exact':>5} {'Sem':>4} {'Fail':>4} {'Latency':>8}"
        print(header)
        print(f"  {'-'*50}")
        for s in sorted(all_summaries, key=lambda x: (-x["accuracy"], x["avg_latency_s"])):
            print(
                f"  {s['short_name']:<18} {s['accuracy']:>5.0%} "
                f"{s['exact']:>5} {s['semantic']:>4} {s['fail']:>4} "
                f"{s['avg_latency_s']:>7.2f}s"
            )


if __name__ == "__main__":
    main()
