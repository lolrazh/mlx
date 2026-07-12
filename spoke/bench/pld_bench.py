#!/usr/bin/env python3
"""
Prompt-Lookup Decoding (PLD) benchmark for the Spoke text-cleaner.

PLD (a.k.a. n-gram / prompt-lookup speculative decoding) exploits the fact that
a text *cleaner*'s output echoes large verbatim spans of its input. At each
decode step we search the context (prompt + generated-so-far) for the most
recent occurrence of the trailing n-gram, splice the following k tokens in as a
DRAFT, then verify the whole block in ONE batched forward pass (accept the
longest matching argmax prefix + 1 bonus token). It is LOSSLESS: output is
token-identical to plain greedy. No draft model, no extra memory.

Reference: apoorvumang/prompt-lookup-decoding, mlx-lm PR #1297.

Usage:
    python spoke/bench/pld_bench.py --model spoke/models/g4e4b-champion-mlx-dwq4-g64
"""

import json
import time
import argparse
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache

import sys
BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR))
from run_benchmark import V2_PROMPT, build_prompt, clean_output, score_output, _THINK_MARKERS


# ── no-think logit bias (same logic as run_benchmark.make_no_think_processors) ──
def no_think_bias_ids(tokenizer):
    """Return list of token ids to ban (thinking-channel openers)."""
    ids_to_ban = []
    for marker in _THINK_MARKERS:
        try:
            ids = tokenizer.encode(marker, add_special_tokens=False)
        except Exception:
            continue
        if len(ids) == 1:
            ids_to_ban.append(ids[0])
    return ids_to_ban


def apply_bias(logits, ban_ids):
    """logits: (positions, vocab). Ban given token columns in-place-ish."""
    if ban_ids:
        logits[:, ban_ids] = -1e9
    return logits


# ── prompt lookup ──────────────────────────────────────────────────────────
def find_draft(context, n, k):
    """Most-recent prior occurrence of trailing n-gram; return following <=k tokens."""
    L = len(context)
    if L < n + 1:
        return []
    ngram = context[L - n:]
    # scan backward from the position just before the trailing n-gram
    for i in range(L - n - 1, -1, -1):
        if context[i:i + n] == ngram:
            draft = context[i + n:i + n + k]
            return draft
    return []


# ── generation ─────────────────────────────────────────────────────────────
def generate_tokens(model, tokenizer, prompt_ids, max_tokens=256,
                    n=2, k=8, ban_ids=None, prefill_step=512, collect_stats=False):
    """
    Greedy (argmax) generation with optional PLD.
      n, k: n-gram match length and max draft tokens. k=0 => plain greedy.
    Returns (generated_token_ids, stats).
    """
    eos = set(tokenizer.eos_token_ids) if tokenizer.eos_token_ids else {tokenizer.eos_token_id}
    ban_ids = ban_ids or []

    cache = make_prompt_cache(model)
    context = list(prompt_ids)

    # Prefill prompt (chunked). Last chunk gives us logits for first token.
    p = mx.array(prompt_ids)[None]
    Lp = p.shape[1]
    off = 0
    logits = None
    while off < Lp:
        chunk = p[:, off:off + prefill_step]
        logits = model(chunk, cache=cache)
        off += prefill_step
        mx.eval([c.state for c in cache])
    # logits: (1, chunk_len, vocab); last position predicts first generated token
    last = logits[:, -1, :]
    last = apply_bias(last, ban_ids)
    cur = int(mx.argmax(last, axis=-1).item())  # first generated token, NOT yet fed

    generated = []
    steps = 0
    accepted_total = 0  # extra tokens gained beyond the mandatory 1/step

    while len(generated) < max_tokens:
        if cur in eos:
            generated.append(cur)
            break
        # propose draft from context (context ends with the last CONFIRMED token;
        # cur is the not-yet-fed next token, so include it for the lookup)
        lookup_ctx = context + [cur]
        draft = find_draft(lookup_ctx, n, k) if k > 0 else []
        # cap draft so we don't exceed max_tokens
        room = max_tokens - len(generated)
        draft = draft[:max(0, room - 1)]

        block = [cur] + draft
        inp = mx.array(block)[None]
        out = model(inp, cache=cache)          # (1, len(block), vocab)
        preds = out[0]                          # (len(block), vocab)
        preds = apply_bias(preds, ban_ids)
        argmax = mx.argmax(preds, axis=-1)
        mx.eval(argmax)
        argmax = [int(x) for x in argmax.tolist()]

        # position 0 predicts token after cur (== true next). Then verify drafts.
        # accept longest prefix where argmax[i] == draft[i]
        n_accept = 0
        for i in range(len(draft)):
            if argmax[i] == draft[i]:
                n_accept += 1
            else:
                break

        # confirmed appended this step: cur is confirmed already-as-cur,
        # we now append: draft[0..n_accept-1] and the bonus token argmax[n_accept]
        # First, register cur as generated.
        generated.append(cur)
        context.append(cur)
        stop = cur in eos

        # append accepted drafts
        for i in range(n_accept):
            if len(generated) >= max_tokens:
                break
            t = draft[i]
            generated.append(t)
            context.append(t)
            if t in eos:
                stop = True
                break
        steps += 1
        accepted_total += n_accept

        if stop or len(generated) >= max_tokens:
            # trim rejected draft positions from cache
            n_fed = len(block)              # positions added to cache this step
            n_kept = 1 + n_accept           # cur + accepted drafts
            if n_fed - n_kept > 0:
                trim_prompt_cache(cache, n_fed - n_kept)
            break

        # bonus token = argmax at position n_accept (token after last accepted/cur)
        bonus = argmax[n_accept]
        # trim rejected drafts from cache (positions after the last accepted draft)
        n_fed = len(block)
        n_kept = 1 + n_accept
        if n_fed - n_kept > 0:
            trim_prompt_cache(cache, n_fed - n_kept)
        cur = bonus  # bonus not yet fed; becomes next cur

    stats = {
        "steps": steps,
        "gen_tokens": len(generated),
        "accepted_per_step": (accepted_total / steps) if steps else 0.0,
        # mean tokens confirmed per forward pass = (gen_tokens)/steps roughly
        "tokens_per_step": (len(generated) / steps) if steps else 0.0,
    }
    return generated, stats


def run(model_path, test_set, n=2, k=8, use_pld=True, warm=True):
    model, tokenizer = mlx_lm.load(model_path)
    ban_ids = no_think_bias_ids(tokenizer)

    # warm up (compile kernels) - discard
    warm_prompt = build_prompt(tokenizer, "test", model_path, prompt_mode="v2")
    wp = tokenizer.encode(warm_prompt)
    generate_tokens(model, tokenizer, wp, max_tokens=8, n=n, k=(k if use_pld else 0), ban_ids=ban_ids)
    mx.eval(mx.array([0]))

    rows = []
    for ex in test_set:
        prompt = build_prompt(tokenizer, ex["input"], model_path, prompt_mode="v2")
        pids = tokenizer.encode(prompt)
        t0 = time.perf_counter()
        toks, stats = generate_tokens(
            model, tokenizer, pids, max_tokens=256,
            n=n, k=(k if use_pld else 0), ban_ids=ban_ids,
        )
        dt = time.perf_counter() - t0
        text = tokenizer.decode(toks)
        rows.append({
            "id": ex["id"], "category": ex["category"],
            "tokens": toks, "text": text,
            "ms": dt * 1000.0, "stats": stats,
            "ideal": ex["ideal"],
        })
    del model, tokenizer
    mx.clear_cache()
    return rows, ban_ids


def summarize(rows, label):
    ms = sorted(r["ms"] for r in rows)
    n = len(ms)
    mean = sum(ms) / n
    med = ms[n // 2] if n % 2 else (ms[n // 2 - 1] + ms[n // 2]) / 2
    acc = sum(r["stats"]["accepted_per_step"] for r in rows) / n
    tps = sum(r["stats"]["tokens_per_step"] for r in rows) / n
    gt = sum(r["stats"]["gen_tokens"] for r in rows) / n
    return {
        "label": label, "mean_ms": round(mean, 1), "median_ms": round(med, 1),
        "accepted_per_step": round(acc, 2), "tokens_per_step": round(tps, 2),
        "mean_gen_tokens": round(gt, 1),
        "p90_ms": round(ms[int(0.9 * n)], 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="spoke/models/g4e4b-champion-mlx-dwq4-g64")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    test_set = json.load(open(BENCH_DIR / "test_set_evals.json"))
    print(f"Loaded {len(test_set)} examples")

    # 1) plain greedy (reference)
    print("\n== plain greedy ==")
    greedy_rows, _ = run(args.model, test_set, use_pld=False)
    gsum = summarize(greedy_rows, "greedy")
    print(gsum)

    # 2) PLD
    print(f"\n== PLD n={args.n} k={args.k} ==")
    pld_rows, _ = run(args.model, test_set, n=args.n, k=args.k, use_pld=True)
    psum = summarize(pld_rows, f"pld_n{args.n}_k{args.k}")
    print(psum)

    # 3) correctness: token-identical?
    mism = []
    for g, p in zip(greedy_rows, pld_rows):
        if g["tokens"] != p["tokens"]:
            mism.append(g["id"])
    identical = len(mism) == 0
    print(f"\nToken-identical to greedy: {identical}  (mismatches: {mism})")

    result = {
        "model": args.model,
        "n_examples": len(test_set),
        "identical": identical,
        "mismatches": mism,
        "greedy": gsum,
        "pld": psum,
        "speedup_mean": round(gsum["mean_ms"] / psum["mean_ms"], 2),
        "speedup_median": round(gsum["median_ms"] / psum["median_ms"], 2),
        "n": args.n, "k": args.k,
    }
    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))

    if args.out:
        # attach per-example detail
        result["greedy_rows"] = [{k: v for k, v in r.items() if k != "tokens"} for r in greedy_rows]
        result["pld_rows"] = [{k: v for k, v in r.items() if k != "tokens"} for r in pld_rows]
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
