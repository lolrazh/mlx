#!/usr/bin/env python3
"""
Persistent prompt-cache reuse + PLD, measured on broad58.

Caches the fixed v2 system prompt ONCE in-process, then for each request feeds
only the user-suffix tokens and decodes (optionally with PLD). After each
request the cache is trimmed back to the system prefix for reuse. Compares:
  greedy (no cache)  |  greedy + sys-cache  |  PLD + sys-cache
"""
import json, sys, time
from pathlib import Path
import mlx.core as mx, mlx_lm
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR))
from pld_bench import no_think_bias_ids, apply_bias, find_draft
from run_benchmark import build_prompt, V2_PROMPT


def decode_with_cache(model, tokenizer, cache, first_from_logits, suffix_ids,
                      base_len, n, k, ban_ids, max_tokens=256):
    """
    cache already holds `base_len` tokens (system prefix). Feed suffix_ids,
    then greedy/PLD decode. Cache is trimmed back to base_len before return.
    Returns (generated_tokens, stats).
    """
    eos = set(tokenizer.eos_token_ids) if tokenizer.eos_token_ids else {tokenizer.eos_token_id}
    context = list(suffix_ids)  # local context for lookup (suffix only is fine; sys prompt rarely echoed)
    # feed suffix, get first token
    out = model(mx.array(suffix_ids)[None], cache=cache)
    last = apply_bias(out[:, -1, :], ban_ids)
    cur = int(mx.argmax(last, axis=-1).item())
    fed = len(suffix_ids)

    generated = []
    steps = 0; accepted = 0
    while len(generated) < max_tokens:
        if cur in eos:
            generated.append(cur); break
        lookup = context + [cur]
        draft = find_draft(lookup, n, k) if k > 0 else []
        draft = draft[:max(0, (max_tokens - len(generated)) - 1)]
        block = [cur] + draft
        o = model(mx.array(block)[None], cache=cache)
        fed += len(block)
        preds = apply_bias(o[0], ban_ids)
        am = mx.argmax(preds, axis=-1); mx.eval(am)
        am = [int(x) for x in am.tolist()]
        na = 0
        for i in range(len(draft)):
            if am[i] == draft[i]: na += 1
            else: break
        generated.append(cur); context.append(cur)
        stop = cur in eos
        for i in range(na):
            if len(generated) >= max_tokens: break
            generated.append(draft[i]); context.append(draft[i])
            if draft[i] in eos: stop = True; break
        steps += 1; accepted += na
        n_kept = 1 + na
        # trim rejected drafts (positions in cache beyond kept)
        if len(block) - n_kept > 0:
            trim_prompt_cache(cache, len(block) - n_kept)
        fed -= (len(block) - n_kept)
        if stop or len(generated) >= max_tokens:
            break
        cur = am[na]
    # trim cache back to base_len
    total_extra = fed  # tokens fed beyond base
    if total_extra > 0:
        trim_prompt_cache(cache, total_extra)
    stats = {"steps": steps, "gen_tokens": len(generated),
             "accepted_per_step": accepted / steps if steps else 0,
             "tokens_per_step": len(generated) / steps if steps else 0}
    return generated, stats


def summ(ms, extra):
    ms = sorted(ms); n = len(ms)
    med = ms[n // 2] if n % 2 else (ms[n//2-1]+ms[n//2])/2
    d = {"mean_ms": round(sum(ms)/n,1), "median_ms": round(med,1), "p90_ms": round(ms[int(0.9*n)],1)}
    d.update(extra); return d


def main():
    mp = "spoke/models/g4e4b-champion-mlx-dwq4-g64"
    if len(sys.argv) > 1: mp = sys.argv[1]
    ts = json.load(open(BENCH_DIR / "test_set_evals.json"))
    model, tokenizer = mlx_lm.load(mp)
    ban = no_think_bias_ids(tokenizer)

    # system prefix tokens (must be exact prefix of each full prompt)
    sysmsg = [{"role": "system", "content": V2_PROMPT}]
    systxt = tokenizer.apply_chat_template(sysmsg, tokenize=False, add_generation_prompt=False)
    sys_tokens = tokenizer.encode(systxt)
    base_len = len(sys_tokens)

    # verify prefix assumption
    bad_prefix = 0
    for ex in ts:
        full = tokenizer.encode(build_prompt(tokenizer, ex["input"], mp, prompt_mode="v2"))
        if full[:base_len] != sys_tokens: bad_prefix += 1
    print(f"system prefix len={base_len}; prompts NOT sharing prefix: {bad_prefix}/{len(ts)}")

    # build persistent cache once
    base_cache = make_prompt_cache(model)
    o = model(mx.array(sys_tokens)[None], cache=base_cache); mx.eval(o)

    def run_cached(n, k, label):
        # warm
        full = tokenizer.encode(build_prompt(tokenizer, "test", mp, prompt_mode="v2"))
        decode_with_cache(model, tokenizer, base_cache, None, full[base_len:], base_len, n, k, ban, max_tokens=8)
        ms=[]; rows=[]; accs=[]; tps=[]
        for ex in ts:
            full = tokenizer.encode(build_prompt(tokenizer, ex["input"], mp, prompt_mode="v2"))
            suffix = full[base_len:]
            t0=time.perf_counter()
            toks, st = decode_with_cache(model, tokenizer, base_cache, None, suffix, base_len, n, k, ban)
            ms.append((time.perf_counter()-t0)*1000)
            rows.append((ex["id"], toks)); accs.append(st["accepted_per_step"]); tps.append(st["tokens_per_step"])
        s = summ(ms, {"label": label, "accepted_per_step": round(sum(accs)/len(accs),2),
                      "tokens_per_step": round(sum(tps)/len(tps),2)})
        return s, dict(rows)

    # greedy + cache (reference for identity)
    g_s, g_tok = run_cached(2, 0, "greedy+cache")
    # PLD + cache configs
    results = [g_s]
    tokmaps = {"greedy+cache": g_tok}
    for (n,k) in [(2,1),(3,3),(3,4),(2,4)]:
        s, tm = run_cached(n, k, f"pld_n{n}_k{k}+cache")
        mism = [i for i in g_tok if g_tok[i] != tm[i]]
        s["identical_to_greedy_cache"] = len(mism)==0
        s["mismatches"] = mism
        results.append(s); tokmaps[s["label"]] = tm

    print("\n== persistent sys-cache results (broad58) ==")
    for s in results:
        idn = s.get("identical_to_greedy_cache", "-")
        print(f"  {s['label']:20} mean={s['mean_ms']:6.1f} med={s['median_ms']:6.1f} p90={s['p90_ms']:6.1f} "
              f"acc/step={s['accepted_per_step']:.2f} tps={s['tokens_per_step']:.2f} identical={idn} mism={s.get('mismatches','')}")
    print("\n  <500ms mean?:")
    for s in results:
        print(f"    {s['label']:20} {s['mean_ms']:.1f}ms  {'YES' if s['mean_ms']<500 else 'no'}")

    json.dump({"model": mp, "base_len": base_len, "results": results},
              open(BENCH_DIR / "result_pld_cache_champion.json", "w"), indent=2, ensure_ascii=False)
    print(f"-> {BENCH_DIR/'result_pld_cache_champion.json'}")


if __name__ == "__main__":
    main()
