# Inference Optimization Audit & Qwen3-1.7B Training Launch

**Date:** 2026-03-04
**Agent:** Claude Opus 4.6
**Status:** 🔄 Ongoing (1.7B training running)

## User Intention
User wanted to step back from incremental inference tricks (prompt caching, KV cache quantization — both tested and failed in prior session) and do a comprehensive audit: which model is actually best for latency, what mistakes does each model make, and what are the real optimization paths forward? The goal is to get post-processor latency under 0.5s for a real-time dictation pipeline (Parakeet ASR + LLM post-processor).

## What We Accomplished
- ✅ **Full model latency/accuracy audit** - Compiled all fine-tuned models with dataset version, iters, error patterns
- ✅ **Error pattern analysis** - Documented characteristic failures per model (Gemma echoes, LFM can't meta-command, Llama truncates quotes, Qwen3 only fails emphasis)
- ✅ **Inference optimization research** - Agent researched speculative decoding (buggy in mlx-lm, issue #250), FlashAttention (already built into MLX, closed Jan 2026), ONNX vs MLX (MLX wins 2-3x), pruning (not in mlx-lm), QAT via Unsloth
- ✅ **ELI5 inference pipeline** - Explained prefill vs decode phases, why smaller model helps both, why speculative decoding only helps decode
- ✅ **Architecture research** - Analyzed how Handy app keeps model resident (persistent in-process, not server), confirmed this is the right pattern for Spoke
- ✅ **Qwen3-1.7B zero-shot benchmark** - 13% accuracy, 0.96s latency (v3 test, v2 prompt). Added model shortcuts to benchmark script.
- ✅ **Training config audit** - Found 5 wrong parameters in config.yaml (leftover Muon-YOLO settings). Fixed all to match T2-v4 exactly.
- 🔄 **Qwen3-1.7B-T1 training launched** - 2000 iters, v4 data, T2-v4 parity config. Running in background on M4.

## Technical Implementation

### Inference Pipeline Breakdown
- **Prefill** (~60% of time): Process all input tokens in parallel. Compute-bound. ~0.3-0.5s for 4B.
- **Decode** (~40% of time): Generate tokens one at a time. Memory-bandwidth-bound (120 GB/s on M4). ~15ms/token.
- For our use case (~200 in, ~50 out), prefill dominates. Speculative decoding only helps decode = marginal.

### Key Optimization Findings
| Technique | Status | Verdict |
|-----------|--------|---------|
| Speculative decoding | Buggy (issue #250, adds 50% overhead) | Skip for <4B models |
| FlashAttention/SDPA | Already in MLX (issue #2955 closed) | Already active |
| KV cache quantization | Tested: 3x slower + accuracy drop | Harmful for short seqs |
| Prompt caching (disk) | Tested: disk I/O > prefill savings | Harmful for 88-token prompts |
| Prompt caching (in-memory) | Available via vllm-mlx | Only useful as persistent server |
| Smaller model (1.7B) | Highest impact | **Recommended path** |
| 2-bit DWQ | Untested, high risk | Risky, expect ~70-80% |

### Qwen3-1.7B Architecture
- model_type: qwen3 (same as 4B)
- 28 layers (vs 36), hidden 2048 (vs 2560), 16 attn heads, 8 KV heads
- LoRA: 8.7M trainable params (0.51%), est. ~8 GB peak memory
- Same vocab (151936), same tokenizer, same chat template

### Config Drift Caught
Current config.yaml had Muon-YOLO settings (optimizer=muon, lr=2e-4, 16 layers, 256 seq, 1000 iters). ALL wrong for T2-v4 parity. Fixed to: adam, lr=1e-5, 28 layers, 512 seq, 2000 iters.

**Files Modified:**
- `spoke/bench/run_benchmark.py` - Added qwen3-1.7b-bf16 and qwen3-1.7b-4bit model shortcuts
- `spoke/config.yaml` - Complete rewrite to Qwen3-1.7B-T1 config (T2-v4 parity)
- `spoke/bench/result_qwen3-1.7b-bf16_v2.json` - Zero-shot benchmark result (new file)

## Bugs & Issues Encountered
1. **Config drift from Muon-YOLO experiment** - config.yaml had 5 wrong parameters that would have produced an incomparable training run
   - **Fix:** Rewrote config.yaml with explicit T2-v4 parity comments. User caught this by insisting on double-checking every parameter.
2. **DWQ benchmark file overwritten** - `result_qwen3-t2-v4-dwq4_v2.json` shows 87%/3.1s instead of canonical 96%/0.88s
   - **Note:** File was overwritten by a later run with kv_bits=4 optimization. Canonical numbers are in LEDGER.md.

## Key Learnings
- **Prefill dominates for short-output tasks** - With ~200 input / ~50 output tokens, decode optimization (speculative decoding) saves ~0.1-0.2s max. Only smaller model or shorter prompt meaningfully reduces latency.
- **MLX already has FlashAttention** - Issue #2955 closed Jan 2026 as COMPLETED. `mx.fast.scaled_dot_product_attention` uses fused kernels. No action needed.
- **Speculative decoding is buggy in mlx-lm** - Issue #250: even with num_draft_tokens=0, the speculative code path adds ~50% overhead. Don't use for small models.
- **Handy keeps model resident in-process** - Uses `Arc<Mutex<Option<LoadedEngine>>>` with configurable idle timeout (default: never unload). Not a separate server. This is the right pattern for Spoke.
- **Parakeet MLX is 6x faster than ONNX** - 0.034x RTF (30x real-time) vs ~0.2x RTF (5x). Use MLX, not ONNX.
- **Qwen3-1.7B has no 2507 version** - The 2507 release only covered 235B-A22B, 30B-A3B, and 4B sizes. Use `Qwen/Qwen3-1.7B-MLX-bf16` (original May 2025 release).
- **Zero-shot is meaningless for fine-tune prediction** - 1.7B scored 13% zero-shot. But LFM2 went 9%→83%, Gemma went 9%→87%. Don't rule out models based on zero-shot.

## Architecture Decisions
- **Qwen3-1.7B over LFM2 with v4 data** - LFM2's spell-replace failures are architectural (conv layers can't parse meta-commands), not data-limited. More data won't fix fundamental architecture mismatch.
- **Qwen3-1.7B over 2-bit DWQ on 4B** - Trading model capacity (which we have excess of at 100%) for latency is safer than trading quantization precision (already tight at 96%).
- **Persistent process over local server** - HTTP overhead adds latency for zero benefit in single-user dictation app. Handy-style in-process model is the right pattern.

## Ready for Next Session
- 🔄 **Qwen3-1.7B-T1 training running** - Monitor via wandb. Expect ~1-1.5 hrs. Check adapters at `spoke/adapters-qwen3-1.7b-t1/`.
- ✅ **Benchmark ready** - Run `python spoke/bench/run_benchmark.py --model qwen3-1.7b-bf16 --prompt-mode v2 --test-set spoke/bench/test_set_v3.json --adapter spoke/adapters-qwen3-1.7b-t1` after training.
- ✅ **DWQ pipeline ready** - If accuracy is good: `mlx_lm.fuse` → `mlx_lm.dwq --bits 4 --data-path spoke/data/v4/train.jsonl --grad-checkpoint --batch-size 1`
- 🔧 **Pruning project planned** - User wants to build MLX pruning from first principles as a separate open-source repo. Not started yet.

## Context for Future
This session pivoted from micro-optimizations (prompt caching, KV cache quant — both backfired) to the right macro-optimization: smaller model. Qwen3-1.7B-T1 training is the highest-leverage experiment remaining. If it hits 85%+ and DWQ to ~0.4s, the full pipeline (Parakeet MLX 30ms + Qwen3-1.7B DWQ ~400ms = ~430ms) achieves real-time dictation locally on M4 in ~1.5 GB resident memory. The pruning project is a fun educational side-quest that could further optimize this.
