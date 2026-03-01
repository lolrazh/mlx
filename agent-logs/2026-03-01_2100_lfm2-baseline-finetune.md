# LFM2 Zero-Shot Baselines & Fine-Tune Launch

**Date:** 2026-03-01
**Agent:** Claude Opus 4.6
**Status:** 🔄 Ongoing (LFM2-2.6B LoRA training in progress)
**Building on:** `2026-03-01_1800_t11-t12-training-data-review.md`

## User Intention
After exhausting Qwen3-4B hyperparameter tuning (T11=83% ceiling), the user wanted to explore alternative model architectures to see if a different model could handle the Spoke ASR post-processing task better — or at least learn it via fine-tuning. Specifically interested in LFM (Liquid Foundation Models) for their efficiency, and also considered Qwen3.5-35B-A3B (MoE). The user's philosophy: "you can't predict how a model learns" — zero-shot scores don't determine fine-tuning potential.

## What We Accomplished
- ✅ **LFM2.5-1.2B zero-shot baseline** — 9% accuracy (v2 prompt, v3 test set). Model doesn't understand meta-linguistic commands at all.
- ✅ **Qwen3.5-35B-A3B feasibility check** — 35B params at 4-bit = ~20 GB. Too tight for M4 24 GB (~22-23 GB total with overhead). Ruled out.
- ✅ **LFM2-2.6B-Exp bf16 zero-shot baseline** — 9% accuracy (v2 prompt). Same fundamental failure as 1.2B despite 2x params and bf16 precision.
- ✅ **LFM2-2.6B-Exp few-shot baseline** — 30% accuracy (spoke prompt with examples). Huge jump — model learns output format from examples but can't execute commands reliably.
- ✅ **LFM2 architecture analysis** — 30 layers: 8 attention [2,5,9,13,17,21,24,27] + 22 ShortConv. ChatML template (same as Qwen3). mlx_lm LoRA auto-discovers all linear layers across both layer types.
- 🔄 **LFM2-2.6B-Exp LoRA fine-tuning launched** — 800 iters, all 30 layers, 12.2M trainable params (0.476%). Training in progress.

## Technical Implementation

### Zero-Shot Baselines
All tested with v2 system prompt (~80 tokens), no few-shot examples, greedy decoding, on v3 test set (23 examples).

| Model | Precision | Prompt | Accuracy | Avg Latency |
|-------|-----------|--------|----------|-------------|
| Qwen3-4B + LoRA (T11) | bf16 | v2 (trained) | **83%** | 2.8s |
| Qwen3-4B zero-shot | bf16 | v2 | 35% | 2.8s |
| LFM2-2.6B-Exp | bf16 | spoke few-shot | 30% | 1.36s |
| LFM2-2.6B-Exp | bf16 | v2 zero-shot | 9% | 0.87s |
| LFM2.5-1.2B | 4-bit | v2 zero-shot | 9% | 0.24s |

### LFM2 Architecture (Key for LoRA)
- **Model type**: `lfm2` (hybrid conv + attention)
- **30 layers**: 8 full GQA attention + 22 ShortConv (depthwise conv1d with gating)
- **Attention layers**: `q_proj`, `k_proj`, `v_proj`, `out_proj` + QK layernorm + RoPE
- **Conv layers**: `in_proj` (3x hidden), `out_proj`, depthwise `conv` (kernel=4)
- **All layers**: SwiGLU MLP (`w1`, `w2`, `w3`)
- **LoRA targets all linear layers**: 32 attn + 44 conv + 90 MLP = ~166 LoRA adapters

### Fine-Tune Config (LFM2-T1)
```yaml
model: mlx-community/LFM2-2.6B-Exp-bf16
data: ./spoke/data/v3        # 535 train, 20 valid
fine_tune_type: lora
optimizer: adam
lora_parameters: { rank: 8, scale: 2.0, dropout: 0.05 }
batch_size: 4
iters: 800                   # ~6 passes/example
learning_rate: 1e-5
num_layers: 30               # all layers
adapter_path: ./spoke/adapters-lfm2-t1
```
Trainable: 12.2M / 2569.3M (0.476%)

**Files Modified:**
- `spoke/config.yaml` — switched from Qwen3-4B to LFM2-2.6B-Exp, 800 iters, num_layers=30
- `spoke/bench/run_benchmark.py` — added `lfm2-2.6b-exp` and `lfm2.5-1.2b` to MODELS dict
- `spoke/bench/result_lfm2.5-1.2b_v2.json` — LFM2.5 zero-shot results
- `spoke/bench/result_lfm2-2.6b-exp_v2.json` — LFM2 zero-shot v2 results
- `spoke/bench/result_lfm2-2.6b-exp_spoke.json` — LFM2 few-shot spoke results

## Bugs & Issues Encountered
1. **Qwen3.5-35B-A3B mlx-community version is VLM** — The `mlx-community/Qwen3.5-35B-A3B-4bit` (20.4 GB) was converted with mlx-vlm, not mlx-lm. Can't use with our text-only benchmark pipeline.
   - **Fix:** Found `NexVeridian/Qwen3.5-35B-A3B-4bit` converted with mlx-lm 0.30.8, but memory still too tight at ~20 GB.
2. **No bf16 MLX version of LFM2-2.6B exists** — Only `LFM2-2.6B-Exp-bf16` (the Exp RL variant). No standard LFM2-2.6B in MLX bf16.
   - **Resolution:** User chose Exp variant since its IFBench score surpasses DeepSeek R1 at instruction following.
3. **LFM2.5-1.2B tested at 4-bit instead of official 8-bit** — Used `lmstudio-community` 4-bit instead of `LiquidAI/LFM2.5-1.2B-Instruct-MLX-8bit`.
   - **Impact:** Minimal — LFM2-2.6B at full bf16 also scored 9%, confirming precision wasn't the bottleneck.

## Key Learnings
- **LFM hybrid architecture fundamentally can't parse meta-linguistic commands zero-shot**: Both 1.2B and 2.6B scored 9%. The conv-dominant architecture (22/30 layers) doesn't provide instruction-following depth for "execute verbal commands" tasks. The model extracts keywords or paraphrases instead of cleaning.
- **Few-shot examples teach output FORMAT but not command EXECUTION**: LFM2 jumped 9% → 30% with spoke few-shot. It learned "output should be a cleaned sentence" but still couldn't execute spelling, quoting, or emoji commands. Format ≠ reasoning.
- **IFBench ≠ meta-linguistic instruction following**: IFBench tests structural instructions ("respond in JSON", "use 3 paragraphs"). Our task requires understanding "spell that K-A-D-A-I" as a command to execute, which is fundamentally different.
- **mlx_lm.lora auto-discovers all Linear layers via `get_keys_for_lora()`**: No need to manually specify LoRA keys for hybrid architectures. It scans all modules and targets any `nn.Linear`, `nn.QuantizedLinear`, or `nn.Embedding`.
- **LFM2 uses ChatML template (same as Qwen3)**: Training data format is compatible — no conversion needed.
- **MoE models: full memory cost for partial compute**: Qwen3.5-35B-A3B has 3B active params but needs all 35B in memory (~20 GB at 4-bit). Not viable on 24 GB consumer hardware.
- **LiquidAI/LFM2-2.6B-Transcript is for meeting summarization**, not ASR post-processing. Despite the name, it doesn't help with our task.

## Architecture Decisions
- **Test LFM2 despite 9% zero-shot**: User's insight that zero-shot doesn't predict fine-tuning is valid. LoRA could teach the task even to a model that can't do it zero-shot. Worth the experiment.
- **800 iters for LFM2**: User requested. ~6 passes/example on 535 training examples. More than Qwen3's ~2.4 passes, but smaller model may need more exposure.
- **All 30 layers for LoRA**: With only 8/30 attention layers, using num_layers=16 would miss most of the model. All 30 ensures LoRA touches every available linear projection.
- **Skipped Qwen3.5-35B-A3B**: 20 GB at 4-bit doesn't leave enough headroom on 24 GB for training or comfortable inference. Would need 48+ GB machine.

## Ready for Next Session
- 🔧 **LFM2-T1 training in progress** — 800 iters on v3 data. When done: benchmark with v2 prompt on v3 test set, compare to T11 (83%).
- ✅ **Benchmark infra supports LFM2** — both `lfm2-2.6b-exp` and `lfm2.5-1.2b` in MODELS dict
- ✅ **All zero-shot baselines recorded** — result JSON files saved in `spoke/bench/`
- 🔧 **Still queued**: Llama 3.2 3B baseline, Q1 mixed-bit quant on T11, encoder-decoder (CoEdIT) testing

## Context for Future
LFM2-2.6B-Exp is the first non-Qwen model we're fine-tuning for Spoke. If it learns the task well despite 9% zero-shot, it validates the user's hypothesis that zero-shot ≠ fine-tune potential, and opens the door to smaller/faster models. If it fails, it confirms that the hybrid conv+attention architecture lacks the representational capacity for meta-linguistic command execution, and we should focus on transformer-only models (Llama 3.2 3B) or encoder-decoder (CoEdIT). Either outcome is informative.
