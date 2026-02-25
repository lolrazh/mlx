"""Step 1: Crack open a real model. See what's inside."""

import json
from pathlib import Path
import mlx.core as mx

# Find the downloaded model
cache_dir = Path.home() / ".cache/huggingface/hub/models--openai-community--gpt2"
snapshot = list((cache_dir / "snapshots").iterdir())[0]

# === 1. config.json — the blueprint ===

with open(snapshot / "config.json") as f:
    config = json.load(f)

print("=== config.json (the blueprint) ===\n")
important_keys = ["model_type", "n_layer", "n_head", "n_embd", "vocab_size", "n_positions"]
for key in important_keys:
    print(f"  {key}: {config.get(key)}")

print(f"""
This tells you everything about the architecture:
  - {config['n_layer']} transformer blocks (we used 4 in our toy model)
  - {config['n_head']} attention heads (we used 4)
  - {config['n_embd']} embedding dim (we used 64)
  - {config['vocab_size']} token vocabulary (we used 65 characters)
  - {config['n_positions']} max context length (we used 128)
""")

# === 2. model.safetensors — the actual weights ===

weights = mx.load(str(snapshot / "model.safetensors"))

print("=== model.safetensors (the weights) ===\n")
print(f"Total weight tensors: {len(weights)}")

total_params = sum(w.size for w in weights.values())
print(f"Total parameters: {total_params:,}")
print(f"Memory at float32: {total_params * 4 / 1e9:.2f} GB")
print(f"Memory at float16: {total_params * 2 / 1e9:.2f} GB")

# Show the first few weights
print(f"\nFirst 15 weight names and shapes:")
for i, (name, tensor) in enumerate(sorted(weights.items())):
    if i >= 15:
        print(f"  ... and {len(weights) - 15} more")
        break
    print(f"  {name:45s} {str(tensor.shape):20s} {tensor.dtype}")

# === 3. Compare to our model ===

print(f"""
=== Our model vs GPT-2 ===

                    Ours        GPT-2
  Layers:           4           {config['n_layer']}
  Heads:            4           {config['n_head']}
  Embedding dim:    64          {config['n_embd']}
  Vocab size:       65          {config['vocab_size']}
  Context window:   128         {config['n_positions']}
  Parameters:       216K        {total_params/1e6:.0f}M

  Same architecture. Different scale. That's it.
""")
