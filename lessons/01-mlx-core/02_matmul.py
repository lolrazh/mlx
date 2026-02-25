"""02 — Matrix multiplication: the one operation that runs the entire AI industry."""

import mlx.core as mx

# === What a single neural network layer actually does ===
#
# You have input data (say, 2 sentences, each represented as 3 numbers).
# You have weights (learned numbers that transform the input).
# The layer computes: output = input @ weights + bias
#
# That @ is matrix multiplication. That's it.

input_data = mx.array([[1.0, 2.0, 3.0],   # "sentence 1" as 3 numbers
                        [4.0, 5.0, 6.0]])  # "sentence 2" as 3 numbers
# shape: (2, 3) — 2 samples, 3 features each

weights = mx.array([[0.1, 0.2],
                     [0.3, 0.4],
                     [0.5, 0.6]])
# shape: (3, 2) — transforms 3 features into 2 features

bias = mx.array([0.01, 0.02])
# shape: (2,) — one bias per output feature

output = input_data @ weights + bias

print("input  shape:", input_data.shape, " — 2 samples, 3 features")
print("weight shape:", weights.shape, " — 3 inputs → 2 outputs")
print("output shape:", output.shape, " — 2 samples, 2 outputs")
print()
print("input:\n", input_data)
print("weights:\n", weights)
print("output = input @ weights + bias:\n", output)

# === The shape rule ===
#
# (2, 3) @ (3, 2) → (2, 2)
#      ^    ^
#      these must match, the rest becomes the output shape
#
# If they don't match, you get an error. Most ML bugs are shape mismatches.

print("\n--- Why this matters ---")
print("A GPT-2 Small has 12 layers. Each layer has ~4 matmuls.")
print("That's ~48 matmuls per token. For 1000 tokens = 48,000 matmuls.")
print("Everything else (softmax, layer norm, etc.) is < 10% of the compute.")
