"""Step 2: Build the model and see what it does BEFORE training."""

import mlx.core as mx
import mlx.nn as nn
import data


# === The Model ===
#
# nn.Module is MLX's base class for anything with learnable weights.
# nn.Linear(in, out) is one layer: it does `output = input @ weights + bias`
# nn.relu: replace negatives with zero (lets the network learn non-linear patterns)
#
# Data flows: 784 → 32 → 32 → 10

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 32)    # 784 pixels → 32 hidden features
        self.layer2 = nn.Linear(32, 32)     # 32 → 32 (deeper patterns)
        self.layer3 = nn.Linear(32, 10)     # 32 → 10 digit scores

    def __call__(self, x):
        x = nn.relu(self.layer1(x))   # matmul + relu
        x = nn.relu(self.layer2(x))   # matmul + relu
        x = self.layer3(x)            # matmul (no relu — raw scores out)
        return x


model = MLP()

# Let's look at what's inside
print("=== Model weights ===")
for name, param in model.parameters().items():
    for layer_name, w in param.items():
        print(f"  {name}.{layer_name}: shape {w.shape}")

# Feed it one image BEFORE any training (weights are random)
train_images, train_labels, _, _ = data.load()
sample = mx.array(train_images[0:1])   # shape (1, 784) — one image
label = train_labels[0]

scores = model(sample)
prediction = mx.argmax(scores, axis=1).item()

print(f"\n=== Before training ===")
print(f"True label: {label}")
print(f"Model output (10 scores): {scores}")
print(f"Model's guess: {prediction}")
print(f"Correct? {prediction == label}")
print(f"\nThe scores are random garbage — the model hasn't learned anything yet.")
