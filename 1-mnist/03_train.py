"""Step 3: Train the model. This is the whole game."""

import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import data


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 32)
        self.layer2 = nn.Linear(32, 32)
        self.layer3 = nn.Linear(32, 10)

    def __call__(self, x):
        x = nn.relu(self.layer1(x))
        x = nn.relu(self.layer2(x))
        return self.layer3(x)


def loss_fn(model, images, labels):
    """How wrong is the model? Lower = better."""
    scores = model(images)
    return nn.losses.cross_entropy(scores, labels, reduction="mean")


# --- Setup ---

train_images, train_labels, test_images, test_labels = map(mx.array, data.load())

model = MLP()
mx.eval(model.parameters())

# SGD = stochastic gradient descent. The simplest optimizer.
# It just does: weight = weight - learning_rate * gradient
optimizer = optim.SGD(learning_rate=0.1)

# This is the key MLX function: computes the loss AND the gradients in one call.
loss_and_grad = nn.value_and_grad(model, loss_fn)

# --- Training loop ---

batch_size = 256

print("=== Training ===\n")

for epoch in range(10):
    tic = time.perf_counter()

    # Shuffle the training data each epoch
    perm = mx.array(np.random.permutation(train_labels.size))

    # Process in batches
    for start in range(0, train_labels.size, batch_size):
        ids = perm[start : start + batch_size]
        batch_images = train_images[ids]
        batch_labels = train_labels[ids]

        # THE training step — three lines that do everything:
        loss, grads = loss_and_grad(model, batch_images, batch_labels)  # 1. forward + backward
        optimizer.update(model, grads)                                  # 2. nudge weights
        mx.eval(model.state)                                            # 3. actually compute it

    # Check accuracy on test set
    test_scores = model(test_images)
    test_preds = mx.argmax(test_scores, axis=1)
    accuracy = mx.mean(test_preds == test_labels).item()

    elapsed = time.perf_counter() - tic
    print(f"  Epoch {epoch:2d}  accuracy: {accuracy:.1%}  time: {elapsed:.2f}s")

print("\nDone. From random guessing (~10%) to reading handwriting in seconds.")
