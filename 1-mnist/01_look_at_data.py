"""Step 1: Download MNIST and look at what we're working with."""

import data
import numpy as np

train_images, train_labels, test_images, test_labels = data.load()

print("=== MNIST Dataset ===")
print(f"Training: {train_images.shape[0]} images")
print(f"Test:     {test_images.shape[0]} images")
print(f"Each image: {train_images.shape[1]} values (28x28 pixels, flattened into a row)")
print(f"Labels: digits 0-9")

# Let's look at one image — just print it as ASCII art
sample = train_images[0].reshape(28, 28)
label = train_labels[0]

print(f"\n=== Sample image (label: {label}) ===")
for row in sample:
    line = ""
    for pixel in row:
        if pixel > 0.75:
            line += "##"
        elif pixel > 0.25:
            line += ".."
        else:
            line += "  "
    print(line)

print(f"\nThis image is a {label}.")
print(f"The model's job: take 784 pixel values as input, output which digit (0-9) it is.")
