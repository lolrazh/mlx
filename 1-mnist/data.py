"""Load the MNIST dataset from pre-downloaded gzip files in /tmp."""

import gzip
import os

import numpy as np


def load(data_dir="/tmp"):
    """Returns (train_images, train_labels, test_images, test_labels) as numpy arrays.

    Expects gzip files already in data_dir (downloaded via curl).
    """

    files = [
        ("train-images-idx3-ubyte.gz", 16, (-1, 28 * 28)),
        ("t10k-images-idx3-ubyte.gz", 16, (-1, 28 * 28)),
        ("train-labels-idx1-ubyte.gz", 8, (-1,)),
        ("t10k-labels-idx1-ubyte.gz", 8, (-1,)),
    ]

    arrays = []
    for fname, offset, shape in files:
        path = os.path.join(data_dir, fname)
        with gzip.open(path, "rb") as f:
            arrays.append(np.frombuffer(f.read(), np.uint8, offset=offset).reshape(shape))

    train_images, test_images, train_labels, test_labels = arrays

    # Normalize pixel values from 0-255 to 0.0-1.0
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    return train_images, train_labels.astype(np.uint32), test_images, test_labels.astype(np.uint32)
