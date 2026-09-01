"""Provided helpers for the Gradient Descent Methods lab.

These functions handle MNIST's binary file format, the supplied softmax
regression model, and the small adapters needed to use JAX derivatives with
SciPy's optimization routines.
"""

import struct
from pathlib import Path

import jax
import jax.numpy as jnp
from jax import grad
import numpy as np


N_CLASSES = 10


def read_idx_images(path, limit=None):
    """Read flattened IDX images and normalize them as float32."""
    with open(path, "rb") as file:
        magic, count, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(
                f"Expected image magic number 2051, received {magic}."
            )
        count = count if limit is None else min(limit, count)
        images = np.fromfile(
            file, dtype=np.uint8, count=count * rows * cols
        ).reshape(count, rows * cols)
    return images.astype(np.float32) / np.float32(255.)


def read_idx_labels(path, limit=None):
    """Read labels from an IDX label file."""
    with open(path, "rb") as file:
        magic, count = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(
                f"Expected label magic number 2049, received {magic}."
            )
        count = count if limit is None else min(limit, count)
        return np.fromfile(file, dtype=np.uint8, count=count)


def load_mnist(data_dir=None, train_size=None, test_size=None):
    """Load MNIST, optionally limiting either split for quick debugging."""
    if data_dir is None:
        data_dir = Path("public") if Path("public").exists() else Path(".")
    data_dir = Path(data_dir)

    x_train = read_idx_images(data_dir / "train-images", train_size)
    y_train = read_idx_labels(data_dir / "train-labels", train_size)
    x_test = read_idx_images(data_dir / "test-images", test_size)
    y_test = read_idx_labels(data_dir / "test-labels", test_size)
    return (x_train, y_train), (x_test, y_test)


def cross_entropy_loss(weights, x, y):
    """Return mean softmax cross-entropy for flattened model parameters."""
    W = weights[:-N_CLASSES].reshape(x.shape[1], N_CLASSES)
    b = weights[-N_CLASSES:]
    log_probs = jax.nn.log_softmax(x @ W + b, axis=1)
    return -jnp.mean(log_probs[jnp.arange(y.shape[0]), y])


def predict(weights, x):
    """Predict the digit represented by each image."""
    W = np.asarray(weights[:-N_CLASSES]).reshape(x.shape[1], N_CLASSES)
    b = np.asarray(weights[-N_CLASSES:])
    return np.argmax(x @ W + b, axis=1)


def accuracy(weights, x, y):
    """Return the proportion of correctly classified images."""
    return np.mean(predict(weights, x) == y)


def hessian_vector_product(weights, vector, x, y):
    """Return the MNIST loss Hessian applied to one vector."""
    def gradient_function(w):
        return grad(cross_entropy_loss, argnums=0)(w, x, y)

    return jax.jvp(gradient_function, (weights,), (vector,))[1]


def scipy_hessian_product(weights, vector, full_hvp, x, y,
                          damping=1e-2):
    """Return a damped Hessian-vector product in SciPy-compatible dtypes."""
    weights = np.asarray(weights, dtype=np.float32)
    vector = np.asarray(vector, dtype=np.float32)
    product = full_hvp(weights, vector, x, y)
    return np.asarray(product + damping * vector, dtype=np.float64)
