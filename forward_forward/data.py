"""Data preparation module for Forward-Forward algorithm.

This module provides functions for embedding labels into MNIST images
and generating positive/negative training batches.
"""

from collections.abc import Generator
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def embed_label(image: ArrayLike, label: int, num_classes: int = 10) -> Array:
    """Embed a one-hot label into the first pixels of an image.

    MNIST images have black borders, so we replace the first 10 pixels
    with a one-hot representation of the label.

    Args:
        image: Flattened image array of shape (784,) for MNIST.
        label: Integer label to embed (0-9 for MNIST).
        num_classes: Number of classes (default 10 for MNIST).

    Returns:
        Modified image with label embedded in first num_classes pixels.
    """
    image = jnp.asarray(image)
    one_hot = jnp.zeros(num_classes, dtype=image.dtype)
    one_hot = one_hot.at[label].set(1.0)
    return jnp.concatenate([one_hot, image[num_classes:]])


def embed_labels_batch(
    images: ArrayLike, labels: ArrayLike, num_classes: int = 10
) -> Array:
    """Embed one-hot labels into a batch of images.

    Args:
        images: Batch of flattened images of shape (batch_size, 784).
        labels: Batch of integer labels of shape (batch_size,).
        num_classes: Number of classes (default 10 for MNIST).

    Returns:
        Batch of modified images with labels embedded.
    """
    images = jnp.asarray(images)
    labels = jnp.asarray(labels)

    batch_size = images.shape[0]
    one_hot = jnp.zeros((batch_size, num_classes), dtype=images.dtype)
    one_hot = one_hot.at[jnp.arange(batch_size), labels].set(1.0)

    return jnp.concatenate([one_hot, images[:, num_classes:]], axis=1)


def generate_wrong_labels(labels: ArrayLike, key: ArrayLike) -> Array:
    """Generate incorrect labels for negative samples.

    For each label, generates a random wrong label (different from original).

    Args:
        labels: Original labels of shape (batch_size,).
        key: JAX random key.

    Returns:
        Wrong labels of shape (batch_size,).
    """
    labels = jnp.asarray(labels)
    batch_size = labels.shape[0]

    # Generate random labels
    wrong_labels = jax.random.randint(key, (batch_size,), 0, 10)

    # Ensure wrong labels are different from original
    # If same, shift by 1 (mod 10)
    wrong_labels = jnp.where(
        wrong_labels == labels, (wrong_labels + 1) % 10, wrong_labels
    )

    return wrong_labels


def generate_batch(
    images: ArrayLike,
    labels: ArrayLike,
    key: ArrayLike,
    num_classes: int = 10,
) -> Tuple[Array, Array]:
    """Generate positive and negative data batches for FF training.

    Positive data: images with correct labels embedded.
    Negative data: same images with incorrect labels embedded.

    Args:
        images: Batch of flattened images of shape (batch_size, 784).
        labels: Batch of integer labels of shape (batch_size,).
        key: JAX random key for generating wrong labels.
        num_classes: Number of classes (default 10 for MNIST).

    Returns:
        Tuple of (positive_batch, negative_batch), each of shape (batch_size, 784).
    """
    images = jnp.asarray(images)
    labels = jnp.asarray(labels)

    # Generate positive samples with correct labels
    positive = embed_labels_batch(images, labels, num_classes)

    # Generate negative samples with wrong labels
    wrong_labels = generate_wrong_labels(labels, key)
    negative = embed_labels_batch(images, wrong_labels, num_classes)

    return positive, negative


def load_mnist(
    train: bool = True, flatten: bool = True, normalize: bool = True
) -> Tuple[Array, Array]:
    """Load MNIST dataset using scikit-learn's fetch_openml.

    Args:
        train: If True, load training set (60k samples), else test set (10k samples).
        flatten: If True, flatten images to 1D vectors (784,).
        normalize: If True, normalize pixel values to [0, 1].

    Returns:
        Tuple of (images, labels) as JAX arrays.
    """
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    # Fetch MNIST data
    # Use parser='liac-arff' to avoid pandas dependency
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X, y = mnist.data, mnist.target.astype(int)

    # Split into train/test (MNIST in fetch_openml is already shuffled)
    if train:
        images, _, labels, _ = train_test_split(X, y, test_size=10000, random_state=42)
    else:
        _, images, _, labels = train_test_split(X, y, test_size=10000, random_state=42)

    # Normalize to [0, 1]
    if normalize:
        images = images.astype(float) / 255.0

    # Convert to JAX arrays
    images = jnp.array(images)
    labels = jnp.array(labels)

    return images, labels


def create_dataloader(
    images: ArrayLike,
    labels: ArrayLike,
    batch_size: int,
    key: ArrayLike,
    shuffle: bool = True,
) -> Generator[Tuple[Array, Array, Array], None, None]:
    """Create batches of data for training.

    Args:
        images: Array of images of shape (num_samples, 784).
        labels: Array of labels of shape (num_samples,).
        batch_size: Size of each batch.
        key: JAX random key for shuffling.
        shuffle: Whether to shuffle the data.

    Returns:
        Generator yielding (batch_images, batch_labels, subkey) tuples.
    """
    images = jnp.asarray(images)
    labels = jnp.asarray(labels)

    num_samples = images.shape[0]
    indices = jnp.arange(num_samples)

    if shuffle:
        indices = jax.random.permutation(key, indices)

    # Split key for batch generation
    num_batches = (num_samples + batch_size - 1) // batch_size
    keys = jax.random.split(key, num_batches + 1)
    _ = keys[0]
    batch_keys = keys[1:]

    for i, batch_key in enumerate(batch_keys):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        yield images[batch_indices], labels[batch_indices], batch_key
