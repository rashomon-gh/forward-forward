"""Evaluation module for Forward-Forward algorithm.

This module implements inference and accuracy evaluation for the
Forward-Forward algorithm. Key features:
- Label iteration for classification
- Goodness accumulation (excluding first layer)
- Batch evaluation for efficiency
"""

from typing import Dict, List

import jax.numpy as jnp
from flax.core import unfreeze
from jax import Array
from jax.typing import ArrayLike

from .loss import calculate_goodness
from .network import FFNetwork


def apply_network_for_inference(
    params: Dict,
    x: ArrayLike,
    num_layers: int = 4,
) -> List[Array]:
    """Apply network and return all pre-normalization activations.

    This function computes activations for all layers without
    stop_gradient, suitable for inference.

    Args:
        params: Network parameters.
        x: Input data of shape (batch_size, input_features).
        num_layers: Number of layers in the network.

    Returns:
        List of pre-normalization activations for each layer.
    """
    x = jnp.asarray(x)
    params_dict = unfreeze(params)["params"]
    activations = []

    for i in range(num_layers):
        layer_key = f"Dense_{i}"
        W = params_dict[layer_key]["kernel"]
        b = params_dict[layer_key]["bias"]
        x = jnp.dot(x, W) + b
        x = jnp.maximum(x, 0)  # ReLU

        # Store pre-normalization activations
        activations.append(x)

        # Apply layer normalization for next layer
        norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1e-8)
        x = x / norm

    return activations


def predict_single_image(
    params: Dict,
    image: ArrayLike,
    network: FFNetwork,
    num_classes: int = 10,
    exclude_first_layer: bool = True,
) -> int:
    """Predict the label for a single image using FF inference.

    This function:
    1. Runs the network 10 times (once per label)
    2. Accumulates goodness from layers 2, 3, 4 (excludes layer 1)
    3. Returns the label with highest accumulated goodness

    Args:
        params: Network parameters.
        image: Single flattened image of shape (784,).
        network: FFNetwork instance.
        num_classes: Number of classes (10 for MNIST).
        exclude_first_layer: If True, exclude first layer from goodness sum.

    Returns:
        Predicted label (integer 0-9).
    """
    image = jnp.asarray(image)
    num_layers = len(network.layer_sizes)

    goodness_per_label = []

    for label in range(num_classes):
        # Embed label into image
        one_hot = jnp.zeros(num_classes, dtype=image.dtype)
        one_hot = one_hot.at[label].set(1.0)
        labeled_image = jnp.concatenate([one_hot, image[num_classes:]])

        # Add batch dimension
        labeled_image = labeled_image[jnp.newaxis, :]

        # Get activations for all layers
        activations = apply_network_for_inference(params, labeled_image, num_layers)

        # Sum goodness for layers (excluding first if specified)
        start_idx = 1 if exclude_first_layer else 0
        total_goodness = sum(
            calculate_goodness(acts) for acts in activations[start_idx:]
        )

        goodness_per_label.append(float(total_goodness))

    return int(jnp.argmax(jnp.array(goodness_per_label)))


def predict_batch(
    params: Dict,
    images: ArrayLike,
    network: FFNetwork,
    num_classes: int = 10,
    exclude_first_layer: bool = True,
) -> Array:
    """Predict labels for a batch of images.

    This is a more efficient batch version that processes all labels
    for all images simultaneously.

    Args:
        params: Network parameters.
        images: Batch of images of shape (batch_size, 784).
        network: FFNetwork instance.
        num_classes: Number of classes.
        exclude_first_layer: If True, exclude first layer from goodness sum.

    Returns:
        Predicted labels of shape (batch_size,).
    """
    images = jnp.asarray(images)
    batch_size = images.shape[0]
    num_layers = len(network.layer_sizes)

    # Create all label-image combinations
    # Shape: (num_classes, batch_size, 784)
    all_inputs = []
    for label in range(num_classes):
        one_hot = jnp.zeros((batch_size, num_classes), dtype=images.dtype)
        one_hot = one_hot.at[:, label].set(1.0)
        labeled_images = jnp.concatenate([one_hot, images[:, num_classes:]], axis=1)
        all_inputs.append(labeled_images)

    # Stack and reshape for batch processing
    all_inputs = jnp.stack(all_inputs, axis=0)  # (num_classes, batch_size, 784)
    original_shape = all_inputs.shape
    flat_inputs = all_inputs.reshape(
        -1, original_shape[-1]
    )  # (num_classes * batch_size, 784)

    # Get activations for all combinations
    activations = apply_network_for_inference(params, flat_inputs, num_layers)

    # Compute goodness for each layer and sum
    start_idx = 1 if exclude_first_layer else 0
    goodness_per_layer = [calculate_goodness(acts) for acts in activations[start_idx:]]
    total_goodness = jnp.sum(jnp.stack(goodness_per_layer, axis=0), axis=0)

    # Reshape back to (num_classes, batch_size)
    total_goodness = total_goodness.reshape(num_classes, batch_size)

    # Get best label for each image
    predictions = jnp.argmax(total_goodness, axis=0)

    return predictions


def evaluate_accuracy(
    params: Dict,
    images: ArrayLike,
    labels: ArrayLike,
    network: FFNetwork,
    batch_size: int = 1000,
    num_classes: int = 10,
) -> float:
    """Evaluate classification accuracy on a dataset.

    Args:
        params: Network parameters.
        images: Test images of shape (num_samples, 784).
        labels: Test labels of shape (num_samples,).
        network: FFNetwork instance.
        batch_size: Batch size for evaluation.
        num_classes: Number of classes.

    Returns:
        Classification accuracy (float between 0 and 1).
    """
    images = jnp.asarray(images)
    labels = jnp.asarray(labels)

    num_samples = images.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    correct = 0
    total = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        batch_images = images[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        predictions = predict_batch(params, batch_images, network, num_classes)

        correct += int(jnp.sum(predictions == batch_labels))
        total += batch_labels.shape[0]

    return correct / total


def evaluate_with_details(
    params: Dict,
    images: ArrayLike,
    labels: ArrayLike,
    network: FFNetwork,
    batch_size: int = 1000,
    num_classes: int = 10,
) -> Dict[str, float]:
    """Evaluate with detailed metrics.

    Args:
        params: Network parameters.
        images: Test images.
        labels: Test labels.
        network: FFNetwork instance.
        batch_size: Batch size for evaluation.
        num_classes: Number of classes.

    Returns:
        Dictionary with accuracy and per-class metrics.
    """
    images = jnp.asarray(images)
    labels = jnp.asarray(labels)

    num_samples = images.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    all_predictions = []
    all_labels = []

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        batch_images = images[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        predictions = predict_batch(params, batch_images, network, num_classes)

        all_predictions.append(predictions)
        all_labels.append(batch_labels)

    all_predictions = jnp.concatenate(all_predictions)
    all_labels = jnp.concatenate(all_labels)

    # Overall accuracy
    accuracy = float(jnp.mean(all_predictions == all_labels))

    # Per-class accuracy
    per_class_acc = {}
    for c in range(num_classes):
        mask = all_labels == c
        if jnp.sum(mask) > 0:
            class_acc = float(jnp.mean(all_predictions[mask] == all_labels[mask]))
            per_class_acc[f"class_{c}_accuracy"] = class_acc

    return {
        "accuracy": accuracy,
        **per_class_acc,
    }
