"""Training module for Forward-Forward algorithm.

This module implements the JIT-compiled training step and training loop
for the Forward-Forward algorithm. Key features:
- Layer-wise gradient updates (no cross-layer gradient flow)
- Separate optimizer states for each layer
- JIT-compatible training functions
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.core import unfreeze
from flax.training.train_state import TrainState
from jax import Array
from jax.typing import ArrayLike

from .loss import layer_loss_fn_pure
from .network import FFNetwork


def create_train_state(
    network: FFNetwork,
    key: ArrayLike,
    input_shape: Tuple[int, ...],
    learning_rate: float = 0.001,
) -> TrainState:
    """Create initial training state for the network.

    Args:
        network: FFNetwork instance.
        key: JAX random key for initialization.
        input_shape: Shape of input (excluding batch dimension).
        learning_rate: Learning rate for Adam optimizer.

    Returns:
        TrainState with initialized parameters and optimizer.
    """
    key = jax.random.PRNGKey(key[0]) if isinstance(key, int) else key
    input_dummy = jnp.ones((1,) + input_shape)
    params = network.init(key, input_dummy, train=False)

    optimizer = optax.adam(learning_rate=learning_rate)
    return TrainState.create(apply_fn=network.apply, params=params, tx=optimizer)


def get_layer_params(params: Dict, layer_idx: int) -> Dict:
    """Extract parameters for a specific layer.

    Args:
        params: Full network parameters.
        layer_idx: Index of the layer (0-indexed).

    Returns:
        Parameters for the specified layer.
    """
    params_dict = unfreeze(params)
    # Flax Dense layers are named "Dense_0", "Dense_1", etc.
    layer_key = f"Dense_{layer_idx}"
    if layer_key in params_dict["params"]:
        return {"params": {layer_key: params_dict["params"][layer_key]}}
    return None


def apply_network_to_layer(
    params: Dict,
    x: ArrayLike,
    layer_idx: int,
    network: FFNetwork,
) -> Array:
    """Apply network up to and including a specific layer.

    This function computes activations for a specific layer by
    running the forward pass through all preceding layers.

    Args:
        params: Network parameters.
        x: Input data of shape (batch_size, input_features).
        layer_idx: Target layer index.
        network: FFNetwork instance.

    Returns:
        Pre-normalization activations for the specified layer.
    """
    x = jnp.asarray(x)
    params_dict = unfreeze(params)["params"]

    # Forward pass through layers up to and including layer_idx
    for i in range(layer_idx + 1):
        layer_key = f"Dense_{i}"
        # Dense transformation
        W = params_dict[layer_key]["kernel"]
        b = params_dict[layer_key]["bias"]
        x = jnp.dot(x, W) + b
        # ReLU activation
        x = jnp.maximum(x, 0)

        # Store pre-normalization activations for the target layer
        if i == layer_idx:
            return x

        # Apply custom layer normalization for intermediate layers
        norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1e-8)
        x = x / norm

        # Stop gradient between layers
        x = jax.lax.stop_gradient(x)

    return x


def make_layer_loss_fn(
    network: FFNetwork,
    layer_idx: int,
    threshold: float,
):
    """Create a loss function for a specific layer.

    Args:
        network: FFNetwork instance.
        layer_idx: Layer index to compute loss for.
        threshold: Threshold for probability computation.

    Returns:
        Loss function that takes (params, pos_data, neg_data).
    """

    def loss_fn(params, pos_data, neg_data):
        # Get pre-normalization activations for this layer
        pos_acts = apply_network_to_layer(params, pos_data, layer_idx, network)
        neg_acts = apply_network_to_layer(params, neg_data, layer_idx, network)

        # Compute loss
        return layer_loss_fn_pure(pos_acts, neg_acts, threshold)

    return loss_fn


@jax.jit
def train_layer_step(
    state: TrainState,
    pos_data: ArrayLike,
    neg_data: ArrayLike,
    layer_idx: int,
    threshold: float,
    network: FFNetwork,
) -> Tuple[TrainState, Array]:
    """JIT-compiled training step for a single layer.

    This function:
    1. Computes activations up to the target layer
    2. Calculates the localized loss for that layer
    3. Computes gradients and applies updates

    Args:
        state: Current training state.
        pos_data: Positive samples of shape (batch_size, input_features).
        neg_data: Negative samples of shape (batch_size, input_features).
        layer_idx: Index of the layer to update.
        threshold: Threshold for probability computation.
        network: FFNetwork instance.

    Returns:
        Tuple of (updated_state, loss_value).
    """
    pos_data = jnp.asarray(pos_data)
    neg_data = jnp.asarray(neg_data)

    def loss_fn(params):
        pos_acts = apply_network_to_layer(params, pos_data, layer_idx, network)
        neg_acts = apply_network_to_layer(params, neg_data, layer_idx, network)
        return layer_loss_fn_pure(pos_acts, neg_acts, threshold)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)

    return state, loss


def create_layer_train_step(network: FFNetwork, threshold: float, layer_idx: int):
    """Create a JIT-compiled training step for a specific layer.

    This factory function creates a specialized training step that
    captures the network, threshold, and layer index in a closure.

    Args:
        network: FFNetwork instance.
        threshold: Threshold for probability computation.
        layer_idx: Index of the layer to train.

    Returns:
        JIT-compiled training step function.
    """

    @jax.jit
    def train_step(
        state: TrainState, pos_data: ArrayLike, neg_data: ArrayLike
    ) -> Tuple[TrainState, Array]:
        """Training step for a specific layer."""

        def loss_fn(params):
            pos_acts = apply_network_to_layer(params, pos_data, layer_idx, network)
            neg_acts = apply_network_to_layer(params, neg_data, layer_idx, network)
            return layer_loss_fn_pure(pos_acts, neg_acts, threshold)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    return train_step


def train_epoch(
    state: TrainState,
    images: ArrayLike,
    labels: ArrayLike,
    key: ArrayLike,
    network: FFNetwork,
    threshold: float,
    batch_size: int = 128,
    num_layers: int = 4,
) -> Tuple[TrainState, Dict[str, float]]:
    """Train for one epoch over the dataset.

    Args:
        state: Current training state.
        images: Training images of shape (num_samples, 784).
        labels: Training labels of shape (num_samples,).
        key: JAX random key.
        network: FFNetwork instance.
        threshold: Threshold for probability computation.
        batch_size: Batch size for training.
        num_layers: Number of layers in the network.

    Returns:
        Tuple of (updated_state, metrics_dict).
    """
    images = jnp.asarray(images)
    labels = jnp.asarray(labels)
    key = jax.random.PRNGKey(key[0]) if isinstance(key, int) else key

    num_samples = images.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    # Shuffle data
    perm = jax.random.permutation(key, num_samples)
    images = images[perm]
    labels = labels[perm]

    total_loss = 0.0
    layer_losses = [0.0] * num_layers

    # Create training steps for each layer
    train_steps = [
        create_layer_train_step(network, threshold, i) for i in range(num_layers)
    ]

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        batch_images = images[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]

        # Generate positive and negative samples
        key, subkey = jax.random.split(key)
        pos_data, neg_data = generate_batch_simple(batch_images, batch_labels, subkey)

        # Train each layer
        for layer_idx in range(num_layers):
            state, loss = train_steps[layer_idx](state, pos_data, neg_data)
            layer_losses[layer_idx] += float(loss)
            total_loss += float(loss)

    # Average losses
    metrics = {
        "total_loss": total_loss / (num_batches * num_layers),
        **{f"layer_{i}_loss": layer_losses[i] / num_batches for i in range(num_layers)},
    }

    return state, metrics


def generate_batch_simple(
    images: ArrayLike,
    labels: ArrayLike,
    key: ArrayLike,
    num_classes: int = 10,
) -> Tuple[Array, Array]:
    """Generate positive and negative batches (simple version for training).

    Args:
        images: Batch of images of shape (batch_size, 784).
        labels: Batch of labels of shape (batch_size,).
        key: JAX random key.
        num_classes: Number of classes.

    Returns:
        Tuple of (positive_batch, negative_batch).
    """
    images = jnp.asarray(images)
    labels = jnp.asarray(labels)

    batch_size = images.shape[0]

    # Create one-hot encoding
    one_hot_pos = jnp.zeros((batch_size, num_classes), dtype=images.dtype)
    one_hot_pos = one_hot_pos.at[jnp.arange(batch_size), labels].set(1.0)
    positive = jnp.concatenate([one_hot_pos, images[:, num_classes:]], axis=1)

    # Generate wrong labels
    wrong_labels = jax.random.randint(key, (batch_size,), 0, num_classes)
    wrong_labels = jnp.where(
        wrong_labels == labels, (wrong_labels + 1) % num_classes, wrong_labels
    )

    one_hot_neg = jnp.zeros((batch_size, num_classes), dtype=images.dtype)
    one_hot_neg = one_hot_neg.at[jnp.arange(batch_size), wrong_labels].set(1.0)
    negative = jnp.concatenate([one_hot_neg, images[:, num_classes:]], axis=1)

    return positive, negative
