"""Loss and optimization module for Forward-Forward algorithm.

This module implements the goodness function, probability computation,
and localized loss function for each layer in the FF algorithm.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def calculate_goodness(hidden_activities: ArrayLike) -> Array:
    """Calculate the goodness of hidden layer activities.

    Goodness is defined as the sum of squared neural activities.
    This is computed per sample (summing over features).

    Args:
        hidden_activities: Array of shape (batch_size, features) containing
            the pre-normalization activations of a hidden layer.

    Returns:
        Goodness values of shape (batch_size,).
    """
    hidden_activities = jnp.asarray(hidden_activities)
    return jnp.sum(hidden_activities**2, axis=-1)


def compute_probability(goodness: ArrayLike, threshold: float) -> Array:
    """Compute the probability that input is positive data.

    Implements the logistic function: p(positive) = σ(goodness - θ)
    where σ is the sigmoid function.

    Args:
        goodness: Goodness values of shape (batch_size,).
        threshold: Threshold value θ to subtract from goodness.

    Returns:
        Probabilities of shape (batch_size,).
    """
    goodness = jnp.asarray(goodness)
    return jax.nn.sigmoid(goodness - threshold)


def layer_loss_fn(
    positive_activations: ArrayLike,
    negative_activations: ArrayLike,
    threshold: float,
) -> Tuple[Array, Array]:
    """Compute the localized loss for a single FF layer.

    The loss uses binary cross-entropy to push:
    - Positive samples toward p(positive) = 1 (goodness >> threshold)
    - Negative samples toward p(positive) = 0 (goodness << threshold)

    Args:
        positive_activations: Pre-normalization activations from positive data
            of shape (batch_size, features).
        negative_activations: Pre-normalization activations from negative data
            of shape (batch_size, features).
        threshold: Threshold value for probability computation.

    Returns:
        Tuple of (total_loss, dict with individual losses for logging).
    """
    positive_activations = jnp.asarray(positive_activations)
    negative_activations = jnp.asarray(negative_activations)

    # Calculate goodness for positive and negative samples
    pos_goodness = calculate_goodness(positive_activations)
    neg_goodness = calculate_goodness(negative_activations)

    # Compute probabilities
    pos_prob = compute_probability(pos_goodness, threshold)
    neg_prob = compute_probability(neg_goodness, threshold)

    # Binary cross-entropy loss
    # For positive: -log(p) -> push p toward 1
    # For negative: -log(1-p) -> push p toward 0
    eps = 1e-8  # Numerical stability
    pos_loss = -jnp.mean(jnp.log(pos_prob + eps))
    neg_loss = -jnp.mean(jnp.log(1 - neg_prob + eps))

    total_loss = pos_loss + neg_loss

    return total_loss, {
        "pos_loss": pos_loss,
        "neg_loss": neg_loss,
        "pos_goodness_mean": jnp.mean(pos_goodness),
        "neg_goodness_mean": jnp.mean(neg_goodness),
        "pos_prob_mean": jnp.mean(pos_prob),
        "neg_prob_mean": jnp.mean(neg_prob),
    }


def layer_loss_fn_pure(
    positive_activations: ArrayLike,
    negative_activations: ArrayLike,
    threshold: float,
) -> Array:
    """Pure loss function for JIT compilation (returns only scalar loss).

    This is a simplified version of layer_loss_fn that returns only the
    scalar loss value, suitable for use with jax.grad.

    Args:
        positive_activations: Pre-normalization activations from positive data.
        negative_activations: Pre-normalization activations from negative data.
        threshold: Threshold value for probability computation.

    Returns:
        Scalar total loss value.
    """
    positive_activations = jnp.asarray(positive_activations)
    negative_activations = jnp.asarray(negative_activations)

    pos_goodness = calculate_goodness(positive_activations)
    neg_goodness = calculate_goodness(negative_activations)

    # Use log-sigmoid for numerical stability
    # log(sigmoid(x)) = -softplus(-x) = -log(1 + exp(-x))
    # log(1 - sigmoid(x)) = log(sigmoid(-x)) = -softplus(x) = -log(1 + exp(x))
    pos_logit = pos_goodness - threshold
    neg_logit = neg_goodness - threshold

    # Binary cross-entropy with log-sigmoid for stability
    # pos_loss = -mean(log(sigmoid(pos_logit))) = mean(softplus(-pos_logit))
    # neg_loss = -mean(log(1 - sigmoid(neg_logit))) = mean(softplus(neg_logit))
    pos_loss = jnp.mean(jax.nn.softplus(-pos_logit))
    neg_loss = jnp.mean(jax.nn.softplus(neg_logit))

    return pos_loss + neg_loss


def compute_layer_goodness_sum(
    activations: ArrayLike,
    exclude_first: bool = True,
) -> Array:
    """Compute the sum of goodness across layers for inference.

    During inference, we accumulate goodness from all layers except
    the first hidden layer (as specified in the FF paper).

    Args:
        activations: List of pre-normalization activations from each layer,
            each of shape (batch_size, features).
        exclude_first: If True, exclude the first layer from the sum.

    Returns:
        Sum of goodness values of shape (batch_size,).
    """
    activations = jnp.asarray(activations)

    if exclude_first:
        start_idx = 1
    else:
        start_idx = 0

    total_goodness = jnp.zeros(activations.shape[0])
    for i, layer_acts in enumerate(activations):
        if i >= start_idx:
            total_goodness = total_goodness + calculate_goodness(layer_acts)

    return total_goodness
