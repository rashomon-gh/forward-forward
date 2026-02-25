"""Custom Flax layers for Forward-Forward algorithm.

This module implements the custom layer normalization and FFLayer
required for the Forward-Forward algorithm.
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax
from jax.typing import ArrayLike


def custom_layer_norm(x: ArrayLike, eps: float = 1e-8) -> jax.Array:
    """Custom layer normalization for Forward-Forward algorithm.

    This implements a simplified version of layer normalization that
    divides by the L2 norm of the activity vector WITHOUT subtracting
    the mean. This prevents subsequent layers from trivially distinguishing
    data based solely on the length of the activity vector.

    Args:
        x: Input array of shape (..., features).
        eps: Small constant for numerical stability.

    Returns:
        Normalized array of same shape as input.
    """
    x = jnp.asarray(x)
    # Compute L2 norm (length) of the activity vector
    norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + eps)
    # Divide by norm without subtracting mean
    return x / norm


class FFLayer(nn.Module):
    """Single Forward-Forward layer.

    This layer applies:
    1. Dense transformation
    2. ReLU activation
    3. Custom layer normalization (L2 norm, no mean subtraction)

    The layer normalization ensures that the next layer cannot trivially
    distinguish positive from negative data based on vector length alone.

    Attributes:
        features: Number of output units in the dense layer.
    """

    features: int

    @nn.compact
    def __call__(
        self, x: ArrayLike, return_pre_norm: bool = False
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Forward pass through the FF layer.

        Args:
            x: Input array of shape (batch_size, input_features).
            return_pre_norm: If True, return activations before normalization
                (needed for goodness calculation).

        Returns:
            If return_pre_norm is False:
                Normalized output of shape (batch_size, features).
            If return_pre_norm is True:
                Tuple of (normalized_output, pre_norm_activations).
        """
        # Dense transformation
        x = nn.Dense(self.features)(x)
        # ReLU activation
        x = nn.relu(x)

        if return_pre_norm:
            # Store pre-normalization activations for goodness calculation
            pre_norm = x
            x = custom_layer_norm(x)
            return x, pre_norm
        else:
            x = custom_layer_norm(x)
            return x


class FFLayerWithStopGrad(nn.Module):
    """FF Layer with stop_gradient for layer isolation.

    This variant applies stop_gradient to the input, ensuring that
    gradients do not flow backward to previous layers during training.
    This is essential for the greedy layer-wise training of FF.

    Attributes:
        features: Number of output units in the dense layer.
    """

    features: int

    @nn.compact
    def __call__(
        self, x: ArrayLike, return_pre_norm: bool = False, train: bool = True
    ) -> jax.Array | tuple[jax.Array, jax.Array]:
        """Forward pass with optional gradient stopping.

        Args:
            x: Input array of shape (batch_size, input_features).
            return_pre_norm: If True, return activations before normalization.
            train: If True, apply stop_gradient to input for layer isolation.

        Returns:
            Normalized output, optionally with pre-normalization activations.
        """
        # Stop gradients from flowing to previous layers during training
        if train:
            x = lax.stop_gradient(x)

        # Dense transformation
        x = nn.Dense(self.features)(x)
        # ReLU activation
        x = nn.relu(x)

        if return_pre_norm:
            pre_norm = x
            x = custom_layer_norm(x)
            return x, pre_norm
        else:
            x = custom_layer_norm(x)
            return x
