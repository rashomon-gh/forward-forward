"""FFNetwork module for Forward-Forward algorithm.

This module implements the main FFNetwork that stacks multiple FFLayers
for the Forward-Forward algorithm.
"""

from typing import List, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import lax
from jax.typing import ArrayLike

from .layers import custom_layer_norm


class FFNetwork(nn.Module):
    """Forward-Forward Network.

    A stack of FFLayers for the Forward-Forward algorithm.
    For MNIST, the standard configuration is 4 hidden layers with 2000 ReLUs each.

    Key features:
    - Layer isolation via stop_gradient between layers
    - Returns intermediate activations for goodness calculation
    - Custom layer normalization (L2 norm, no mean subtraction)

    Attributes:
        layer_sizes: Tuple of layer sizes. Default is (2000, 2000, 2000, 2000).
    """

    layer_sizes: Tuple[int, ...] = (2000, 2000, 2000, 2000)

    @nn.compact
    def __call__(
        self,
        x: ArrayLike,
        return_intermediates: bool = False,
        train: bool = True,
    ) -> jax.Array | Tuple[List[jax.Array], List[jax.Array]]:
        """Forward pass through the FF network.

        Args:
            x: Input array of shape (batch_size, input_features).
            return_intermediates: If True, return intermediate activations.
            train: If True, apply stop_gradient between layers for training.

        Returns:
            If return_intermediates is False:
                Final output of shape (batch_size, last_layer_size).
            If return_intermediates is True:
                Tuple of (normalized_activations, pre_norm_activations), each
                being a list of activations from each layer.
        """
        x = jnp.asarray(x)

        if return_intermediates:
            normalized_activations = []
            pre_norm_activations = []

            for i, size in enumerate(self.layer_sizes):
                # Apply stop_gradient between layers (not on first layer input)
                if i > 0 and train:
                    x = lax.stop_gradient(x)

                # Dense transformation
                x = nn.Dense(size)(x)
                # ReLU activation
                x = nn.relu(x)

                # Store pre-normalization activations for goodness calculation
                pre_norm_activations.append(x)

                # Apply custom layer normalization
                x = custom_layer_norm(x)
                normalized_activations.append(x)

            return normalized_activations, pre_norm_activations
        else:
            for i, size in enumerate(self.layer_sizes):
                # Apply stop_gradient between layers
                if i > 0 and train:
                    x = lax.stop_gradient(x)

                x = nn.Dense(size)(x)
                x = nn.relu(x)
                x = custom_layer_norm(x)

            return x


class FFNetworkWithLayerNorm(FFNetwork):
    """FFNetwork variant that applies layer norm to input.

    This variant normalizes the input before passing it through the network,
    which can help with training stability.
    """

    @nn.compact
    def __call__(
        self,
        x: ArrayLike,
        return_intermediates: bool = False,
        train: bool = True,
    ) -> jax.Array | Tuple[List[jax.Array], List[jax.Array]]:
        """Forward pass with input normalization."""
        x = jnp.asarray(x)
        x = custom_layer_norm(x)
        return super().__call__(x, return_intermediates, train)


def create_ff_network(
    layer_sizes: Tuple[int, ...] = (2000, 2000, 2000, 2000),
) -> FFNetwork:
    """Factory function to create an FFNetwork with specified layer sizes.

    Args:
        layer_sizes: Tuple of layer sizes.

    Returns:
        FFNetwork instance.
    """
    return FFNetwork(layer_sizes=layer_sizes)
