#!/usr/bin/env python3
"""Main training script for Forward-Forward algorithm on MNIST.

This script trains a Forward-Forward network on MNIST and evaluates
its performance. The FF algorithm uses two forward passes (positive
and negative data) instead of traditional backpropagation.

Usage:
    python main.py

Hyperparameters can be modified at the top of the script.
"""

import time

import jax
import jax.numpy as jnp

from forward_forward import (
    FFNetwork,
    create_train_state,
    evaluate_accuracy,
    load_mnist,
    train_epoch,
)

# ============================================================================
# Hyperparameters
# ============================================================================

# Network architecture
LAYER_SIZES = (2000, 2000, 2000, 2000)  # 4 layers with 2000 ReLUs each

# Training
LEARNING_RATE = 0.001
BATCH_SIZE = 128
NUM_EPOCHS = 60
THRESHOLD = 2.0  # θ in the FF paper

# Evaluation
EVAL_BATCH_SIZE = 1000

# Random seed
SEED = 42


def main():
    """Main training function."""
    print("=" * 60)
    print("Forward-Forward Algorithm on MNIST")
    print("=" * 60)

    # Print device info
    print(f"\nJAX devices: {jax.devices()}")
    print(f"Using: {jax.devices()[0].platform.upper()}")

    # Initialize random key
    key = jax.random.PRNGKey(SEED)

    # Load MNIST data
    print("\nLoading MNIST data...")
    train_images, train_labels = load_mnist(train=True)
    test_images, test_labels = load_mnist(train=False)

    print(f"Training samples: {train_images.shape[0]}")
    print(f"Test samples: {test_images.shape[0]}")
    print(f"Image shape: {train_images.shape[1]}")

    # Create network
    print("\nCreating network...")
    network = FFNetwork(layer_sizes=LAYER_SIZES)
    print(f"Layer sizes: {LAYER_SIZES}")
    print(
        f"Total parameters: {sum(p.size for p in jax.tree_util.tree_leaves(network.init(key, jnp.ones((1, 784)))))}"
    )

    # Initialize training state
    key, init_key = jax.random.split(key)
    state = create_train_state(
        network=network,
        key=init_key,
        input_shape=(784,),
        learning_rate=LEARNING_RATE,
    )

    # Training loop
    print("\nStarting training...")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Threshold (θ): {THRESHOLD}")
    print(f"Number of epochs: {NUM_EPOCHS}")
    print("-" * 60)

    best_accuracy = 0.0
    accuracy = 0.0
    metrics = {
        "total_loss": 0.0,
        **{f"layer_{i}_loss": 0.0 for i in range(len(LAYER_SIZES))},
    }

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()

        # Train for one epoch
        key, train_key = jax.random.split(key)
        state, metrics = train_epoch(
            state=state,
            images=train_images,
            labels=train_labels,
            key=train_key,
            network=network,
            threshold=THRESHOLD,
            batch_size=BATCH_SIZE,
            num_layers=len(LAYER_SIZES),
        )

        epoch_time = time.time() - epoch_start

        # Evaluate on test set
        accuracy = evaluate_accuracy(
            params=state.params,
            images=test_images,
            labels=test_labels,
            network=network,
            batch_size=EVAL_BATCH_SIZE,
        )

        # Track best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        # Print progress
        print(
            f"Epoch {epoch + 1:3d}/{NUM_EPOCHS} | "
            f"Loss: {metrics['total_loss']:.4f} | "
            f"Accuracy: {accuracy:.2%} | "
            f"Best: {best_accuracy:.2%} | "
            f"Time: {epoch_time:.1f}s"
        )

    # Final evaluation
    print("-" * 60)
    print("\nTraining complete!")
    print(f"Final test accuracy: {accuracy:.2%}")
    print(f"Best test accuracy: {best_accuracy:.2%}")

    # Print layer-wise losses
    print("\nFinal layer losses:")
    for i in range(len(LAYER_SIZES)):
        print(f"  Layer {i + 1}: {metrics[f'layer_{i}_loss']:.4f}")


if __name__ == "__main__":
    main()
