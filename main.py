#!/usr/bin/env python3
"""Main training script for Forward-Forward algorithm on MNIST.

This script trains a Forward-Forward network on MNIST and evaluates
its performance. The FF algorithm uses two forward passes (positive
and negative data) instead of traditional backpropagation.

Usage:
    python main.py                                    # Use defaults
    python main.py --learning_rate 0.01               # Custom learning rate
    python main.py --num_epochs 100 --batch_size 64   # Multiple options
    python main.py 512 512 512                        # Custom layer sizes
    python main.py --help                             # Show all options
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from fire import Fire
from tensorboardX import SummaryWriter

from forward_forward import (
    FFNetwork,
    create_train_state,
    evaluate_accuracy,
    load_mnist,
    predict_batch,
    train_epoch,
)

from loguru import logger


def main(
    learning_rate: float = 0.001,
    batch_size: int = 128,
    num_epochs: int = 60,
    threshold: float = 2.0,
    eval_batch_size: int = 1000,
    seed: int = 42,
    log_dir: str = "runs/forward_forward",
    *layer_sizes: int,
) -> None:
    """Main training function.

    Args:
        learning_rate: Learning rate for the optimizer.
        batch_size: Batch size for training.
        num_epochs: Number of training epochs.
        threshold: Threshold θ for the FF paper's goodness function.
        eval_batch_size: Batch size for evaluation.
        seed: Random seed for reproducibility.
        log_dir: Directory for TensorBoard logs.
        layer_sizes: Variable number of layer sizes for the network architecture.
            Defaults to (2000, 2000, 2000, 2000) if not provided.
    """
    # Use default layer sizes if none provided
    if not layer_sizes:
        layer_sizes = (2000, 2000, 2000, 2000)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=Path(log_dir))
    logger.info("=" * 60)
    logger.info("Forward-Forward Algorithm on MNIST")
    logger.info("=" * 60)

    # logger.info device info
    logger.info(f"\nJAX devices: {jax.devices()}")
    logger.info(f"Using: {jax.devices()[0].platform.upper()}")

    # Initialize random key
    key = jax.random.PRNGKey(seed)

    # Load MNIST data
    logger.info("\nLoading MNIST data...")
    train_images, train_labels = load_mnist(train=True)
    test_images, test_labels = load_mnist(train=False)

    logger.info(f"Training samples: {train_images.shape[0]}")
    logger.info(f"Test samples: {test_images.shape[0]}")
    logger.info(f"Image shape: {train_images.shape[1]}")

    # Create network
    logger.info("\nCreating network...")
    network = FFNetwork(layer_sizes=layer_sizes)
    logger.info(f"Layer sizes: {layer_sizes}")
    logger.info(
        f"Total parameters: {sum(p.size for p in jax.tree_util.tree_leaves(network.init(key, jnp.ones((1, 784)))))}"
    )

    # Initialize training state
    key, init_key = jax.random.split(key)
    state = create_train_state(
        network=network,
        key=init_key,
        input_shape=(784,),
        learning_rate=learning_rate,
    )

    # Training loop
    logger.info("\nStarting training...")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Threshold (θ): {threshold}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info("-" * 60)

    best_accuracy = 0.0
    accuracy = 0.0
    metrics = {
        "total_loss": 0.0,
        **{f"layer_{i}_loss": 0.0 for i in range(len(layer_sizes))},
    }

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Train for one epoch
        key, train_key = jax.random.split(key)
        state, metrics = train_epoch(
            state=state,
            images=train_images,
            labels=train_labels,
            key=train_key,
            network=network,
            threshold=threshold,
            batch_size=batch_size,
            num_layers=len(layer_sizes),
        )

        epoch_time = time.time() - epoch_start

        # Evaluate on test set
        accuracy = evaluate_accuracy(
            params=state.params,  # type: ignore
            images=test_images,
            labels=test_labels,
            network=network,
            batch_size=eval_batch_size,
        )

        # Track best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        # Log to TensorBoard
        writer.add_scalar("Loss/train", metrics["total_loss"], epoch)
        writer.add_scalar("Accuracy/test", accuracy, epoch)
        for i in range(len(layer_sizes)):
            writer.add_scalar(f"Loss/layer_{i}", metrics[f"layer_{i}_loss"], epoch)

        # logger.info progress
        logger.info(
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Loss: {metrics['total_loss']:.4f} | "
            f"Accuracy: {accuracy:.2%} | "
            f"Best: {best_accuracy:.2%} | "
            f"Time: {epoch_time:.1f}s"
        )

    # Log sample predictions
    sample_images = test_images[:25]
    sample_labels = test_labels[:25]
    pred_labels = predict_batch(
        params=state.params,  # type: ignore
        images=sample_images,
        network=network,
    )

    fig = create_prediction_grid(sample_images, sample_labels, pred_labels)
    writer.add_figure("Predictions/sample", fig, num_epochs)

    # Final evaluation
    logger.info("-" * 60)
    logger.info("\nTraining complete!")
    logger.info(f"Final test accuracy: {accuracy:.2%}")
    logger.info(f"Best test accuracy: {best_accuracy:.2%}")

    # logger.info layer-wise losses
    logger.info("\nFinal layer losses:")
    for i in range(len(layer_sizes)):
        logger.info(f"  Layer {i + 1}: {metrics[f'layer_{i}_loss']:.4f}")

    writer.close()


def create_prediction_grid(
    images: jnp.ndarray, labels: jnp.ndarray, predictions: jnp.ndarray
):
    """Create a grid visualization of sample predictions."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.imshow(np.array(images[i]).reshape(28, 28), cmap="gray")
        color = "green" if predictions[i] == labels[i] else "red"
        ax.set_title(f"Pred: {predictions[i]}, True: {labels[i]}", color=color)
        ax.axis("off")

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    Fire(main)
