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
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from fire import Fire
from flax.core import unfreeze
from tensorboardX import SummaryWriter

from forward_forward import (
    FFNetwork,
    create_train_state,
    load_mnist,
    predict_batch,
    train_epoch,
)
from forward_forward.loss import layer_loss_fn_pure

from loguru import logger


def compute_validation_loss(
    params: Dict,
    images: jnp.ndarray,
    labels: jnp.ndarray,
    network: FFNetwork,
    threshold: float,
    batch_size: int = 1000,
    num_classes: int = 10,
    num_layers: int = 4,
) -> Dict[str, float]:
    """Compute validation loss across all layers.

    Args:
        params: Network parameters.
        images: Validation images.
        labels: Validation labels.
        network: FFNetwork instance.
        threshold: Threshold for loss computation.
        batch_size: Batch size for validation.
        num_classes: Number of classes.
        num_layers: Number of layers.

    Returns:
        Dictionary with total and per-layer validation losses.
    """
    num_samples = images.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    total_loss = 0.0
    layer_losses = [0.0] * num_layers

    params_dict = unfreeze(params)["params"]

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        batch_images = images[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        current_batch_size = batch_images.shape[0]

        # Generate positive samples (correct labels)
        one_hot_pos = jnp.zeros(
            (current_batch_size, num_classes), dtype=batch_images.dtype
        )
        one_hot_pos = one_hot_pos.at[jnp.arange(current_batch_size), batch_labels].set(
            1.0
        )
        positive = jnp.concatenate([one_hot_pos, batch_images[:, num_classes:]], axis=1)

        # Generate negative samples (wrong labels)
        key = jax.random.PRNGKey(batch_idx)
        wrong_labels = jax.random.randint(key, (current_batch_size,), 0, num_classes)
        wrong_labels = jnp.where(
            wrong_labels == batch_labels, (wrong_labels + 1) % num_classes, wrong_labels
        )
        one_hot_neg = jnp.zeros(
            (current_batch_size, num_classes), dtype=batch_images.dtype
        )
        one_hot_neg = one_hot_neg.at[jnp.arange(current_batch_size), wrong_labels].set(
            1.0
        )
        negative = jnp.concatenate([one_hot_neg, batch_images[:, num_classes:]], axis=1)

        # Compute layer-wise losses
        x_pos = positive
        x_neg = negative

        for layer_idx in range(num_layers):
            layer_key = f"Dense_{layer_idx}"
            W = params_dict[layer_key]["kernel"]
            b = params_dict[layer_key]["bias"]

            # Forward pass for positive and negative
            pos_acts = jnp.dot(x_pos, W) + b
            neg_acts = jnp.dot(x_neg, W) + b
            pos_acts = jnp.maximum(pos_acts, 0)
            neg_acts = jnp.maximum(neg_acts, 0)

            # Compute loss
            loss = float(layer_loss_fn_pure(pos_acts, neg_acts, threshold))
            layer_losses[layer_idx] += loss
            total_loss += loss

            # Normalize for next layer
            norm_pos = jnp.sqrt(jnp.sum(pos_acts**2, axis=-1, keepdims=True) + 1e-8)
            norm_neg = jnp.sqrt(jnp.sum(neg_acts**2, axis=-1, keepdims=True) + 1e-8)
            x_pos = pos_acts / norm_pos
            x_neg = neg_acts / norm_neg
            x_pos = jax.lax.stop_gradient(x_pos)
            x_neg = jax.lax.stop_gradient(x_neg)

    # Average losses
    metrics = {
        "total_val_loss": total_loss / (num_batches * num_layers),
        **{
            f"layer_{i}_val_loss": layer_losses[i] / num_batches
            for i in range(num_layers)
        },
    }
    return metrics


def log_weight_histograms(
    writer: SummaryWriter, params: Dict, epoch: int, num_layers: int
) -> None:
    """Log weight and bias histograms for all layers.

    Args:
        writer: TensorBoard writer.
        params: Network parameters.
        epoch: Current epoch number.
        num_layers: Number of layers.
    """
    params_dict = unfreeze(params)["params"]

    for i in range(num_layers):
        layer_key = f"Dense_{i}"
        W = np.array(params_dict[layer_key]["kernel"])
        b = np.array(params_dict[layer_key]["bias"])

        writer.add_histogram(f"Weights/layer_{i}", W, epoch)
        writer.add_histogram(f"Bias/layer_{i}", b, epoch)
        writer.add_scalar(f"Weights/mean_layer_{i}", float(np.mean(W)), epoch)
        writer.add_scalar(f"Weights/std_layer_{i}", float(np.std(W)), epoch)
        writer.add_scalar(f"Bias/mean_layer_{i}", float(np.mean(b)), epoch)
        writer.add_scalar(f"Bias/std_layer_{i}", float(np.std(b)), epoch)


def create_confusion_matrix(
    predictions: jnp.ndarray, labels: jnp.ndarray, num_classes: int = 10
) -> np.ndarray:
    """Create a confusion matrix.

    Args:
        predictions: Predicted labels.
        labels: True labels.
        num_classes: Number of classes.

    Returns:
        Confusion matrix as numpy array.
    """
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for pred, true in zip(predictions, labels):
        confusion[int(true), int(pred)] += 1
    return confusion


def log_confusion_matrix(
    writer: SummaryWriter,
    confusion: np.ndarray,
    epoch: int,
    class_names: Tuple[str, ...],
) -> None:
    """Log confusion matrix as text to TensorBoard.

    Args:
        writer: TensorBoard writer.
        confusion: Confusion matrix.
        epoch: Current epoch number.
        class_names: Names of the classes.
    """
    # Create formatted text representation
    header = "True\\Pred " + " ".join(f"{c:>5}" for c in class_names)
    rows = [header]
    for i, row in enumerate(confusion):
        row_str = f"{class_names[i]:>10} " + " ".join(f"{val:>5}" for val in row)
        rows.append(row_str)

    text_matrix = "\n".join(rows)
    writer.add_text("Confusion_Matrix", text_matrix, epoch)


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

    # Log hyperparameters
    logger.info("\nLogging hyperparameters to TensorBoard...")
    writer.add_text(
        "Hyperparameters",
        f"""
| Parameter | Value |
|-----------|-------|
| Learning Rate | {learning_rate} |
| Batch Size | {batch_size} |
| Num Epochs | {num_epochs} |
| Threshold (θ) | {threshold} |
| Layer Sizes | {layer_sizes} |
| Seed | {seed} |
    """.strip(),
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

    # Sample predictions interval (log every N epochs)
    log_pred_interval = max(1, num_epochs // 10)

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

        train_time = time.time() - epoch_start
        eval_start = time.time()

        # Evaluate on test set - get predictions for confusion matrix
        pred_labels = predict_batch(
            params=state.params,  # type: ignore
            images=test_images,
            network=network,
        )

        eval_time = time.time() - eval_start
        epoch_time = train_time + eval_time

        # Compute accuracy
        accuracy = float(jnp.mean(pred_labels == test_labels))

        # Compute validation loss
        val_metrics = compute_validation_loss(
            params=state.params,  # type: ignore
            images=test_images,
            labels=test_labels,
            network=network,
            threshold=threshold,
            batch_size=eval_batch_size,
            num_layers=len(layer_sizes),
        )

        # Track best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        # Log training metrics to TensorBoard
        writer.add_scalar("Loss/train", metrics["total_loss"], epoch)
        writer.add_scalar("Loss/val", val_metrics["total_val_loss"], epoch)
        writer.add_scalar("Accuracy/test", accuracy, epoch)
        writer.add_scalar("Accuracy/best", best_accuracy, epoch)
        writer.add_scalar("Time/train_epoch", train_time, epoch)
        writer.add_scalar("Time/eval_epoch", eval_time, epoch)
        writer.add_scalar("Time/total_epoch", epoch_time, epoch)
        writer.add_scalar("LearningRate/actual", learning_rate, epoch)

        # Log layer-wise losses (train and val)
        for i in range(len(layer_sizes)):
            writer.add_scalar(
                f"Loss/train_layer_{i}", metrics[f"layer_{i}_loss"], epoch
            )
            writer.add_scalar(
                f"Loss/val_layer_{i}", val_metrics[f"layer_{i}_val_loss"], epoch
            )

        # Log weight histograms (every 5 epochs to save disk space)
        if epoch % 5 == 0:
            log_weight_histograms(writer, state.params, epoch, len(layer_sizes))  # type: ignore

        # Log confusion matrix (every 5 epochs)
        if epoch % 5 == 0:
            confusion = create_confusion_matrix(pred_labels, test_labels)
            class_names = tuple(str(i) for i in range(10))
            log_confusion_matrix(writer, confusion, epoch, class_names)

            # Log per-class accuracy
            for i in range(10):
                mask = test_labels == i
                if jnp.sum(mask) > 0:
                    class_acc = float(jnp.mean(pred_labels[mask] == test_labels[mask]))
                    writer.add_scalar(f"Accuracy/class_{i}", class_acc, epoch)

        # Log sample predictions periodically
        if epoch % log_pred_interval == 0 or epoch == num_epochs - 1:
            sample_images = test_images[:25]
            sample_labels = test_labels[:25]
            sample_preds = predict_batch(
                params=state.params,  # type: ignore
                images=sample_images,
                network=network,
            )
            fig = create_prediction_grid(sample_images, sample_labels, sample_preds)
            writer.add_figure("Predictions/sample", fig, epoch)

        # Log progress
        logger.info(
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Train Loss: {metrics['total_loss']:.4f} | "
            f"Val Loss: {val_metrics['total_val_loss']:.4f} | "
            f"Accuracy: {accuracy:.2%} | "
            f"Best: {best_accuracy:.2%} | "
            f"Time: {epoch_time:.1f}s"
        )

    # Final evaluation
    logger.info("-" * 60)
    logger.info("\nTraining complete!")
    logger.info(f"Final test accuracy: {accuracy:.2%}")
    logger.info(f"Best test accuracy: {best_accuracy:.2%}")

    # Log final summary to TensorBoard
    writer.add_text(
        "Training_Summary",
        f"""
## Training Results

| Metric | Value |
|--------|-------|
| Final Accuracy | {accuracy:.2%} |
| Best Accuracy | {best_accuracy:.2%} |
| Final Train Loss | {metrics['total_loss']:.4f} |
| Final Val Loss | {val_metrics['total_val_loss']:.4f} |
| Total Epochs | {num_epochs} |

### Final Layer-wise Training Losses
{chr(10).join(f'- Layer {i+1}: {metrics[f"layer_{i}_loss"]:.4f}' for i in range(len(layer_sizes)))}

### Final Layer-wise Validation Losses
{chr(10).join(f'- Layer {i+1}: {val_metrics[f"layer_{i}_val_loss"]:.4f}' for i in range(len(layer_sizes)))}
    """.strip(),
    )

    # Log final layer-wise losses
    logger.info("\nFinal layer-wise training losses:")
    for i in range(len(layer_sizes)):
        logger.info(f"  Layer {i + 1}: {metrics[f'layer_{i}_loss']:.4f}")

    logger.info("\nFinal layer-wise validation losses:")
    for i in range(len(layer_sizes)):
        logger.info(f"  Layer {i + 1}: {val_metrics[f'layer_{i}_val_loss']:.4f}")

    logger.info(f"\nView TensorBoard logs at: {log_dir}")
    logger.info(f"Run: tensorboard --logdir {log_dir}")

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
