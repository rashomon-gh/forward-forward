"""Forward-Forward Algorithm Implementation.

This package implements Geoffrey Hinton's Forward-Forward (FF) algorithm
using JAX, Flax, and Optax.

Main modules:
- data: Data preparation (label embedding, batch generation)
- layers: Custom FF layers with specialized normalization
- network: FFNetwork stacking multiple FFLayers
- loss: Goodness calculation and localized loss functions
- training: JIT-compiled training loop
- evaluation: Inference and accuracy evaluation

Example usage:
    ```python
    import jax
    import jax.numpy as jnp
    from forward_forward import (
        FFNetwork,
        create_train_state,
        train_epoch,
        evaluate_accuracy,
        load_mnist,
    )

    # Load data
    train_images, train_labels = load_mnist(train=True)
    test_images, test_labels = load_mnist(train=False)

    # Create network
    network = FFNetwork()
    key = jax.random.PRNGKey(42)
    state = create_train_state(network, key, (784,))

    # Train
    for epoch in range(num_epochs):
        key, subkey = jax.random.split(key)
        state, metrics = train_epoch(
            state, train_images, train_labels, subkey, network, threshold=2.0
        )
        accuracy = evaluate_accuracy(state.params, test_images, test_labels, network)
        logger.info(f"Epoch {epoch}: Accuracy = {accuracy:.2%}")
    ```
"""

from .data import (
    embed_label,
    embed_labels_batch,
    generate_batch,
    generate_wrong_labels,
    load_mnist,
    create_dataloader,
)
from .evaluation import (
    predict_single_image,
    predict_batch,
    evaluate_accuracy,
    evaluate_with_details,
)
from .layers import FFLayer, FFLayerWithStopGrad, custom_layer_norm
from .loss import (
    calculate_goodness,
    compute_probability,
    layer_loss_fn,
    layer_loss_fn_pure,
    compute_layer_goodness_sum,
)
from .network import FFNetwork, FFNetworkWithLayerNorm, create_ff_network
from .training import (
    create_train_state,
    train_epoch,
    train_layer_step,
    create_layer_train_step,
)

__all__ = [
    # Data module
    "embed_label",
    "embed_labels_batch",
    "generate_batch",
    "generate_wrong_labels",
    "load_mnist",
    "create_dataloader",
    # Layers module
    "FFLayer",
    "FFLayerWithStopGrad",
    "custom_layer_norm",
    # Network module
    "FFNetwork",
    "FFNetworkWithLayerNorm",
    "create_ff_network",
    # Loss module
    "calculate_goodness",
    "compute_probability",
    "layer_loss_fn",
    "layer_loss_fn_pure",
    "compute_layer_goodness_sum",
    # Training module
    "create_train_state",
    "train_epoch",
    "train_layer_step",
    "create_layer_train_step",
    # Evaluation module
    "predict_single_image",
    "predict_batch",
    "evaluate_accuracy",
    "evaluate_with_details",
]

__version__ = "0.1.0"
