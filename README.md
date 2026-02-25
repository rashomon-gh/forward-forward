# forward-forward

A JAX implementation of Geoffrey Hinton's Forward-Forward algorithm for training neural networks on MNIST without traditional backpropagation.

## Setup & Run

```bash
uv sync
python main.py
```

Example with custom hyperparameters:

```bash
python main.py --learning_rate 0.01 --num_epochs 100 --batch_size 64
```

## TensorBoard Visualisation

```bash
tensorboard --logdir runs/forward_forward
```

Logged metrics:
- Training loss and test accuracy curves
- Per-layer loss curves
- Sample prediction visualisations
