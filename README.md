# forward-forward

A JAX implementation of [Forward-Forward algorithm](https://arxiv.org/abs/2212.13345) for training neural networks on MNIST without traditional backpropagation.

## Setup & Run

```bash
# requires python 3.12
uv sync
python main.py
```

Example with custom hyperparameters:

```bash
python main.py --learning_rate 0.01 --num_epochs 100 --batch_size 64
```
