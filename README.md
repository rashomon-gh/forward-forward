# forward-forward

A JAX/Flax implementation of the [Forward-Forward algorithm](https://arxiv.org/abs/2212.13345) for training neural networks on MNIST without traditional backpropagation.



## Setup & Run

```bash
# Install dependencies
uv sync

# Run with default settings (60 epochs, 4 layers of 2000 units each)
python main.py

# Custom hyperparameters
python main.py --learning_rate 0.01 --num_epochs 100 --batch_size 64

# Custom architecture (3 layers of 512 units each)
python main.py 512 512 512
```

## Command Line Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning_rate` | 0.001 | Adam optimizer learning rate |
| `--batch_size` | 128 | Training batch size |
| `--num_epochs` | 60 | Number of training epochs |
| `--threshold` | 2.0 | Goodness threshold θ |
| `----eval_batch_size` | 1000 | Batch size for evaluation |
| `--seed` | 42 | Random seed |
| `--log_dir` | runs/forward_forward | TensorBoard log directory |
| `layer_sizes...` | 2000 2000 2000 2000 | Layer sizes (positional args) |

## TensorBoard Visualization

```bash
tensorboard --logdir runs/forward_forward
```

### Logged Metrics

**Scalars:**
- Training and validation loss curves
- Test accuracy (overall and per-class)
- Best accuracy tracking
- Per-layer loss curves (train and val)
- Epoch timing breakdown (train vs eval)
- Learning rate

**Histograms:**
- Weight distributions per layer
- Bias distributions per layer
- Weight/bias mean and std over time

**Visualizations:**
- Sample prediction grids (5×5 grid with correct/incorrect coloring)
- Confusion matrix (text format)
- Hyperparameters table
- Training summary

**Text:**
- Hyperparameters
- Confusion matrix
- Training summary

## Architecture

```
Input (784) → [Dense 2000 + ReLU + Norm] × 4 → Output
              ↑ Label embedded in first 10 pixels
```

- **Label embedding**: First 10 pixels replaced with one-hot label
- **Custom layer norm**: L2 normalization (no mean subtraction)
- **Layer isolation**: `stop_gradient` between layers
- **Inference**: Run 10 times (one per label), accumulate goodness excluding first layer

## Implementation Details

| Module | File | Description |
|--------|------|-------------|
| Data | [data.py](forward_forward/data.py) | Label embedding, positive/negative batch generation |
| Layers | [layers.py](forward_forward/layers.py) | Custom layer norm, FFLayer implementation |
| Network | [network.py](forward_forward/network.py) | FFNetwork stacking multiple layers |
| Loss | [loss.py](forward_forward/loss.py) | Goodness calculation, binary cross-entropy |
| Training | [training.py](forward_forward/training.py) | JIT-compiled layer-wise training steps |
| Evaluation | [evaluation.py](forward_forward/evaluation.py) | Label iteration inference, accuracy |

## Results

After 60 epochs with default settings:
- **Test Accuracy**: ~93-95%
- **Parameters**: 13.6M (4 layers × 2000 units)
- **Training time**: ~30s/epoch on GPU

