# Aurora Multi-GPU Training

This repository contains a refactored and clean implementation of Aurora weather forecasting model training with multi-GPU support using JAX data parallelism.

## Overview

The Aurora model is a state-of-the-art neural weather prediction model. This implementation provides:

- **Multi-GPU training** with data parallelism following the Aurora paper approach
- **Clean, modular code structure** with separated configuration and utilities
- **Configurable training parameters** via command line or configuration classes
- **Comprehensive logging** with Weights & Biases integration
- **Robust checkpoint management** for reliable training

## Code Structure

```
├── aurora_train.py          # Main training script
├── config.py               # Configuration classes and parameters
├── training_utils.py       # Multi-GPU utilities and helper functions
└── README.md              # This file
```

### Key Files

- **`aurora_train.py`**: Main training script with clean argument parsing and training loop
- **`config.py`**: Centralized configuration management with `TrainingConfig` dataclass
- **`training_utils.py`**: Multi-GPU setup, data sharding, and training utilities

## Requirements

- JAX with GPU support
- Flax
- Optax
- Orbax (for checkpointing)
- PyTorch (for data loading)
- Weights & Biases (optional, for logging)

## Installation

```bash
# Install JAX with GPU support
pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install other dependencies
pip install flax optax orbax-checkpoint torch wandb
```

## Usage

### Basic Usage

```bash
# Train with default configuration
python aurora_train.py \
    --dataset_path /path/to/your/dataset.zarr \
    --checkpoint_dir /path/to/pretrained/checkpointEncoder \
    --output_dir ./outputs

# Train with custom parameters
python aurora_train.py \
    --num_gpus 4 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --epochs 50 \
    --dataset_path /path/to/dataset.zarr \
    --checkpoint_dir /path/to/checkpointEncoder \
    --output_dir ./outputs
```

### Configuration

You can configure training by either:

1. **Command line arguments** (shown above)
2. **Modifying the `TrainingConfig` class** in `config.py`

Key configuration parameters:

```python
@dataclass
class TrainingConfig:
    # Model parameters
    batch_size: int = 2              # Total batch size across all GPUs
    learning_rate: float = 5e-5      # Peak learning rate
    epochs: int = 20                 # Number of training epochs
    rollout_steps: int = 1           # Number of rollout prediction steps
    
    # GPU configuration
    num_gpus: int = 2                # Number of GPUs to use
    
    # Data parameters
    dataset_path: str = "data/hres_dataset.zarr"
    num_workers: int = 4             # DataLoader workers
    
    # Checkpoint paths
    checkpoint_encoder: str = "checkpointEncoder/encoder"
    checkpoint_backbone: str = "checkpointEncoder/backbone"
    checkpoint_decoder: str = "checkpointEncoder/decoder"
    
    # Output directories
    output_dir: str = "outputs"
    
    # Training options
    average_rollout_loss: bool = True    # Average loss across rollout steps
    save_every_n_steps: int = 200       # Checkpoint saving frequency
    
    # Wandb configuration
    wandb_project: str = "aurora-rollout-training"
    use_wandb: bool = True
```

## Multi-GPU Training

The implementation uses **data parallelism** following the Aurora paper approach:

- **Model replication**: Full model is copied to each GPU
- **Data sharding**: Each GPU processes different data samples
- **Gradient synchronization**: Gradients are averaged across GPUs

### GPU Requirements

- The code automatically detects available GPUs
- Batch size must be divisible by the number of GPUs
- Each GPU will process `batch_size / num_gpus` samples

Example for 4 GPUs with batch size 8:
- Each GPU processes 2 samples per batch
- Model parameters are replicated on all 4 GPUs
- Gradients are synchronized across all GPUs

## Data Format

The training expects data in Zarr format with the structure used by the Aurora dataset. The data loader should provide:

- **Input batch**: Initial atmospheric and surface conditions
- **Target batches**: Sequence of target states for rollout training

## Checkpointing

The implementation provides robust checkpointing:

- **Loading**: Automatically loads pre-trained encoder, backbone, and decoder checkpointEncoder
- **Saving**: Saves checkpointEncoder every N steps (configurable)
- **Recovery**: Can resume training from saved checkpointEncoder

Checkpoint structure:
```
checkpointEncoder/
├── encoder/     # Encoder model weights
├── backbone/    # Backbone model weights
└── decoder/     # Decoder model weights
```

## Logging

Training metrics are logged to:

- **Console**: Progress updates and validation metrics
- **Weights & Biases**: Detailed training curves and hyperparameters (if enabled)

Logged metrics include:
- Training/validation MAE and RMSE
- Learning rate schedule
- Gradient norms
- Training progress

## Example Output

```
🚀 Starting Aurora multi-GPU training:
   • 4 GPUs with data parallelism
   • Batch size: 8 (2 per GPU)
   • Epochs: 50
   • Learning rate: 5e-05
   • Rollout steps: 1
   • Output directory: ./outputs

✓ Validated GPU setup: 4 GPUs available
✓ Batch size 8 will use 2 samples per GPU
✓ Successfully loaded all model checkpointEncoder

Epoch  1 — Validation MAE: 0.0234, RMSE: 0.0456
Epoch  2 — Validation MAE: 0.0218, RMSE: 0.0431
...
✓ Saved checkpoint at step 200
✓ Saved checkpoint at step 400
...
Epoch 50 — Validation MAE: 0.0156, RMSE: 0.0298

✅ Training completed successfully!
```

## Customization

The modular structure makes it easy to customize:

- **Model architecture**: Modify the model initialization in `aurora_train.py`
- **Loss functions**: Update loss computation in the training step
- **Data loading**: Modify `create_data_loaders()` in `training_utils.py`
- **Multi-GPU strategy**: Adjust sharding in `setup_mesh_and_sharding()`

## Troubleshooting

**GPU Detection Issues**:
```bash
# Check JAX can see your GPUs
python -c "import jax; print(jax.devices())"
```

**Memory Issues**:
- Reduce batch size
- Reduce number of workers in data loader
- Use gradient checkpointing if available

**Checkpoint Loading Errors**:
- Verify checkpoint paths exist
- Check checkpoint format compatibility
- Review the checkpoint loading function output

## Contributing

When contributing to this codebase:

1. Follow the modular structure (separate configs, utilities, and main logic)
2. Add proper type hints and docstrings
3. Keep paths configurable (avoid hardcoding)
4. Add validation for new configuration parameters
5. Update this README for any new features

## License

[Add your license information here]
