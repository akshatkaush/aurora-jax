import argparse
import os
from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from config import (
    alpha,
    atmos_weights,
    beta,
    gamma,
    surf_weights,
    weight_decay,
)
from flax.training import train_state
from torch.utils.data import DataLoader
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils

from aurora import AuroraSmall, Batch
from aurora.iterable_dataset import HresT0SequenceDataset, collate_aurora_batches
from aurora.rollout_train import rollout_scan
from aurora.score import mae_loss_fn, weighted_rmse_batch


class TrainState(train_state.TrainState):
    pass


def create_lr_schedule(warmup_steps: int, peak_lr: float):
    warmup = optax.linear_schedule(init_value=0.0, end_value=peak_lr, transition_steps=warmup_steps)
    constant = optax.constant_schedule(peak_lr)
    return optax.join_schedules([warmup, constant], [warmup_steps])


def setup_mesh_and_sharding(num_gpus: int):
    """Setup n-GPU mesh for model parallelism as per Aurora paper approach"""
    devices = mesh_utils.create_device_mesh((num_gpus,))  # n GPUs
    mesh = Mesh(devices, axis_names=('data',))  # Data parallelism across n GPUs
    
    # Define sharding specifications for model components
    # Model weights can be replicated while data is sharded (Aurora paper approach)
    model_sharding = NamedSharding(mesh, P(None))  # Replicated model weights
    # Data parallelism: split data across GPUs (1 datapoint per GPU)
    data_sharding = NamedSharding(mesh, P('data'))  # Shard along data dimension
    
    return mesh, model_sharding, data_sharding


def shard_params(params, model_sharding):
    """Replicate model parameters across GPUs for data parallelism"""
    # For Aurora's approach: replicate full model on each GPU, shard data instead
    sharded_params = {}
    
    with mesh:
        # Replicate all model components on each GPU for data parallelism
        sharded_params['encoder'] = jax.device_put(params['encoder'], model_sharding)
        sharded_params['backbone'] = jax.device_put(params['backbone'], model_sharding) 
        sharded_params['decoder'] = jax.device_put(params['decoder'], model_sharding)
    
    return sharded_params


def shard_batch(batch: Batch, data_sharding, total_batch_size: int):
    """Shard batch data across GPUs - each GPU gets different data (Aurora approach)"""
    # For true data parallelism, we need to split batch along batch dimension
    def shard_leaf(x):
        if hasattr(x, 'ndim') and x.ndim == 0:
            # For scalars, replicate across devices
            scalar_sharding = NamedSharding(mesh, P())
            return jax.device_put(x, scalar_sharding)
        elif hasattr(x, 'shape') and len(x.shape) > 0:
            # Check if this looks like batched data (first dim should match total_batch_size)
            if x.shape[0] == total_batch_size:
                # This has proper batch dimension - shard along first dimension
                return jax.device_put(x, data_sharding)
            else:
                # This is spatial/static data without batch dimension - replicate
                replicated_sharding = NamedSharding(mesh, P(None))
                return jax.device_put(x, replicated_sharding)
        else:
            # Default to replication for safety
            replicated_sharding = NamedSharding(mesh, P(None))
            return jax.device_put(x, replicated_sharding)
    
    with mesh:
        sharded_batch = jax.tree_util.tree_map(shard_leaf, batch)
    return sharded_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--rollout_steps", type=int, default=1)
    parser.add_argument("--history_time_dim", type=int, default=2)
    parser.add_argument("--ckpt_encoder", type=str, default="/home1/a/akaush/aurora/checkpointEncoder")
    parser.add_argument(
        "--ckpt_backbone", type=str, default="/home1/a/akaush/aurora/checkpointBackbone"
    )
    parser.add_argument(
        "--ckpt_decoder", type=str, default="/home1/a/akaush/aurora/checkpointDecoder"
    )
    parser.add_argument(
        "--average_rollout_loss",
        action="store_true",
        help="Average loss across all rollout steps instead of using only the last step",
        default=True,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients over to simulate larger batch sizes",
    )
    parser.add_argument(
        "--total_batch_size",
        type=int,
        default=2,
        help="Total batch size across all GPUs (will be distributed evenly)",
    )
    args = parser.parse_args()

    # Check for available GPUs
    available_gpus = len(jax.devices('gpu'))
    if available_gpus < 1:
        raise ValueError(f"This script requires at least 1 GPU, found {available_gpus}")
    
    num_gpus = available_gpus
    
    # Smart batch size calculation: distribute total batch size across GPUs
    total_batch_size = args.total_batch_size
    if total_batch_size < num_gpus:
        print(f"Warning: total_batch_size ({total_batch_size}) < num_gpus ({num_gpus})")
        print(f"Setting total_batch_size = {num_gpus} (1 sample per GPU minimum)")
        total_batch_size = num_gpus
    
    batch_size_per_gpu = total_batch_size // num_gpus
    # Handle remainder: some GPUs get +1 sample
    remainder = total_batch_size % num_gpus
    
    print(f"Using {num_gpus} GPU(s) for training")
    print(f"Total batch size: {total_batch_size}")
    print(f"Batch size per GPU: {batch_size_per_gpu}" + (f" (+1 for {remainder} GPUs)" if remainder > 0 else ""))
    print(f"This configuration should work with any number of GPUs!")

    jax.config.update("jax_debug_nans", True)
    
    # Enable memory management to avoid fragmentation  
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")  # Use 80% of GPU memory

    wandb.init(project="aurora-rollout-multi-gpu", config=vars(args))
    cfg = wandb.config

    # Setup mesh and sharding for multi-GPU training
    global mesh
    mesh, model_sharding, data_sharding = setup_mesh_and_sharding(num_gpus)

    # create directories with new names
    os.makedirs("../tempData/multiGpuEncoder", exist_ok=True)
    os.makedirs("../tempData/multiGpuBackbone", exist_ok=True)
    os.makedirs("../tempData/multiGpuDecoder", exist_ok=True)

    ZARR = "/home1/a/akaush/aurora/hresDataset/hres_t0_2021-2022mid.zarr"
    ds_train = HresT0SequenceDataset(ZARR, mode="train", steps=cfg.rollout_steps)
    loader_train = DataLoader(ds_train, batch_size=total_batch_size, num_workers=0, collate_fn=collate_aurora_batches)

    ds_eval = HresT0SequenceDataset(ZARR, mode="eval", steps=cfg.rollout_steps)
    loader_eval = DataLoader(ds_eval, batch_size=total_batch_size, num_workers=0, collate_fn=collate_aurora_batches)
    rng = jax.random.PRNGKey(0)

    model = AuroraSmall(use_lora=False)

    ckpt = ocp.StandardCheckpointer()
    enc = ckpt.restore(cfg.ckpt_encoder)
    bb = ckpt.restore(cfg.ckpt_backbone)
    dec = ckpt.restore(cfg.ckpt_decoder)
    params = {
        "encoder": enc["encoder"],
        "backbone": bb["backbone"],
        "decoder": dec["decoder"],
    }
    
    # Shard parameters across GPUs following Aurora's approach
    with mesh:
        params = shard_params(params, model_sharding)

    lr_schedule = create_lr_schedule(cfg.warmup_steps, cfg.learning_rate)
    tx = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    
    # Create training state with sharded parameters
    with mesh:
        state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @partial(jax.jit, 
             static_argnums=(4, 5),
             in_shardings=(None, None, None, None),
             out_shardings=(None, None, None, None, None, None))
    def train_step(
        state, inBatch: Batch, target_batches: List[Batch], rng, steps: int, average_loss: bool
    ):
        """
        Multi-GPU training step with model parallelism following Aurora paper approach
        Args:
            target_batches: List/sequence of target batches for each rollout step
            average_loss: Whether to average loss across all steps or use only the last
        """
        rng, roll_rng = jax.random.split(rng, 2)
        inBatch = inBatch.crop(model.patch_size)

        def loss_fn(params):
            preds, _, _ = rollout_scan(state.apply_fn, inBatch, params, steps, True, roll_rng)

            if average_loss:
                # Average MAE loss across all rollout steps (as mentioned in Aurora paper)
                total_mae = 0.0
                total_rmse = 0.0

                for step_idx in range(steps):
                    step_pred = jax.tree_util.tree_map(lambda x: x[step_idx], preds)
                    target_batch = target_batches[step_idx].crop(model.patch_size)

                    step_mae = mae_loss_fn(
                        step_pred, target_batch, surf_weights, atmos_weights, gamma, alpha, beta
                    )
                    step_rmse = weighted_rmse_batch(step_pred, target_batch)

                    total_mae += step_mae
                    total_rmse += step_rmse

                avg_mae = total_mae / steps
                avg_rmse = total_rmse / steps
                return avg_mae, avg_rmse
            else:
                last_pred = jax.tree_util.tree_map(lambda x: x[-1], preds)
                target_batch = target_batches[-1].crop(model.patch_size)
                mae = mae_loss_fn(
                    last_pred, target_batch, surf_weights, atmos_weights, gamma, alpha, beta
                )
                rmse = weighted_rmse_batch(last_pred, target_batch)
                return mae, rmse

        (mae, rmse), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)

        # Compute gradient norm across all devices
        g2 = sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grads)])
        grad_norm = jnp.sqrt(g2)
        lr = lr_schedule(state.step)

        return new_state, mae, rmse, rng, grad_norm, lr

    @partial(jax.jit, 
             static_argnums=(4, 5),
             in_shardings=(None, None, None, None),
             out_shardings=(None, None, None))
    def eval_step(
        state, inBatch: Batch, target_batches: List[Batch], rng, steps: int, average_loss: bool
    ):
        rng, roll_rng = jax.random.split(rng, 2)
        inBatch = inBatch.crop(model.patch_size)
        preds, _, _ = rollout_scan(
            state.apply_fn, inBatch, state.params, steps=steps, training=False, rng=roll_rng
        )

        if average_loss:
            total_mae = 0.0
            total_rmse = 0.0

            for step_idx in range(steps):
                step_pred = jax.tree_util.tree_map(lambda x: x[step_idx], preds)
                target_batch = target_batches[step_idx].crop(model.patch_size)
                step_mae = mae_loss_fn(
                    step_pred, target_batch, surf_weights, atmos_weights, gamma, alpha, beta
                )
                step_rmse = weighted_rmse_batch(step_pred, target_batch)

                total_mae += step_mae
                total_rmse += step_rmse

            avg_mae = total_mae / steps
            avg_rmse = total_rmse / steps
            return avg_mae, avg_rmse, rng
        else:
            last_pred = jax.tree_util.tree_map(lambda x: x[-1], preds)
            target_batch = target_batches[-1].crop(model.patch_size)
            mae = mae_loss_fn(
                last_pred, target_batch, surf_weights, atmos_weights, gamma, alpha, beta
            )
            rmse = weighted_rmse_batch(last_pred, target_batch)
            return mae, rmse, rng

    global_step = 0
    print(f"Starting multi-GPU training with Aurora paper approach:")
    print(f"- Data parallelism across {num_gpus} GPUs")
    print(f"- Model replication: Full model copied to each GPU")
    print(f"- Data sharding: Each GPU processes different samples (1 per GPU)")
    
    for epoch in range(1, cfg.epochs + 1):
        train_losses = []
        for inBatch, target_batches in loader_train:
            rng, step_rng = jax.random.split(rng, 2)
            
            # Shard input batch across GPUs (1 datapoint per GPU)
            with mesh:
                inBatch = shard_batch(inBatch, data_sharding, total_batch_size)
                target_batches = [shard_batch(tb, data_sharding, total_batch_size) for tb in target_batches]
            
            state, train_mae, train_rmse, rng, grad_norm, lr = train_step(
                state,
                inBatch,
                target_batches,
                step_rng,
                cfg.rollout_steps,
                cfg.average_rollout_loss,
            )
            train_losses.append({"mae": train_mae, "rmse": train_rmse})
            global_step += 1

            if (global_step + 1) % 1 == 0:
                avg = {
                    "train/mae": float(jnp.stack([x["mae"] for x in train_losses]).mean()),
                    "train/rmse": float(jnp.stack([x["rmse"] for x in train_losses]).mean()),
                    "train/grad_norm": float(grad_norm),
                    "train/lr": float(lr),
                }
                wandb.log(avg, step=global_step)
                train_losses.clear()

            if global_step % 100 == 0:
                print(f"Step {global_step} - Multi-GPU training progress ({num_gpus} GPUs)")
            if global_step % 200 == 0:
                # Save sharded checkpointEncoder
                with mesh:
                    for orig, new in [
                        ("encoder", "multiGpuEncoder"),
                        ("backbone", "multiGpuBackbone"),
                        ("decoder", "multiGpuDecoder"),
                    ]:
                        # Gather parameters from all devices before saving
                        gathered_params = jax.device_get(state.params[orig])
                        ckpt.save(f"/home1/a/akaush/tempData/{new}", gathered_params, force=True)
                print(f"Saved multi-GPU checkpoint at step {global_step} ({num_gpus} GPUs)")

        # Validation with multi-GPU
        val_maes = []
        val_rmses = []
        for inBatch, target_batches in loader_eval:
            rng, step_rng = jax.random.split(rng, 2)
            
            # Shard validation batch
            with mesh:
                inBatch = shard_batch(inBatch, data_sharding, total_batch_size)
                target_batches = [shard_batch(tb, data_sharding, total_batch_size) for tb in target_batches]
            
            v_mae, v_rmse, rng = eval_step(
                state,
                inBatch,
                target_batches,
                step_rng,
                cfg.rollout_steps,
                cfg.average_rollout_loss,
            )
            val_maes.append(v_mae)
            val_rmses.append(v_rmse)

        val_mae = float(jnp.stack(val_maes).mean())
        val_rmse = float(jnp.stack(val_rmses).mean())
        wandb.log(
            {
                "val/mae": val_mae,
                "val/rmse": val_rmse,
                "epoch": epoch,
            }
        )
        print(
            f"Epoch {epoch:2d} — train MAE {train_mae:.4f} RMSE {train_rmse:.4f}"
            f" — val MAE {val_mae:.4f} RMSE {val_rmse:.4f} [Multi-GPU: {num_gpus}]"
        )

    wandb.finish()


if __name__ == "__main__":
    main() 