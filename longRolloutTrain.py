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

from aurora import AuroraSmall, Batch, Metadata
from aurora.IterableDataset import HresT0SequenceDataset
from aurora.rolloutTrain import rollout_scan
from aurora.score import mae_loss_fn, weighted_rmse_batch


class LoRATrainState(train_state.TrainState):
    """Extended TrainState that separates base and LoRA parameters."""

    base_params: any = None

    def apply_gradients(self, *, grads, **kwargs):
        """Only apply gradients to LoRA parameters, keep base parameters frozen."""
        # The grads should only contain gradients for LoRA parameters
        # base_params remain unchanged
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )


def create_lr_schedule(warmup_steps: int, peak_lr: float):
    warmup = optax.linear_schedule(init_value=0.0, end_value=peak_lr, transition_steps=warmup_steps)
    constant = optax.constant_schedule(peak_lr)
    return optax.join_schedules([warmup, constant], [warmup_steps])


def separate_lora_params(params):
    """
    Separate LoRA parameters from base parameters.
    Returns: (base_params, lora_params)
    """

    def is_lora_param(path, param):
        path_str = ".".join(path)
        return "lora_" in path_str.lower()

    def separate_recursive(params_dict, path=[]):
        base_dict = {}
        lora_dict = {}

        for key, value in params_dict.items():
            current_path = path + [key]

            if isinstance(value, dict):
                base_sub, lora_sub = separate_recursive(value, current_path)
                if base_sub:
                    base_dict[key] = base_sub
                if lora_sub:
                    lora_dict[key] = lora_sub
            else:
                if is_lora_param(current_path, value):
                    lora_dict[key] = value
                else:
                    base_dict[key] = value

        return base_dict, lora_dict

    return separate_recursive(params)


def create_lora_partition(params):
    """
    Create partition for LoRA parameters using JAX tree utilities.
    Returns a tree with the same structure as params but with 'lora' or 'frozen' labels.
    """

    def label_param(path_elements, param):
        # Convert path elements to string and check if it contains 'lora_'
        path_str = ".".join(str(p) for p in path_elements)
        return "lora" if "lora_" in path_str.lower() else "frozen"

    def create_partition_recursive(params_dict, path=[]):
        partition_dict = {}

        for key, value in params_dict.items():
            current_path = path + [key]

            if isinstance(value, dict):
                partition_dict[key] = create_partition_recursive(value, current_path)
            else:
                partition_dict[key] = label_param(current_path, value)

        return partition_dict

    return create_partition_recursive(params)


def merge_params_with_lora_structure(saved_params, lora_params, component_name):
    """
    Merge saved parameters (without LoRA) with LoRA parameter structure.
    LoRA parameters that don't exist in saved_params will keep their initialized values.
    """

    def merge_recursive(saved_dict, lora_dict, path=""):
        merged = {}

        # First, copy all LoRA parameters (this includes new LoRA-specific params)
        for key, value in lora_dict.items():
            current_path = f"{path}.{key}" if path else key

            if key in saved_dict:
                if isinstance(value, dict) and isinstance(saved_dict[key], dict):
                    # Recursively merge nested dictionaries
                    merged[key] = merge_recursive(saved_dict[key], value, current_path)
                else:
                    # Use saved parameter if available and shapes match
                    if hasattr(value, "shape") and hasattr(saved_dict[key], "shape"):
                        if value.shape == saved_dict[key].shape:
                            # Ensure dtype consistency - convert to float32
                            merged[key] = jnp.asarray(saved_dict[key], dtype=jnp.float32)
                            print(f"Loaded {component_name}.{current_path} from checkpoint")
                        else:
                            print(
                                f"Shape mismatch for {component_name}.{current_path}: "
                                f"saved {saved_dict[key].shape} vs expected {value.shape}, using initialized"
                            )
                            merged[key] = value
                    else:
                        # Ensure dtype consistency for non-array parameters too
                        if hasattr(saved_dict[key], "dtype"):
                            merged[key] = jnp.asarray(saved_dict[key], dtype=jnp.float32)
                        else:
                            merged[key] = saved_dict[key]
                        print(f"Loaded {component_name}.{current_path} from checkpoint")
            else:
                # Keep LoRA initialized parameter (new LoRA components)
                merged[key] = value
                print(f"Using initialized {component_name}.{current_path} (new LoRA parameter)")

        return merged

    return merge_recursive(saved_params, lora_params, "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--rollout_steps", type=int, default=1)
    parser.add_argument("--ckpt_encoder", type=str, default="/home1/a/akaush/aurora/checkpoints")
    parser.add_argument(
        "--ckpt_backbone", type=str, default="/home1/a/akaush/aurora/checkpointsTillBackbone"
    )
    parser.add_argument(
        "--ckpt_decoder", type=str, default="/home1/a/akaush/aurora/checkpointsTillDecoder"
    )
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA parameters (typically higher than base)",
    )
    parser.add_argument(
        "--freeze_base",
        action="store_true",
        default=True,
        help="Freeze base model parameters and only train LoRA",
    )
    parser.add_argument(
        "--average_rollout_loss",
        action="store_true",
        help="Average loss across all rollout steps instead of using only the last step",
        default=True,
    )
    args = parser.parse_args()

    jax.config.update("jax_debug_nans", True)

    wandb.init(project="aurora-rollout2-long-finetuning", config=vars(args))
    cfg = wandb.config

    # create directories with new names
    os.makedirs("../tempData/singleStepEncoder", exist_ok=True)
    os.makedirs("../tempData/singleStepBackbone", exist_ok=True)
    os.makedirs("../tempData/singleStepDecoder", exist_ok=True)

    ZARR = "/home1/a/akaush/aurora/hresDataset/hres_t0_2021-2022mid.zarr"
    ds_train = HresT0SequenceDataset(ZARR, mode="train", steps=cfg.rollout_steps)
    loader_train = DataLoader(ds_train, batch_size=None, num_workers=0)

    ds_eval = HresT0SequenceDataset(ZARR, mode="eval", steps=cfg.rollout_steps)
    loader_eval = DataLoader(ds_eval, batch_size=None, num_workers=0)
    rng = jax.random.PRNGKey(0)

    model = AuroraSmall(use_lora=True)

    key = jax.random.key(0)
    sample_batch = Batch(
        surf_vars={
            k: jax.random.normal(jax.random.split(key, 4)[i], (1, 2, 720, 1440)).astype(jnp.float32)
            for i, k in enumerate(("2t", "10u", "10v", "msl"))
        },
        static_vars={
            k: jax.random.normal(jax.random.split(key, 3)[i], (720, 1440)).astype(jnp.float32)
            for i, k in enumerate(("z", "slt", "lsm"))
        },
        atmos_vars={
            k: jax.random.normal(jax.random.split(key, 5)[i], (1, 2, 13, 720, 1440)).astype(
                jnp.float32
            )
            for i, k in enumerate(("t", "u", "v", "q", "z"))
        },
        metadata=Metadata(
            lat=jnp.linspace(90, -90, 720).astype(jnp.float32),
            lon=jnp.linspace(0, 360, 1440 + 1)[:-1].astype(jnp.float32),
            time=(jnp.array((1672570800), dtype=jnp.int64),),
            atmos_levels=(1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50),
        ),
    )
    init_vars = model.init(key, sample_batch, training=False, rng=key)
    full_params = init_vars["params"]

    # Initialize checkpointer
    ckpt = ocp.StandardCheckpointer()

    # Load saved parameters and merge with LoRA structure
    try:
        print("Loading saved parameters...")

        # Load saved parameters (without LoRA)
        saved_encoder = ckpt.restore(cfg.ckpt_encoder)
        saved_backbone = ckpt.restore(cfg.ckpt_backbone)
        saved_decoder = ckpt.restore(cfg.ckpt_decoder)

        # Merge with LoRA structure
        full_params["encoder"] = merge_params_with_lora_structure(
            saved_encoder["encoder"], full_params["encoder"], "encoder"
        )
        full_params["backbone"] = merge_params_with_lora_structure(
            saved_backbone["backbone"], full_params["backbone"], "backbone"
        )
        full_params["decoder"] = merge_params_with_lora_structure(
            saved_decoder["decoder"], full_params["decoder"], "decoder"
        )

        print("Successfully loaded and merged parameters with LoRA structure")

    except Exception as e:
        print(f"Error loading checkpoints: {e}")
        print("Using randomly initialized parameters")
        # full_params already contains the initialized LoRA parameters

    # Ensure all parameters are float32 and move to GPU
    def ensure_float32(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return jnp.asarray(x, dtype=jnp.float32)
        return x

    # Fix deprecated jax.tree_map -> jax.tree_util.tree_map
    full_params = jax.tree_util.tree_map(ensure_float32, full_params)
    params = jax.device_put(full_params, device=jax.devices("gpu")[0])

    # Separate base and LoRA parameters for LoRA fine-tuning
    if cfg.freeze_base:
        base_params, lora_params = separate_lora_params(params)
        print("\nLoRA Fine-tuning Setup:")
        # Fix deprecated jax.tree_leaves -> jax.tree_util.tree_leaves
        print(
            f"Base parameters (frozen): {sum(x.size for x in jax.tree_util.tree_leaves(base_params)):,}"
        )
        print(
            f"LoRA parameters (trainable): {sum(x.size for x in jax.tree_util.tree_leaves(lora_params)):,}"
        )

        # Create optimizer only for LoRA parameters
        lr_schedule = create_lr_schedule(cfg.warmup_steps, cfg.lora_learning_rate)
        tx = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)

        # Create partition for frozen/trainable parameters using our custom function
        partition = create_lora_partition(params)

        # Only optimize LoRA parameters using optax.multi_transform
        tx = optax.multi_transform({"lora": tx, "frozen": optax.set_to_zero()}, partition)

        state = LoRATrainState.create(
            apply_fn=model.apply, params=params, base_params=base_params, tx=tx
        )
    else:
        # Standard training of all parameters
        lr_schedule = create_lr_schedule(cfg.warmup_steps, cfg.learning_rate)
        tx = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
        state = LoRATrainState.create(apply_fn=model.apply, params=params, tx=tx)
        print("Training all parameters (no LoRA separation)")

    @partial(jax.jit, static_argnums=(4, 5))
    def train_step(
        state, inBatch: Batch, target_batches: List[Batch], rng, steps: int, average_loss: bool
    ):
        """
        Args:
            target_batches: List/sequence of target batches for each rollout step
            average_loss: Whether to average loss across all steps or use only the last
        """
        rng, roll_rng = jax.random.split(rng, 2)
        inBatch = inBatch.crop(model.patch_size)

        def loss_fn(params):
            preds, _, _ = rollout_scan(state.apply_fn, inBatch, params, steps, True, roll_rng)

            if average_loss:
                # Average MAE loss across all rollout steps (as mentioned in paper)
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

        # todo, remove later, only for testing, compute gradient norm
        if cfg.freeze_base:
            # Only compute gradient norm for LoRA parameters
            # Get LoRA gradients by filtering based on partition
            def get_lora_grads(grads, partition):
                def filter_lora(g, p):
                    return g if p == "lora" else jnp.zeros_like(g)

                return jax.tree_util.tree_map(filter_lora, grads, partition)

            lora_grads = get_lora_grads(grads, partition)
            g2 = sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(lora_grads)])
        else:
            g2 = sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grads)])
        grad_norm = jnp.sqrt(g2)

        # Get learning rate
        if cfg.freeze_base:
            lr = lr_schedule(state.step) if callable(lr_schedule) else cfg.lora_learning_rate
        else:
            lr = lr_schedule(state.step)

        return new_state, mae, rmse, rng, grad_norm, lr

    @partial(jax.jit, static_argnums=(4, 5))
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
                # target_batch = target_batch.crop(model.patch_size)
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
    for epoch in range(1, cfg.epochs + 1):
        train_losses = []
        for inBatch, target_batches in loader_train:
            rng, step_rng = jax.random.split(rng, 2)
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

                # Add LoRA-specific metrics
                if cfg.freeze_base:
                    # Count trainable vs frozen parameters by filtering based on partition
                    def count_lora_params(params, partition):
                        def sum_if_lora(p, part):
                            return p.size if part == "lora" else 0

                        sizes = jax.tree_util.tree_map(sum_if_lora, params, partition)
                        return sum(jax.tree_util.tree_leaves(sizes))

                    trainable_params = count_lora_params(state.params, partition)
                    total_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
                    avg.update(
                        {
                            "lora/trainable_params": trainable_params,
                            "lora/total_params": total_params,
                            "lora/param_efficiency": trainable_params / total_params,
                        }
                    )
                wandb.log(avg, step=global_step)
                train_losses.clear()

            if global_step % 100 == 0:
                print(global_step)
            if global_step % 200 == 0:
                # save using the new names
                for orig, new in [
                    ("encoder", "singleStepEncoderLora"),
                    ("backbone", "singleStepBackboneLora"),
                    ("decoder", "singleStepDecoderLora"),
                ]:
                    ckpt.save(f"/home1/a/akaush/tempData/{new}", state.params[orig], force=True)
                print(f"Saved checkpoint at step {global_step}")

        # Validation
        val_maes = []
        val_rmses = []
        for inBatch, target_batches in loader_eval:
            rng, step_rng = jax.random.split(rng, 2)
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
            f" — val MAE {val_mae:.4f} RMSE {val_rmse:.4f}"
        )

    wandb.finish()


if __name__ == "__main__":
    main()
