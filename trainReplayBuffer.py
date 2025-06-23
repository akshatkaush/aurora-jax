# trainReplayBuffer.py
import argparse
import os
from functools import partial
from typing import List  # Added for type hinting

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
from jax.tree_util import tree_leaves, tree_map
from replay_buffer import ReplayBuffer
from torch.utils.data import DataLoader

from aurora import AuroraSmall, Batch, Metadata  # Batch, Metadata might be needed for type hints
from aurora.IterableDataset import HresT0SequenceDataset
from aurora.rolloutTrain import rollout_scan_stop_gradients
from aurora.score import mae_loss_fn, weighted_rmse_batch


class LoRATrainState(train_state.TrainState):
    """Extended TrainState that separates base and LoRA parameters."""

    base_params: any = None

    def apply_gradients(self, *, grads, **kwargs):
        """Only apply gradients to LoRA parameters, keep base parameters frozen."""
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


def active_lr_schedule_fn(step: int, cfg) -> float:
    """
    Returns the current learning rate based on the training step.
    This is used for logging purposes to track the learning rate over time.
    """
    if cfg.freeze_base:
        # When using LoRA, return the LoRA learning rate
        return cfg.lora_learning_rate
    else:
        # When training all parameters, return the base learning rate
        return cfg.learning_rate


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
        for key, value in lora_dict.items():
            current_path = f"{path}.{key}" if path else key
            if key in saved_dict:
                if isinstance(value, dict) and isinstance(saved_dict[key], dict):
                    merged[key] = merge_recursive(saved_dict[key], value, current_path)
                else:
                    if hasattr(value, "shape") and hasattr(saved_dict[key], "shape"):
                        if value.shape == saved_dict[key].shape:
                            merged[key] = jnp.asarray(saved_dict[key], dtype=jnp.float32)
                            # print(f"Loaded {component_name}.{current_path} from checkpoint") # Keep original verbosity
                        else:
                            print(
                                f"Shape mismatch for {component_name}.{current_path}: "
                                f"saved {saved_dict[key].shape} vs expected {value.shape}, using initialized"
                            )
                            merged[key] = value
                    else:
                        if hasattr(saved_dict[key], "dtype"):
                            merged[key] = jnp.asarray(saved_dict[key], dtype=jnp.float32)
                        else:
                            merged[key] = saved_dict[key]
                        # print(f"Loaded {component_name}.{current_path} from checkpoint") # Keep original verbosity
            else:
                merged[key] = value
                # print(f"Using initialized {component_name}.{current_path} (new LoRA parameter)") # Keep original verbosity
        # Add parameters from saved_dict that are not in lora_dict (base parameters)
        for key, value in saved_dict.items():
            if key not in merged:
                current_path = f"{path}.{key}" if path else key
                merged[key] = (
                    jnp.asarray(value, dtype=jnp.float32) if hasattr(value, "dtype") else value
                )
                # print(f"Loaded base parameter {component_name}.{current_path} from checkpoint") # Keep original verbosity
        return merged

    return merge_recursive(saved_params, lora_params, "")


@partial(jax.jit, static_argnums=(4, 5, 6))  # apply_fn, rollout_steps, patch_size
def train_step_fn(
    state: LoRATrainState,
    sampled_inBatch: Batch,
    fresh_target_last_step_cropped: Batch,
    rng: jax.random.PRNGKey,
    apply_fn,
    rollout_steps: int,
    patch_size: int,
):
    """
    Performs a single training step including loss calculation, gradient update,
    and prediction for the replay buffer.
    """
    rng, loss_rng, pred_rng = jax.random.split(rng, 3)
    sampled_inBatch = sampled_inBatch.crop(patch_size)
    fresh_target_last_step_cropped = fresh_target_last_step_cropped.crop(patch_size)

    def loss_fn(params):
        # Predictions are based on the sampled_inBatch from the replay buffer
        preds, _, _ = rollout_scan_stop_gradients(
            apply_fn, sampled_inBatch, params, rollout_steps, True, loss_rng
        )
        last_pred = tree_map(lambda x: x[-1], preds)

        # Loss is calculated against the (cropped) last step of the fresh target batch
        mae = mae_loss_fn(
            last_pred,
            fresh_target_last_step_cropped,
            surf_weights,
            atmos_weights,
            gamma,
            alpha,
            beta,
        )
        rmse = weighted_rmse_batch(last_pred, fresh_target_last_step_cropped)
        return mae, rmse

    (mae, rmse), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    new_state = state.apply_gradients(grads=grads)

    # Generate predictions with the new_state.params for the replay buffer
    # These predictions are also based on the sampled_inBatch
    preds_for_buffer, _, _ = rollout_scan_stop_gradients(
        apply_fn, sampled_inBatch, new_state.params, rollout_steps, True, pred_rng
    )
    # We only need the last step of these predictions for the buffer
    pred_target_for_buffer = tree_map(lambda x: x[-1], preds_for_buffer)

    return new_state, mae, rmse, rng, pred_target_for_buffer


@partial(
    jax.jit, static_argnums=(4, 5, 6, 7)
)  # apply_fn, rollout_steps, average_rollout_loss, patch_size
def eval_step_fn(
    state: LoRATrainState,
    inBatch: Batch,
    target_batches: List[Batch],
    rng: jax.random.PRNGKey,
    apply_fn,
    rollout_steps: int,
    average_rollout_loss: bool,
    patch_size: int,
):
    """
    Performs a single evaluation step.
    target_batches is a list of Batch objects, one for each rollout step.
    """
    rng, step_rng = jax.random.split(rng)
    inBatch = inBatch.crop(patch_size)
    for target_batch in target_batches:
        target_batch = target_batch.crop(patch_size)

    preds, _, _ = rollout_scan_stop_gradients(
        apply_fn, inBatch, state.params, steps=rollout_steps, training=False, rng=step_rng
    )

    if average_rollout_loss:
        total_mae = 0.0
        total_rmse = 0.0
        # This loop should be JAX-traceable if rollout_steps is static.
        for step_idx in range(rollout_steps):
            step_pred = tree_map(lambda x: x[step_idx], preds)
            target_batch_step = target_batches[step_idx].crop(patch_size)
            total_mae += mae_loss_fn(
                step_pred, target_batch_step, surf_weights, atmos_weights, gamma, alpha, beta
            )
            total_rmse += weighted_rmse_batch(step_pred, target_batch_step)
        val_mae = total_mae / rollout_steps
        val_rmse = total_rmse / rollout_steps
    else:
        last_pred = tree_map(lambda x: x[-1], preds)
        target_batch_last = target_batches[-1].crop(patch_size)
        val_mae = mae_loss_fn(
            last_pred, target_batch_last, surf_weights, atmos_weights, gamma, alpha, beta
        )
        val_rmse = weighted_rmse_batch(last_pred, target_batch_last)

    return val_mae, val_rmse, rng


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--rollout_steps", type=int, default=1)
    parser.add_argument("--ckpt_encoder", type=str, default="/home1/a/akaush/aurora/checkpointEncoder")
    parser.add_argument(
        "--ckpt_backbone", type=str, default="/home1/a/akaush/aurora/checkpointBackbone"
    )
    parser.add_argument(
        "--ckpt_decoder", type=str, default="/home1/a/akaush/aurora/checkpointDecoder"
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
    parser.add_argument("--dataset_sampling_period", type=int, default=2)
    parser.add_argument("--replay_buffer_capacity", type=int, default=2)
    args = parser.parse_args()

    jax.config.update("jax_debug_nans", True)

    wandb.init(
        project="aurora-replay-buffer-refactored", config=vars(args)
    )  # Changed project name slightly
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
        print(f"Error loading checkpointEncoder: {e}")
        print("Using randomly initialized parameters")
        # full_params already contains the initialized LoRA parameters

    # Ensure all parameters are float32 and move to GPU
    def ensure_float32(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return jnp.asarray(x, dtype=jnp.float32)
        return x

    full_params = jax.tree_util.tree_map(ensure_float32, full_params)
    params = jax.device_put(full_params, device=jax.devices("gpu")[0])

    # Separate base and LoRA parameters for LoRA fine-tuning
    if cfg.freeze_base:
        base_params, lora_params = separate_lora_params(params)
        print("\nLoRA Fine-tuning Setup:")
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

    buffer = ReplayBuffer(capacity=cfg.replay_buffer_capacity)
    patch_size = model.patch_size
    print("Initializing replay buffer...")
    # The dataloader yields (input_sequence_batch, target_sequence_batch)
    # For buffer initialization, we add the first element of the input_sequence_batch
    for i, (inBatch_sequence, _) in enumerate(loader_train):
        if i >= cfg.replay_buffer_capacity:
            break
        buffer.add(inBatch_sequence.crop(patch_size))  # Ensure cropped before adding

    print("Training with replay buffer...")
    global_step = 0
    for epoch in range(cfg.epochs):
        # Training Loop
        for step_idx, (fresh_inBatch_t0, fresh_targets_sequence) in enumerate(loader_train):
            rng, step_rng = jax.random.split(rng)

            sampled_inBatch = buffer.sample()  # This is a Batch object

            # The loss is calculated against the last target in the fresh sequence, cropped
            fresh_target_last_step_cropped = fresh_targets_sequence[-1].crop(patch_size)

            state, mae, rmse, rng, pred_target_for_buffer = train_step_fn(
                state,
                sampled_inBatch,
                fresh_target_last_step_cropped,
                step_rng,
                state.apply_fn,
                cfg.rollout_steps,
                patch_size,
            )

            # Create a Batch object from the prediction for the replay buffer
            # sampled_inBatch is already cropped in train_step_fn, so we can use it directly
            cropped_sampled_inBatch = sampled_inBatch.crop(patch_size)
            pred_batch_for_buffer = Batch(
                surf_vars=pred_target_for_buffer.surf_vars,
                static_vars=cropped_sampled_inBatch.static_vars,  # Use cropped static_vars
                atmos_vars=pred_target_for_buffer.atmos_vars,
                metadata=cropped_sampled_inBatch.metadata,
            )
            buffer.add(pred_batch_for_buffer)

            # Periodically add a fresh batch (t0 input) to the buffer
            if global_step % cfg.dataset_sampling_period == 0:
                buffer.add(fresh_inBatch_t0.crop(patch_size))  # Ensure cropped before adding

            wandb.log(
                {
                    "train/mae": float(mae),
                    "train/rmse": float(rmse),
                    "train/lr": float(active_lr_schedule_fn(state.step, cfg)),
                    # LoRA specific logging from original code
                    "lora/trainable_params": sum(
                        x.size
                        for x in tree_leaves(
                            tree_map(
                                lambda p: p if "lora_" in str(p).lower() else None, state.params
                            )
                        )
                        if x is not None
                    ),
                    "lora/total_params": sum(x.size for x in tree_leaves(state.params)),
                    "lora/param_efficiency": (
                        sum(
                            x.size
                            for x in tree_leaves(
                                tree_map(
                                    lambda p: p if "lora_" in str(p).lower() else None, state.params
                                )
                            )
                            if x is not None
                        )
                        / sum(x.size for x in tree_leaves(state.params))
                    )
                    if sum(x.size for x in tree_leaves(state.params)) > 0
                    else 0,
                },
                step=global_step,
            )
            global_step += 1

        print(f"Epoch {epoch + 1} complete. Buffer size: {len(buffer)}. Global step: {global_step}")

        # Evaluation Loop
        val_maes, val_rmses = [], []
        for eval_idx, (inBatch_eval, target_batches_eval) in enumerate(loader_eval):
            rng, eval_rng = jax.random.split(rng)

            # Ensure data is on the correct device (JAX JIT usually handles this for inputs)
            # inBatch_eval = jax.device_put(inBatch_eval)
            # target_batches_eval = tree_map(jax.device_put, target_batches_eval)

            val_mae_step, val_rmse_step, rng = eval_step_fn(
                state,
                inBatch_eval,  # This is the t0 input Batch for the eval sequence
                target_batches_eval,  # This is List[Batch] of targets for the eval sequence
                eval_rng,
                state.apply_fn,
                cfg.rollout_steps,
                cfg.average_rollout_loss,
                patch_size,
            )
            val_maes.append(val_mae_step)
            val_rmses.append(val_rmse_step)

        if val_maes:  # Ensure there were evaluation steps
            val_mae_epoch = float(jnp.stack(val_maes).mean())
            val_rmse_epoch = float(jnp.stack(val_rmses).mean())
            wandb.log(
                {"val/mae": val_mae_epoch, "val/rmse": val_rmse_epoch, "epoch": epoch + 1},
                step=global_step,
            )
            print(f"Epoch {epoch + 1:2d} — val MAE {val_mae_epoch:.4f} RMSE {val_rmse_epoch:.4f}")
        else:
            print(f"Epoch {epoch + 1:2d} — No evaluation data processed.")

    wandb.finish()


if __name__ == "__main__":
    main()
