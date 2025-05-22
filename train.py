import argparse
import os
from functools import partial

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

from score import mae_loss_fn, weighted_rmse_batch

from aurora import AuroraSmall, Batch
from aurora.IterableDataset import HresT0SequenceDataset
from aurora.rolloutTrain import rollout_scan


class TrainState(train_state.TrainState):
    pass


def create_lr_schedule(warmup_steps: int, peak_lr: float):
    warmup = optax.linear_schedule(init_value=0.0, end_value=peak_lr, transition_steps=warmup_steps)
    constant = optax.constant_schedule(peak_lr)
    return optax.join_schedules([warmup, constant], [warmup_steps])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--rollout_steps", type=int, default=2)
    parser.add_argument("--history_time_dim", type=int, default=2)
    parser.add_argument("--ckpt_encoder", type=str, default="/home1/a/akaush/aurora/checkpoints")
    parser.add_argument(
        "--ckpt_backbone", type=str, default="/home1/a/akaush/aurora/checkpointsTillBackbone"
    )
    parser.add_argument(
        "--ckpt_decoder", type=str, default="/home1/a/akaush/aurora/checkpointsTillDecoder"
    )
    args = parser.parse_args()

    jax.config.update("jax_debug_nans", True)

    wandb.init(project="aurora-rollout2", config=vars(args))
    cfg = wandb.config

    os.makedirs("../tempData/encoder", exist_ok=True)
    os.makedirs("../tempData/backbone", exist_ok=True)
    os.makedirs("../tempData/decoder", exist_ok=True)

    ZARR = "/home1/a/akaush/aurora/hresDataset/hres_t0_2021-2022mid.zarr"
    ds_train = HresT0SequenceDataset(ZARR, mode="train")
    loader_train = DataLoader(ds_train, batch_size=None, num_workers=0)

    ds_eval = HresT0SequenceDataset(ZARR, mode="eval")
    loader_eval = DataLoader(ds_eval, batch_size=None, num_workers=0)
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
    params = jax.device_put(params, device=jax.devices("gpu")[0])

    lr_schedule = create_lr_schedule(cfg.warmup_steps, cfg.learning_rate)
    tx = optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @partial(jax.jit, static_argnums=(4,))
    def train_step(state, inBatch: Batch, outBatch: Batch, rng, steps: int):
        rng, roll_rng = jax.random.split(rng, 2)
        outBatch = outBatch.crop(model.patch_size)

        def loss_fn(params):
            preds, _, _ = rollout_scan(
                state.apply_fn, model, inBatch, params, steps=steps, training=True, rng=roll_rng
            )
            last_pred = jax.tree_util.tree_map(lambda x: x[-1], preds)
            mae = mae_loss_fn(last_pred, outBatch, surf_weights, atmos_weights, gamma, alpha, beta)
            rmse = weighted_rmse_batch(last_pred, outBatch)
            return mae, rmse

        (mae, rmse), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)

        # for debug purposes
        # todo remove after done
        # compute global‐norm of grads
        g2 = sum([jnp.vdot(g, g) for g in jax.tree_util.tree_leaves(grads)])
        grad_norm = jnp.sqrt(g2)
        lr = lr_schedule(state.step)

        return new_state, mae, rmse, rng, grad_norm, lr

    @partial(jax.jit, static_argnums=(4,))
    def eval_step(state, inBatch: Batch, outBatch: Batch, rng, steps: int):
        rng, roll_rng = jax.random.split(rng, 2)
        preds, _, _ = rollout_scan(
            state.apply_fn, model, inBatch, state.params, steps=steps, training=False, rng=roll_rng
        )
        last_pred = jax.tree_util.tree_map(lambda x: x[-1], preds)
        mae = mae_loss_fn(last_pred, outBatch, surf_weights, atmos_weights, gamma, alpha, beta)
        rmse = weighted_rmse_batch(last_pred, outBatch)
        return mae, rmse, rng

    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        train_losses = []
        for inBatch, outBatch in loader_train:
            rng, step_rng = jax.random.split(rng, 2)
            state, train_mae, train_rmse, rng, grad_norm, lr = train_step(
                state, inBatch, outBatch, step_rng, cfg.rollout_steps
            )
            train_losses.append({"mae": train_mae, "rmse": train_rmse})
            global_step += 1
            if (global_step + 1) % 20 == 0:
                avg = {
                    "train/mae": float(jnp.stack([x["mae"] for x in train_losses]).mean()),
                    "train/rmse": float(jnp.stack([x["rmse"] for x in train_losses]).mean()),
                    "train/grad_norm": float(grad_norm),
                    "train/lr": float(lr),
                }
                wandb.log(avg, step=global_step)
                train_losses.clear()

            if global_step % 50 == 0:
                print(global_step)
            # if global_step % 150 == 0:
            #     ckpt.save("../tempData/encoder",
            #             state.params["encoder"],
            #             force=True)
            #     ckpt.save("../tempData/backbone",
            #             state.params["backbone"],
            #             force=True)
            #     ckpt.save("../tempData/decoder",
            #             state.params["decoder"],
            #             force=True)
            #     print(f"Saved checkpoint at step {global_step}")

        # Validation
        val_maes = []
        val_rmses = []
        for inBatch, outBatch in loader_eval:
            rng, step_rng = jax.random.split(rng, 2)
            v_mae, v_rmse, rng = eval_step(state, inBatch, outBatch, step_rng, cfg.rollout_steps)
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
