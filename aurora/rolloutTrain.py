import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_leaves, tree_map

from aurora.batch import Batch


def single_step(
    apply_fn,
    batch: Batch,
    params,
    steps: int,
    training: bool,
    rng: jax.random.key,
):
    """
    Single‐step forward pass:
      preds: PyTree with leading time‐dim = 1
      final_batch: unchanged input batch
      final_rng: post‐split RNG
    """
    # bring batch to correct dtype & device
    p0 = tree_leaves(params)[0]
    batch = batch.type(p0.dtype)
    # batch = batch.crop(model.patch_size)

    # split RNG once
    rng, step_rng = jax.random.split(rng)
    # single‐step prediction
    pred = apply_fn({"params": params}, batch, training=training, rng=step_rng)

    # # wrap into length-1 time axis
    preds = tree_map(lambda x: x[None], pred)

    return preds, batch, rng


def rollout_scan(
    apply_fn,
    # patch_size,
    batch: Batch,
    params,
    steps: int,
    training: bool,
    rng: jax.random.key,
):
    """
    Returns:
      preds: PyTree of shape (steps, ...) for each leaf
      final_batch: the Batch after the last step
      final_rng: RNG after all splits
    """
    # bring batch to correct dtype & device
    p0 = tree_leaves(params)[0]
    batch = batch.type(p0.dtype)
    # batch = batch.crop(patch_size)

    def _step_fn(carry, _):
        batch, rng = carry
        rng, step_rng = jax.random.split(rng)
        pred = apply_fn({"params": params}, batch, training=training, rng=step_rng)
        next_batch = batch.replace(
            surf_vars={
                k: jnp.concatenate([batch.surf_vars[k][:, 1:], v], axis=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: jnp.concatenate([batch.atmos_vars[k][:, 1:], v], axis=1)
                for k, v in pred.atmos_vars.items()
            },
        )
        return (next_batch, rng), pred

    remat_step = jax.remat(_step_fn)
    (final_batch, final_rng), preds = lax.scan(
        remat_step,
        init=(batch, rng),
        xs=None,
        length=steps,
    )

    return preds, final_batch, final_rng


def rollout_scan_stop_gradients(
    apply_fn,
    batch: Batch,
    params,
    steps: int,
    training: bool,
    rng: jax.random.KeyArray,
):
    """
    rollout with grad only through the last step, using lax.scan for JIT compatibility.
    """
    p0 = tree_leaves(params)[0]
    batch = batch.type(p0.dtype)

    def _step_fn(carry, step_idx):
        batch, rng = carry
        rng, step_rng = jax.random.split(rng)
        pred = apply_fn({"params": params}, batch, training=training, rng=step_rng)

        next_batch = batch.replace(
            surf_vars={
                k: jnp.concatenate([batch.surf_vars[k][:, 1:], v], axis=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: jnp.concatenate([batch.atmos_vars[k][:, 1:], v], axis=1)
                for k, v in pred.atmos_vars.items()
            },
        )

        # stop gradient for all steps except last
        pred = jax.lax.cond(
            step_idx < steps - 1,
            lambda _: jax.tree_util.tree_map(jax.lax.stop_gradient, pred),
            lambda _: pred,
            operand=None,
        )
        next_batch = jax.lax.cond(
            step_idx < steps - 1,
            lambda _: jax.tree_util.tree_map(jax.lax.stop_gradient, next_batch),
            lambda _: next_batch,
            operand=None,
        )

        return (next_batch, rng), pred

    (final_batch, final_rng), preds = jax.lax.scan(
        _step_fn,
        init=(batch, rng),
        xs=jnp.arange(steps),
    )

    return preds, final_batch, final_rng
