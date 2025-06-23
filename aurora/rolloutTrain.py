import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_leaves

from aurora.batch import Batch


def rollout_scan(
    apply_fn,
    batch: Batch,
    params,
    steps: int,
    training: bool,
    rng: jax.random.key,
    use_remat: bool = True,
):
    """
    Enhanced rollout_scan with configurable gradient checkpointing.
    
    Args:
        apply_fn: Model apply function
        batch: Input batch
        params: Model parameters
        steps: Number of rollout steps
        training: Whether in training mode
        rng: Random key
        use_remat: Whether to use gradient checkpointing (jax.remat)
    
    Returns:
      preds: PyTree of shape (steps, ...) for each leaf
      final_batch: the Batch after the last step
      final_rng: RNG after all splits
    """
    p0 = tree_leaves(params)[0]
    batch = batch.type(p0.dtype)

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

    # Use gradient checkpointing (remat) if requested
    step_fn = jax.remat(_step_fn) if use_remat else _step_fn
    
    (final_batch, final_rng), preds = lax.scan(
        step_fn,
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
    rng: jax.random.key,
    use_remat: bool = False,
):
    """
    Enhanced rollout with grad only through the last step, using lax.scan for JIT compatibility.
    
    Args:
        apply_fn: Model apply function
        batch: Input batch
        params: Model parameters
        steps: Number of rollout steps
        training: Whether in training mode
        rng: Random key
        use_remat: Whether to use gradient checkpointing (usually not needed with stop_gradient)
    
    Returns:
      preds: PyTree of shape (steps, ...) for each leaf
      final_batch: the Batch after the last step
      final_rng: RNG after all splits
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

        # Stop gradients for all but the last step
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

    # Use gradient checkpointing if requested (usually not needed with stop_gradient)
    step_fn = jax.remat(_step_fn) if use_remat else _step_fn

    (final_batch, final_rng), preds = jax.lax.scan(
        step_fn,
        init=(batch, rng),
        xs=jnp.arange(steps),
    )

    return preds, final_batch, final_rng


def rollout_efficient(
    apply_fn,
    batch: Batch,
    params,
    steps: int,
    training: bool,
    rng: jax.random.key,
    strategy: str = "remat",
):
    """
    Memory-efficient rollout with different gradient strategies.
    
    Args:
        apply_fn: Model apply function
        batch: Input batch
        params: Model parameters
        steps: Number of rollout steps
        training: Whether in training mode
        rng: Random key
        strategy: One of "remat", "stop_grad", "none"
            - "remat": Use gradient checkpointing for all steps
            - "stop_grad": Only keep gradients for last step
            - "none": No special memory optimization
    
    Returns:
      preds: PyTree of shape (steps, ...) for each leaf
      final_batch: the Batch after the last step
      final_rng: RNG after all splits
    """
    if strategy == "remat":
        return rollout_scan(apply_fn, batch, params, steps, training, rng, use_remat=True)
    elif strategy == "stop_grad":
        return rollout_scan_stop_gradients(apply_fn, batch, params, steps, training, rng, use_remat=False)
    elif strategy == "none":
        return rollout_scan(apply_fn, batch, params, steps, training, rng, use_remat=False)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Choose from 'remat', 'stop_grad', 'none'")


# Keep original functions for backward compatibility
def rollout_scan_original(
    apply_fn,
    batch: Batch,
    params,
    steps: int,
    training: bool,
    rng: jax.random.key,
):
    """Original rollout_scan function (with remat enabled by default)."""
    return rollout_scan(apply_fn, batch, params, steps, training, rng, use_remat=True)


def rollout_scan_stop_gradients_original(
    apply_fn,
    batch: Batch,
    params,
    steps: int,
    training: bool,
    rng: jax.random.key,
):
    """Original rollout_scan_stop_gradients function."""
    return rollout_scan_stop_gradients(apply_fn, batch, params, steps, training, rng, use_remat=False)
