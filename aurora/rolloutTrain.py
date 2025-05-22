import jax
from jax.tree_util import tree_leaves, tree_map

from aurora.batch import Batch


def rollout_scan(
    apply_fn,
    model,
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
    batch = batch.crop(model.patch_size)

    # split RNG once
    rng, step_rng = jax.random.split(rng)
    # single‐step prediction
    pred = apply_fn({"params": params}, batch, training=training, rng=step_rng)

    # # wrap into length-1 time axis
    preds = tree_map(lambda x: x[None], pred)

    return preds, batch, rng


# import functools

# import jax
# import jax.numpy as jnp
# from jax import lax
# from jax.tree_util import tree_leaves, tree_map

# from aurora.batch import Batch
# from aurora.model.aurora import Aurora
# from score import mae_loss_fn
# from config import surf_weights, atmos_weights, gamma, alpha, beta


# def rollout_scan(
#     model: Aurora,
#     batch: Batch,
#     params,
#     steps: int,
#     training: bool,
#     rng: jax.random.key,
# ):
#     """
#     Returns:
#       preds: PyTree of shape (steps, ...) for each leaf
#       final_batch: the Batch after the last step
#       final_rng: RNG after all splits
#     """
#     # bring batch to correct dtype & device
#     p0    = tree_leaves(params)[0]
#     batch = batch.type(p0.dtype)
#     batch = batch.crop(model.patch_size)

#     def _step_fn(carry, _):
#         batch, rng = carry
#         rng, step_rng = jax.random.split(rng)
#         pred = model.apply({"params": params}, batch, training=training, rng=step_rng)
#         next_batch = batch.replace(
#             surf_vars={
#                 k: jnp.concatenate([batch.surf_vars[k][:, 1:], v], axis=1)
#                 for k, v in pred.surf_vars.items()
#             },
#             atmos_vars={
#                 k: jnp.concatenate([batch.atmos_vars[k][:, 1:], v], axis=1)
#                 for k, v in pred.atmos_vars.items()
#             },
#         )
#         return (next_batch, rng), pred

#     (final_batch, final_rng), preds = lax.scan(
#         _step_fn,
#         init=(batch, rng),
#         xs=None,
#         length=steps,
#     )
#     return preds, final_batch, final_rng
