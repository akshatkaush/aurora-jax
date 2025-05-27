"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from typing import Generator

import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves

from aurora.batch import Batch, Metadata
from aurora.model.aurora import Aurora

from functools import partial

__all__ = ["rollout"]


def rollout(
    model: Aurora, batch: Batch, steps: int, params, training: bool, rng
) -> Generator[Batch, None, None]:
    """Perform a roll-out to make long-term predictions."""
    p = tree_leaves(params)[0]
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(jax.devices(p.device.platform)[p.device.host_id])
    rng, key = jax.random.split(rng, 2)
    mock_batch = Batch(
        surf_vars={
            k: jax.random.normal(jax.random.split(key, 4)[i],
                           (1, 2, 720, 1440)).astype(jnp.float32)
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
    apply_with_params = partial(model.apply, {"params": params}, training=False)
    jitted_function = jax.jit(apply_with_params)
    _ = jitted_function(mock_batch, rng=rng)
    # jax.tree_util.tree_map(lambda x: x.block_until_ready(), mock_batch)

    # timed_batch = batch
    # timed_rng = rng

    # def one_step():
    #     nonlocal timed_batch, timed_rng
    #     # split RNG, run model, then block until ready
    #     for _ in range(steps):
    #         timed_rng, step_rng = jax.random.split(timed_rng, 2)
    #         pred = jitted_function(timed_batch, rng=step_rng)
    #         # jax.tree_util.tree_map(lambda x: x.block_until_ready(), pred)
    #         # _ = pred.surf_vars["2t"][0, 0, 0, 0].block_until_ready()
    #         # update batch so next call uses the new state
    #         timed_batch = pred.replace(
    #             surf_vars={
    #                 k: jnp.concatenate([timed_batch.surf_vars[k][:, 1:], v], axis=1)
    #                 for k, v in pred.surf_vars.items()
    #             },
    #             atmos_vars={
    #                 k: jnp.concatenate([timed_batch.atmos_vars[k][:, 1:], v], axis=1)
    #                 for k, v in pred.atmos_vars.items()
    #             },
    #         )

    # n_iters = 1
    # total_sec = timeit(one_step, number=n_iters)
    # avg_ms = total_sec * 1000.0 / n_iters
    # print(f"Average single-step time over {n_iters} runs: {avg_ms:.2f} ms")

    # temp_batch = batch
    # n_iters = 20
    # start = time.time()
    # for _ in range(n_iters):
    # batch = temp_batch
    print("Starting rollout")
    for _ in range(steps):
        rng, step_rng = jax.random.split(rng, 2)
        pred = jitted_function(batch, rng=step_rng)
        # pred = model.apply({"params": params}, batch, training=training, rng=step_rng)
        # err.throw()
        yield pred
        batch = pred.replace(
            surf_vars={
                k: jnp.concatenate([batch.surf_vars[k][:, 1:], v], axis=1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars={
                k: jnp.concatenate([batch.atmos_vars[k][:, 1:], v], axis=1)
                for k, v in pred.atmos_vars.items()
            },
        )
    # end = time.time()
    # avg_ms = (end - start) * 1000.0/ n_iters
    # print(f"Average single-step time over {n_iters} runs: {avg_ms:.2f} ms")
