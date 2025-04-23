"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from typing import Tuple

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn

__all__ = [
    "unpatchify",
    "check_lat_lon_dtype",
    "maybe_adjust_windows",
    "init_weights",
]


def unpatchify(x: jnp.ndarray, V: int, H: int, W: int, P: int) -> jnp.ndarray:
    """Unpatchify hidden representation.

    Args:
        x (torch.Tensor): Patchified input of shape `(B, L, C, V * P^2)` where `P` is the
            patch size.
        V (int): Number of variables.
        H (int): Number of latitudes.
        W (int): Number of longitudes.

    Returns:
        torch.Tensor: Unpatchified representation of shape `(B, V, C, H, W)`.
    """
    B, C = x.shape[0], x.shape[2]
    h_patches = H // P
    w_patches = W // P

    x = x.reshape(B, h_patches, w_patches, C, P, P, V)
    x = rearrange(x, "B h w C p1 p2 V -> B V C (h p1) (w p2)")
    return x


def check_lat_lon_dtype(lat: jnp.ndarray, lon: jnp.ndarray) -> None:
    """Assert that `lat` and `lon` are at least `float32` precision."""
    assert lat.dtype in [
        jnp.float32,
        jnp.float64,
    ], f"Latitudes need float32/64 for stability. Found: {lat.dtype}"
    assert lon.dtype in [
        jnp.float32,
        jnp.float64,
    ], f"Longitudes need float32/64 for stability. Found: {lon.dtype}"


# @partial(jax.jit, static_argnums=(0, 1, 2))
def maybe_adjust_windows(window_size, shift_size, res):
    new_ws, new_ss = [], []
    for ws, ss, r in zip(window_size, shift_size, res):
        if r <= ws:
            new_ws.append(r)
            new_ss.append(0)
        else:
            new_ws.append(ws)
            new_ss.append(ss)
    return tuple(new_ws), tuple(new_ss)

    # mut_shift_size, mut_window_size = jnp.ndarray(shift_size), list(window_size)
    # for i in range(len(res)):
    #     if res[i] <= window_size[i]:
    #         mut_shift_size[i] = 0
    #         mut_window_size[i] = res[i]

    # new_window_size: T = tuple(mut_window_size)  # type: ignore[assignment]
    # new_shift_size: T = tuple(mut_shift_size)

    # return new_window_size, new_shift_size


def trunc_normal(std: float = 0.02, mean: float = 0.0, lower: float = -2.0, upper: float = 2.0):
    """Truncated normal initializer for JAX."""

    def init(key, shape, dtype=jnp.float32):
        # Generate truncated normal values
        return mean + jax.random.truncated_normal(key, lower, upper, shape, dtype) * std

    return init


def init_weights(
    key: jax.random.PRNGKey, shape: Tuple[int, ...], dtype: jnp.dtype = jnp.float32
) -> jnp.ndarray:
    """
    Flax initializer mimicking your original:
      - multi‑dimensional params (e.g. kernels)  → truncated normal (σ=0.02)
      - 1D params         (e.g. biases, LN scale/bias) → zeros
    Use as:
      nn.Dense(..., kernel_init=init_weights, bias_init=init_weights)
    """
    # fixed base seed
    base_key = jax.random.PRNGKey(0)
    # unique per‑shape key
    shape_key = jax.random.fold_in(base_key, hash(tuple(shape)) & 0xFFFFFFFF)
    if len(shape) > 1:
        # weight
        return nn.initializers.truncated_normal(stddev=0.02)(shape_key, shape, dtype)
    else:
        # bias or 1‑d scale
        return jnp.zeros(shape, dtype)
