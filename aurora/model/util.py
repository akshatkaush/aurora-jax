"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from typing import TypeVar

import jax
import jax.numpy as jnp
import torch
from einops import rearrange
from flax import linen as nn

__all__ = [
    "unpatchify",
    "check_lat_lon_dtype",
    "maybe_adjust_windows",
    "init_weights",
]


def unpatchify(x: torch.Tensor, V: int, H: int, W: int, P: int) -> torch.Tensor:
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
    assert x.dim() == 4, f"Expected 4D tensor, but got {x.dim()}D."
    B, C = x.size(0), x.size(2)
    H = H // P
    W = W // P
    assert x.size(1) == H * W
    assert x.size(-1) == V * P**2

    x = x.reshape(shape=(B, H, W, C, P, P, V))
    x = rearrange(x, "B H W C P1 P2 V -> B V C H P1 W P2")
    x = x.reshape(shape=(B, V, C, H * P, W * P))
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


T = TypeVar("T", tuple[int, int], tuple[int, int, int])


def maybe_adjust_windows(window_size: T, shift_size: T, res: T) -> tuple[T, T]:
    """Adjust the window size and shift size if the input resolution is smaller than the window
    size."""
    err_msg = f"Expected same length, found {len(window_size)}, {len(shift_size)} and {len(res)}."
    assert len(window_size) == len(shift_size) == len(res), err_msg

    mut_shift_size, mut_window_size = list(shift_size), list(window_size)
    for i in range(len(res)):
        if res[i] <= window_size[i]:
            mut_shift_size[i] = 0
            mut_window_size[i] = res[i]

    new_window_size: T = tuple(mut_window_size)  # type: ignore[assignment]
    new_shift_size: T = tuple(mut_shift_size)  # type: ignore[assignment]

    assert min(new_window_size) > 0, f"Window size must be positive. Found {new_window_size}."
    assert min(new_shift_size) >= 0, f"Shift size must be non-negative. Found {new_shift_size}."

    return new_window_size, new_shift_size


def trunc_normal(std: float = 0.02, mean: float = 0.0, lower: float = -2.0, upper: float = 2.0):
    """Truncated normal initializer for JAX."""

    def init(key, shape, dtype=jnp.float32):
        # Generate truncated normal values
        return mean + jax.random.truncated_normal(key, lower, upper, shape, dtype) * std

    return init


def init_weights(m: nn.Module):
    """Initializes weights using derived keys from a base seed."""
    # Create fresh key sequence for this initialization
    base_key = jax.random.PRNGKey(0)

    if isinstance(m, (nn.Dense, nn.Conv, nn.ConvTranspose)):
        # Create unique key for this parameter shape
        if hasattr(m, "kernel"):
            shape_key = jax.random.fold_in(base_key, hash(tuple(m.kernel.shape)))
            m.kernel = nn.initializers.truncated_normal(0.02)(shape_key, m.kernel.shape)

        if hasattr(m, "bias") and m.bias is not None:
            bias_key = jax.random.fold_in(base_key, hash(tuple(m.bias.shape)) + 1)
            m.bias = nn.initializers.zeros(bias_key, m.bias.shape)

    elif isinstance(m, nn.LayerNorm):
        if hasattr(m, "scale") and m.scale is not None:
            m.scale = nn.initializers.ones(base_key, m.scale.shape)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias = nn.initializers.zeros(base_key, m.bias.shape)
