"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Parts of this code are inspired by

    https://github.com/microsoft/ClimaX/blob/6d5d354ffb4b91bb684f430b98e8f6f8af7c7f7c/src/climax/utils/pos_embed.py
"""

import jax
import jax.numpy as jnp
from jax import lax
from timm.models.layers.helpers import to_2tuple

from aurora.model.fourier import FourierExpansion

__all__ = ["pos_scale_enc"]


def avg_pool2d(x, kernel_size):
    """Average pooling using JAX."""
    return lax.reduce_window(
        x,
        init_value=0.0,
        computation=lax.add,
        window_dimensions=(1, kernel_size[0], kernel_size[1]),
        window_strides=(1, kernel_size[0], kernel_size[1]),
        padding="VALID",
    ) / (kernel_size[0] * kernel_size[1])


def max_pool2d(x, kernel_size):
    """Max pooling using JAX."""
    return lax.reduce_window(
        x,
        init_value=-jnp.inf,
        computation=lax.max,
        window_dimensions=(1, kernel_size[0], kernel_size[1]),
        window_strides=(1, kernel_size[0], kernel_size[1]),
        padding="VALID",
    )


def patch_root_area(
    lat_min: jnp.ndarray,
    lon_min: jnp.ndarray,
    lat_max: jnp.ndarray,
    lon_max: jnp.ndarray,
) -> jnp.ndarray:
    """For a rectangular patch on a sphere, compute the square root of the area of the patch in
    units km^2. The root is taken to return units of km, and thus stay scalable between different
    resolutions.

    Args:
        lat_min (torch.Tensor): Minimum latitutes of patches.
        lon_min (torch.Tensor): Minimum longitudes of patches.
        lat_max (torch.Tensor): Maximum latitudes of patches.
        lon_max (torch.Tensor): Maximum longitudes of patches.

    Returns:
        torch.Tensor: Square root of the area.
    """
    # Calculate area of latitude-longitude grid using the following formula. Phis are latitudes
    # and thetas are longitudes.
    #
    #   area = R**2 * pi * (sin(phi_1) - sin(phi_2)) * (theta_1 - theta_2)
    #
    # Taken from
    #
    #   https://www.johndcook.com/blog/2023/02/21/sphere-grid-area/
    #

    # todo:  Check if the following assertions are necessary.
    #  Not recommended in jax, but for control flow, I am allowing it.
    if not jnp.all(lat_max > lat_min):
        raise ValueError("All lat_max values must exceed lat_min values")
    if not jnp.all(lon_max > lon_min):
        raise ValueError("All lon_max values must exceed lon_min values (no wrap-around)")
    if not jnp.all(jnp.abs(lat_max) <= 90.0) & jnp.all(jnp.abs(lat_min) <= 90.0):
        raise ValueError("Latitudes out of [-90, 90] degree range")
    if not jnp.all(lon_max <= 360.0) & jnp.all(lon_min <= 360.0):
        raise ValueError("Longitudes exceed 360 degree maximum")
    if not jnp.all(lon_max >= 0.0) & jnp.all(lon_min >= 0.0):
        raise ValueError("Negative longitudes detected")
    area = (
        6371**2
        * jnp.pi
        * (jnp.sin(jnp.deg2rad(lat_max)) - jnp.sin(jnp.deg2rad(lat_min)))
        * (jnp.deg2rad(lon_max) - jnp.deg2rad(lon_min))
    )

    if not jnp.all(area > 0.0):
        raise ValueError("Non-positive area calculated - check input coordinates")

    return jnp.sqrt(area)


def pos_scale_enc_grid(
    encode_dim: int,
    grid: jnp.ndarray,
    patch_dims: tuple,
    pos_expansion: FourierExpansion,
    scale_expansion: FourierExpansion,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute the position and scale encoding for a latitude-longitude grid.

    Requires batch dimensions in the input and returns a batch dimension.

    Args:
        encode_dim (int): Output encoding dimension `D`. Must be a multiple of four: splits
            across latitudes and longitudes and across sines and cosines.
        grid (torch.Tensor): Latitude-longitude grid of dimensions `(B, 2, H, W)`. `grid[:, 0]`
            should be the latitudes of `grid[:, 1]` should be the longitudes.
        patch_dims (tuple): Patch dimensions. Different x-values and y-values are supported.
        pos_expansion (:class:`aurora.model.fourier.FourierExpansion`): Fourier expansion for the
            latitudes and longitudes.
        scale_expansion (:class:`aurora.model.fourier.FourierExpansion`): Fourier expansion for the
            patch areas.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Positional encoding and scale encoding of shape
            `(B, H/patch[0] * W/patch[1], D)`.
    """
    rng = jax.random.PRNGKey(0)
    assert encode_dim % 4 == 0
    assert grid.ndim == 4

    # Take the 2D pooled values of the mesh. This is the same as subsequent 1D pooling over the
    # x-axis and then ove the y-axis.
    grid_h = avg_pool2d(grid[:, 0], patch_dims)
    grid_w = avg_pool2d(grid[:, 1], patch_dims)

    # Compute the square root of the area of the patches specified by the latitude-longitude
    # grid.
    grid_lat_max = max_pool2d(grid[:, 0], patch_dims)
    grid_lat_min = -max_pool2d(-grid[:, 0], patch_dims)
    grid_lon_max = max_pool2d(grid[:, 1], patch_dims)
    grid_lon_min = -max_pool2d(-grid[:, 1], patch_dims)
    root_area = patch_root_area(grid_lat_min, grid_lon_min, grid_lat_max, grid_lon_max)

    # Use half of dimensions for the latitudes of the midpoints of the patches and the other
    # half for the longitudes. Before computing the encodings, flatten over the spatial dimensions.
    B = grid_h.shape[0]
    variables = pos_expansion.init(rng, grid_h.reshape(B, -1), encode_dim // 2)

    # Then apply it with the variables
    # Then apply it with the variables
    encode_h = pos_expansion.apply(variables, grid_h.reshape(B, -1), encode_dim // 2)  # (B, L, D/2)
    encode_w = pos_expansion(grid_w.reshape(B, -1), encode_dim // 2)  # (B, L, D/2)
    pos_encode = jnp.concatenate((encode_h, encode_w), axis=-1)  # (B, L, D)

    # Scale encoding (assuming root_area is already a jnp.ndarray)
    scale_encode = scale_expansion(root_area.reshape(B, -1), encode_dim)  # (B, L, D)

    return pos_encode, scale_encode


def lat_lon_meshgrid(lat: jnp.ndarray, lon: jnp.ndarray) -> jnp.ndarray:
    """Construct a meshgrid of latitude and longitude coordinates.

    `torch.meshgrid(*tensors, indexing="xy")` gives the same behavior as calling
    `numpy.meshgrid(*arrays, indexing="ij")`::

        lat = torch.tensor([1, 2, 3])
        lon = torch.tensor([4, 5, 6])
        grid_x, grid_y = torch.meshgrid(lat, lon, indexing='xy')
        grid_x = tensor([[1, 2, 3], [1, 2, ,3], [1, 2, 3]])
        grid_y = tensor([[4, 4, 4], [5, 5, ,5], [6, 6, 6]])

    Args:
        lat (torch.Tensor): Vector of latitudes.
        lon (torch.Tensor): Vector of longitudes.

    Returns:
        torch.Tensor: Meshgrid of shape `(2, len(lat), len(lon))`.
    """
    assert lat.ndim == 1
    assert lon.ndim == 1

    grid = jnp.stack(jnp.meshgrid(lat, lon, indexing="xy"), axis=0)
    grid = grid.transpose(0, 2, 1)

    return grid


def pos_scale_enc(
    encode_dim: int,
    lat: jnp.ndarray,
    lon: jnp.ndarray,
    patch_dims: int | list | tuple,
    pos_expansion: FourierExpansion,
    scale_expansion: FourierExpansion,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Positional encoding of latitude-longitude data.

    Does not support batch dimensions in the input and does not return batch dimensions either.

    Args:
        encode_dim (int): Output encoding dimension `D`.
        lat (torch.Tensor): Latitudes, `H`. Can be either a vector or a matrix.
        lon (torch.Tensor): Longitudes, `W`. Can be either a vector or a matrix.
        patch_dims (Union[list, tuple]): Patch dimensions. Different x-values and y-values are
            supported.
        pos_expansion (:class:`aurora.model.fourier.FourierExpansion`): Fourier expansion for the
            latitudes and longitudes.
        scale_expansion (:class:`aurora.model.fourier.FourierExpansion`): Fourier expansion for the
            patch areas.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Positional encoding and scale encoding of shape
            `(H/patch[0] * W/patch[1], D)`.
    """
    if lat.ndim == lon.ndim == 1:
        grid = lat_lon_meshgrid(lat, lon)
    elif lat.ndim == lon.ndim == 2:
        grid = jnp.stack([lat, lon], axis=0)
    else:
        raise ValueError(
            f"Latitudes and longitudes must either both be vectors or both be matrices, "
            f"but have dimensionalities {lat.dim()} and {lon.dim()} respectively."
        )

    grid = grid[None,]  # Add batch dimension.

    pos_encoding, scale_encoding = pos_scale_enc_grid(
        encode_dim,
        grid,
        to_2tuple(patch_dims),
        pos_expansion=pos_expansion,
        scale_expansion=scale_expansion,
    )

    return pos_encoding.squeeze(0), scale_encoding.squeeze(0)  # Return without batch dimension.
