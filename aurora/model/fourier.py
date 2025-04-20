"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import jax
import jax.numpy as jnp
from flax import linen as nn

__all__ = [
    "FourierExpansion",
    # "pos_expansion",
    # "scale_expansion",
    # "lead_time_expansion",
    # "levels_expansion",
    # "absolute_time_expansion",
]


class FourierExpansion(nn.Module):
    """A Fourier series-style expansion into a high-dimensional space.

    Attributes:
        lower (float): Lower wavelength.
        upper (float): Upper wavelength.
        assert_range (bool): Assert that the encoded tensor is within the specified wavelength
            range.
    """

    lower: float
    upper: float
    assert_range: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, d: int) -> jnp.ndarray:
        """Perform the expansion.

        Adds a dimension of length `d` to the end of the shape of `x`.

        Args:
            x (:class:`jnp.ndarray`): Input to expand of shape `(..., n)`. All elements of `x` must
                lie within `[self.lower, self.upper]` if `self.assert_range` is `True`.
            d (int): Dimensionality. Must be a multiple of two.

        Raises:
            AssertionError: If `self.assert_range` is `True` and not all elements of `x` are not
                within `[self.lower, self.upper]`.
            ValueError: If `d` is not a multiple of two.

        Returns:
            jnp.ndarray: Fourier series-style expansion of `x` of shape `(..., n, d)`.
        """
        # If the input is not within the configured range, the embedding might be ambiguous!
        in_range = jnp.logical_and(self.lower <= jnp.abs(x), jnp.all(jnp.abs(x) <= self.upper))
        in_range_or_zero = jnp.all(jnp.logical_or(in_range, x == 0))  # Allow zeros to pass through.

        if self.assert_range:
            # JAX doesn't support raising exceptions in a computation graph, so we use assert_shape
            # as a pattern to conditionally continue execution
            jax.lax.cond(
                in_range_or_zero,
                lambda _: None,
                lambda _: jax.debug.print(
                    f"The input tensor is not within the configured range "
                    f"[{self.lower}, {self.upper}]."
                ),
                operand=None,
            )

        # We will use half of the dimensionality for `sin` and the other half for `cos`.
        if not (d % 2 == 0):
            raise ValueError("The dimensionality must be a multiple of two.")

        # Always perform the expansion with `float64`s to avoid numerical accuracy shenanigans.
        x = x.astype(jnp.float64)

        wavelengths = jnp.logspace(
            jnp.log10(jnp.array(self.lower)),
            jnp.log10(jnp.array(self.upper)),
            d // 2,
            base=10,
            dtype=x.dtype,
        )

        scale = 2 * jnp.pi / wavelengths
        prod = x[..., None] * scale  # Shape (..., i, j)

        prod = jnp.einsum("...i,j->...ij", x, 2 * jnp.pi / wavelengths, optimize=False)
        encoding = jnp.concatenate((jnp.sin(prod), jnp.cos(prod)), axis=-1)

        return encoding.astype(jnp.float32)


# Determine a reasonable smallest value for the scale embedding by assuming a smallest delta in
# latitudes and longitudes.
# _delta = 0.01  # Reasonable smallest delta in latitude and longitude
# coords = jnp.array(
#     [[90.0, 0.0], [90.0, _delta], [90.0 - _delta, _delta], [90.0 - _delta, 0.0]],
#                                               dtype=jnp.float64
# )

# _min_patch_area: float = area(coords)
# _area_earth = 4 * jnp.pi * radius_earth * radius_earth

# pos_expansion = FourierExpansion(_delta, 720)
# """:class:`.FourierExpansion`: Fourier expansion for the encoding of latitudes and longitudes in
# degrees."""

# scale_expansion = FourierExpansion(_min_patch_area, _area_earth)
# """:class:`.FourierExpansion`: Fourier expansion for the encoding of patch areas in squared
# kilometers."""

# lead_time_expansion = FourierExpansion(1 / 60, 24 * 7 * 3)
# """:class:`.FourierExpansion`: Fourier expansion for the lead time encoding in hours."""

# levels_expansion = FourierExpansion(0.01, 1e5)
# """:class:`.FourierExpansion`: Fourier expansion for the pressure level encoding in hPa."""

# absolute_time_expansion = FourierExpansion(1, 24 * 365.25, assert_range=False)
# """:class:`.FourierExpansion`: Fourier expansion for the absolute time encoding in hours."""
