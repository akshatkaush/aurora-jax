"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import jax
import jax.numpy as jnp
from flax import linen as nn

__all__ = ["FourierExpansion"]


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

        prod = jnp.einsum("...i,j->...ij", x, 2 * jnp.pi / wavelengths)
        encoding = jnp.concatenate((jnp.sin(prod), jnp.cos(prod)), axis=-1)

        return encoding.astype(jnp.float32)
