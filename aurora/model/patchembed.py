import math
from typing import Any, Optional, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from timm.layers.helpers import to_2tuple

__all__ = ["LevelPatchEmbed"]


class LevelPatchEmbed(nn.Module):
    """At either the surface or at a single pressure level, maps all variables into a single
    embedding."""

    var_names: Tuple[str, ...]
    patch_size: int
    embed_dim: int
    history_size: int = 1
    norm_layer: Optional[Any] = None
    flatten: bool = True

    def setup(self):
        """Initialize the module parameters."""
        self.kernel_size = (self.history_size,) + to_2tuple(self.patch_size)

        self.weights = flax.core.FrozenDict(
            {
                name: self.param(
                    f"weights_{name}", self._weight_init, (self.embed_dim, 1) + self.kernel_size
                )
                for name in self.var_names
            }
        )

        self.bias = self.param("bias", self._bias_init, (self.embed_dim,))
        self.norm = self.norm_layer(self.embed_dim) if self.norm_layer else lambda x: x

    def _weight_init(self, key, shape, dtype=jnp.float32):
        """Initialize weights using Kaiming uniform."""
        # Kaiming uniform initialization
        fan_in = shape[1] * shape[2] * shape[3] * shape[4]  # C_in * T * H * W
        bound = math.sqrt(6 / fan_in)  # equivalent to kaiming_uniform with a=sqrt(5)
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    def _bias_init(self, key, shape, dtype=jnp.float32):
        """Initialize bias using uniform distribution."""
        # Get fan_in from the first weight (all weights have the same fan_in)
        weight_shape = (self.embed_dim, 1) + self.kernel_size
        fan_in = weight_shape[1] * weight_shape[2] * weight_shape[3] * weight_shape[4]

        if fan_in != 0:
            bound = 1 / math.sqrt(fan_in)
            return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)
        else:
            return jnp.zeros(shape, dtype)

    def __call__(self, x, var_names):
        """Run the embedding.

        Args:
            x: Tensor to embed of shape (B, V, T, H, W).
            var_names: Names of the variables in x. The length should be equal to V.

        Returns:
            Embedded tensor of shape (B, L, D) if flattened, where L = H * W / P^2.
            Otherwise, the shape is (B, D, H', W').
        """
        B, V, T, H, W = x.shape
        assert len(var_names) == V, f"{V} != {len(var_names)}."
        assert self.kernel_size[0] >= T, f"{T} > {self.kernel_size[0]}."
        assert H % self.kernel_size[1] == 0, f"{H} % {self.kernel_size[1]} != 0."
        assert W % self.kernel_size[2] == 0, f"{W} % {self.kernel_size[2]} != 0."
        assert len(set(var_names)) == len(var_names), f"{var_names} contains duplicates."

        weight_list = []
        for name in var_names:
            weight = self.weights[name][:, :, :T, ...]
            weight_list.append(weight)

        weight = jnp.concatenate(weight_list, axis=1)

        stride = (T,) + self.kernel_size[1:]
        proj = jax.lax.conv_general_dilated(
            x,
            weight,
            window_strides=stride,
            padding="VALID",
            dimension_numbers=("NCTHW", "OITHW", "NCTHW"),
        )

        proj += jnp.reshape(self.bias, (1, -1, 1, 1, 1))

        if self.flatten:
            proj = jnp.reshape(proj, (B, self.embed_dim, -1))
            proj = jnp.transpose(proj, (0, 2, 1))

        return self.norm(proj)
