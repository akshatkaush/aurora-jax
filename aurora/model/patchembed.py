"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import math
from typing import Any, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from timm.models.layers.helpers import to_2tuple

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

        # Initialize weights for each variable
        self.weights = {}
        for name in self.var_names:
            # Shape (embed_dim, 1, history_size, patch_size, patch_size)
            weight_shape = (self.embed_dim, 1) + self.kernel_size
            self.weights[name] = self.param(f"weights_{name}", self._weight_init, weight_shape)

        # Initialize bias
        self.bias = self.param("bias", self._bias_init, (self.embed_dim,))

        # Set up normalization layer
        self.norm = self.norm_layer(self.embed_dim) if self.norm_layer else lambda x: x

    def _weight_init(self, key, shape, dtype=jnp.float32):
        """Initialize weights using Kaiming uniform."""
        # Kaiming uniform initialization
        fan_in = shape[1] * shape[2] * shape[3] * shape[4]  # C_in * T * H * W
        bound = math.sqrt(6 / fan_in)  # equivalent to kaiming_uniform with a=sqrt(5)
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    # todo:  Kaiming/He initialization for the bias? what is this, copied, check.
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

        # Select the weights of the variables and history dimensions that are present in the batch
        weight_list = []
        for name in var_names:
            # Select the appropriate time slices if T < history_size
            weight = self.weights[name][:, :, :T, ...]
            weight_list.append(weight)

        # Concatenate weights along the input channel dimension
        weight = jnp.concatenate(weight_list, axis=1)

        # Adjust the stride if history is smaller than maximum
        stride = (T,) + self.kernel_size[1:]

        # Perform 3D convolution
        # In JAX/Flax, we'll use lax.conv_general_dilated for 3D convolution
        proj = self._conv3d(x, weight, self.bias, stride)

        if self.flatten:
            # Reshape to (B, D, L)
            proj = jnp.reshape(proj, (B, self.embed_dim, -1))
            # Transpose to (B, L, D)
            proj = jnp.transpose(proj, (0, 2, 1))

        # Apply normalization
        return self.norm(proj)

    def _conv3d(self, x, weight, bias, stride):
        """Implement 3D convolution using JAX's lax.conv_general_dilated."""
        # Rearrange dimensions to match JAX convention (NDHWC)
        # From (B, V, T, H, W) to (B, T, H, W, V)
        x = jnp.transpose(x, (0, 2, 3, 4, 1))

        # Rearrange weight from (D, V, T, H, W) to (D, T, H, W, V)
        weight = jnp.transpose(weight, (0, 2, 3, 4, 1))

        # Define convolution dimensions
        dimension_numbers = jax.lax.ConvDimensionNumbers(
            lhs_spec=(0, 1, 2, 3, 4),  # (N, D, H, W, C)
            rhs_spec=(0, 1, 2, 3, 4),  # (O, D, H, W, I)
            out_spec=(0, 1, 2, 3, 4),  # (N, D, H, W, C)
        )

        # Perform convolution
        result = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=weight,
            window_strides=stride,
            padding="VALID",
            dimension_numbers=dimension_numbers,
        )

        # Add bias
        result = result + bias.reshape(1, 1, 1, 1, -1)

        # Rearrange back to (B, D, 1, H/P, W/P)
        result = jnp.transpose(result, (0, 4, 1, 2, 3))

        return result
