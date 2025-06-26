"""FiLM (Feature-wise Linear Modulation) layer implementation."""

import flax.linen as nn
import jax.numpy as jnp

__all__ = ["AdaptiveLayerNorm"]


class AdaptiveLayerNorm(nn.Module):
    """Adaptive layer normalisation with scale and shift modulation."""

    dim: int
    context_dim: int
    scale_bias: float = 0.0

    def setup(self):
        self.ln = nn.LayerNorm(use_scale=False, use_bias=False)
        self.ln_modulation = nn.Dense(
            features=self.dim * 2,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x, c):
        """
        Args:
            x: Input tensor of shape (B, L, D)
            c: Conditioning tensor of shape (B, D)

        Returns:
            Output tensor of shape (B, L, D)
        """
        # Process conditioning tensor
        modulation = self.ln_modulation(nn.silu(c))  # (B, 2*D)
        modulation = modulation[:, None, :]  # Add sequence dimension (B, 1, 2*D)
        shift, scale = jnp.split(modulation, 2, axis=-1)

        # Apply layer norm and modulation
        x_norm = self.ln(x)
        return x_norm * (self.scale_bias + scale) + shift
