# lora.py
"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from typing import Literal

import jax.numpy as jnp
from flax import linen as nn

LoRAMode = Literal["single", "all"]


class LoRA(nn.Module):
    """LoRA adaptation for a linear layer in Flax."""

    in_features: int
    out_features: int
    r: int = 4
    alpha: int = 1
    dropout: float = 0.0

    def setup(self):
        assert self.r > 0, "The rank must be strictly positive."
        # mirror PyTorch naming
        self.lora_alpha = self.alpha
        self.scaling = self.lora_alpha / self.r
        self.lora_dropout = nn.Dropout(rate=self.dropout)

        # low-rank adapters
        self.lora_A = self.param(
            "lora_A",
            nn.initializers.variance_scaling(1.0, "fan_in", "normal"),
            (self.r, self.in_features),
        )
        self.lora_B = self.param(
            "lora_B",
            nn.initializers.zeros,
            (self.out_features, self.r),
        )

    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        # x: [..., in_features]
        x = self.lora_dropout(x, deterministic=deterministic)
        delta = x @ self.lora_A.T @ self.lora_B.T  # [..., out_features]
        return delta * self.scaling


class LoRARollout(nn.Module):
    """Per‐rollout‐step LoRA, storing all adapter weights in big arrays."""

    in_features: int
    out_features: int
    r: int = 8
    alpha: int = 8
    dropout: float = 0.0
    max_steps: int = 40
    mode: LoRAMode = "single"

    def setup(self):
        # scaling & dropout
        self.scaling = self.alpha / self.r
        self.lora_dropout = nn.Dropout(rate=self.dropout)

        # how many distinct adapters?
        n = self.max_steps if self.mode == "all" else 1

        # one big array of A’s and B’s
        self.lora_A = self.param(
            "lora_A",
            nn.initializers.variance_scaling(1.0, "fan_in", "normal"),
            (n, self.r, self.in_features),
        )
        self.lora_B = self.param(
            "lora_B",
            nn.initializers.zeros,
            (n, self.out_features, self.r),
        )

    def __call__(
        self,
        x: jnp.ndarray,
        step: jnp.ndarray,
        *,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        x: [..., in_features]
        step: scalar rollout step
        returns: [..., out_features] or zeros if step ≥ max_steps
        """
        # apply dropout once
        x_drop = self.lora_dropout(x, deterministic=deterministic)

        # bring step on‐device
        step = jnp.asarray(step)
        assert step.ndim == 0, "step must be a scalar integer"

        # number of stored adapters
        n = self.lora_A.shape[0]

        # clamp index: [0, n-1]
        idx = jnp.minimum(step, n - 1)

        # select A and B
        A = self.lora_A[idx]  # (r, in_features)
        B = self.lora_B[idx]  # (out_features, r)

        # compute LoRA update
        delta = x_drop @ A.T @ B.T  # [..., out_features]

        # mask out if step ≥ n
        mask = (step < n).astype(x.dtype)

        return delta * self.scaling * mask
