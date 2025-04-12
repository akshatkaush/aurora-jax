"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from typing import Literal

import flax.linen as nn
import jax.numpy as jnp
from jax.nn.initializers import variance_scaling, zeros

__all__ = ["LoRA", "LoRARollout", "LoRAMode"]

LoRAMode = Literal["single", "all"]


class LoRA(nn.Module):
    """LoRA adaptation for a linear layer."""

    in_features: int
    out_features: int
    r: int = 4
    alpha: int = 1
    dropout: float = 0.0

    def setup(self):
        self.lora_alpha = self.alpha
        self.scaling = self.lora_alpha / self.r

        self.lora_A = self.param(
            "lora_A", variance_scaling(1.0, "fan_in", "normal"), (self.r, self.in_features)
        )
        self.lora_B = self.param("lora_B", zeros, (self.out_features, self.r))

        self.lora_dropout = nn.Dropout(rate=self.dropout)

    def __call__(self, x, training=False):
        x = self.lora_dropout(x, deterministic=not training)
        x = jnp.dot(x, self.lora_A.T)
        x = jnp.dot(x, self.lora_B.T)

        return x * self.scaling


class LoRARollout(nn.Module):
    """Per-roll-out-step LoRA finetuning."""

    in_features: int
    out_features: int
    r: int = 8
    alpha: int = 8
    dropout: float = 0.0
    max_steps: int = 40
    mode: LoRAMode = "single"

    def setup(self):
        lora_layers = self.max_steps if self.mode == "all" else 1
        self.loras = [
            LoRA(
                in_features=self.in_features,
                out_features=self.out_features,
                r=self.r,
                alpha=self.alpha,
                dropout=self.dropout,
            )
            for _ in range(lora_layers)
        ]

    def __call__(self, x, step, deterministic=True) -> jnp.ndarray:
        """Compute the LoRA adaptation.

        Args:
            x (jnp.ndarray): Input to the linear layer.
            step (int): Roll-out step, starting at zero.
            deterministic (bool): Whether to apply dropout.

        Returns:
            jnp.ndarray: Additive correction for the output of the linear layer.
        """
        valid_step = (step >= 0) & (step < self.max_steps)

        result = jnp.where(
            valid_step,
            self._apply_lora(x, step, not deterministic),
            jnp.zeros((x.shape[0], self.out_features), dtype=x.dtype),
        )

        return result

    def _apply_lora(self, x, step, training):
        if self.mode == "single":
            return self.loras[0](x, training=training)
        elif self.mode == "all":
            safe_step = jnp.minimum(step, self.max_steps - 1)
            return self.loras[safe_step](x, training=training)
