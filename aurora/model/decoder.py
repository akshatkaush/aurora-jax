"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from typing import Tuple

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn

from aurora.batch import Batch, Metadata
from aurora.model.fourier import FourierExpansion
from aurora.model.perceiver import PerceiverResampler
from aurora.model.util import (
    check_lat_lon_dtype,
    init_weights,
    unpatchify,
)

__all__ = ["Perceiver3DDecoder"]


class Perceiver3DDecoder(nn.Module):
    surf_vars: Tuple[str, ...]
    atmos_vars: Tuple[str, ...]
    patch_size: int = 4
    embed_dim: int = 1024
    depth: int = 1
    head_dim: int = 64
    num_heads: int = 8
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    perceiver_ln_eps: float = 1e-5

    def setup(self):
        # self.level_decoder = nn.remat(PerceiverResampler,
        #                               static_argnums=(2,))(
        self.level_decoder = PerceiverResampler(
            latent_dim=self.embed_dim,
            context_dim=self.embed_dim,
            depth=self.depth,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            drop=self.drop_rate,
            residual_latent=True,
            ln_eps=self.perceiver_ln_eps,
        )

        for name in self.surf_vars:
            setattr(
                self,
                f"surf_head_{name}",
                nn.Dense(
                    features=self.patch_size**2,
                    kernel_init=init_weights,
                    bias_init=init_weights,
                    name=f"surf_head_{name}",
                ),
            )
        # Atmospheric prediction heads
        for name in self.atmos_vars:
            setattr(
                self,
                f"atmos_head_{name}",
                nn.Dense(
                    features=self.patch_size**2,
                    kernel_init=init_weights,
                    bias_init=init_weights,
                    name=f"atmos_head_{name}",
                ),
            )
        # Embed atmospheric level expansions
        self.atmos_levels_embed = nn.Dense(
            features=self.embed_dim,
            kernel_init=init_weights,
            bias_init=init_weights,
        )

        self.levels_expansion = FourierExpansion(0.01, 1e5)

    def deaggregate_levels(self, level_embed: jnp.ndarray, x: jnp.ndarray, training) -> jnp.ndarray:
        B, L, C, D = level_embed.shape
        # Flatten batch and level dims
        level_embed = level_embed.reshape((B * L, C, D))
        x = x.reshape((B * L, x.shape[2], D))
        # Cross-attend and project
        x = self.level_decoder(level_embed, x, deterministic=not training)
        return x.reshape((B, L, C, D))

    def __call__(
        self,
        x: jnp.ndarray,
        batch: Batch,
        patch_res: Tuple[int, int, int],
        lead_time: int,
        training: bool = False,
        rng: jax.random.PRNGKey = None,
    ) -> Batch:
        # surf_vars = batch.surf_vars_ordered_keys()
        # atmos_vars = batch.atmos_vars_ordered_keys()
        surf_vars = tuple(batch.surf_vars.keys())
        atmos_vars = tuple(batch.atmos_vars.keys())
        atmos_levels = batch.metadata.atmos_levels

        B, _, _ = x.shape

        lat, lon = batch.metadata.lat, batch.metadata.lon
        check_lat_lon_dtype(lat, lon)
        lat = lat.astype(jnp.float32)
        lon = lon.astype(jnp.float32)
        H, W = lat.shape[0], lon.shape[-1]

        C, H_p, W_p = patch_res
        x = rearrange(x, "B (C H W) D -> B (H W) C D", C=C, H=H_p, W=W_p)

        surf_outs = []
        for name in surf_vars:
            head = getattr(self, f"surf_head_{name}")
            surf_outs.append(head(x[..., :1, :]))
        x_surf = jnp.stack(surf_outs, axis=-1)
        x_surf = x_surf.reshape((B, x.shape[1], 1, -1))
        surf_preds = unpatchify(x_surf, len(surf_vars), H, W, self.patch_size)
        surf_preds = surf_preds.squeeze(2)

        levels_enc = self.levels_expansion(jnp.array(atmos_levels), self.embed_dim).astype(x.dtype)
        levels_embed = self.atmos_levels_embed(levels_enc)  # (C_A, D)
        levels_embed = jnp.broadcast_to(
            levels_embed, (B, x.shape[1], levels_embed.shape[0], levels_embed.shape[1])
        )

        x_atmos = self.deaggregate_levels(levels_embed, x[..., 1:, :], training)  # (B, L, C_A, D)
        x_atmos = jnp.stack(
            [getattr(self, f"atmos_head_{name}")(x_atmos) for name in atmos_vars], axis=-1
        )  # (B, N, C_A, V_A * p*p)
        x_atmos = x_atmos.reshape((B, x.shape[1], x_atmos.shape[2], -1))
        atmos_preds = unpatchify(x_atmos, len(atmos_vars), H, W, self.patch_size)

        new_time_array = tuple(t + lead_time for t in batch.metadata.time)

        # Construct output batch
        return Batch(
            {v: surf_preds[:, i] for i, v in enumerate(surf_vars)},
            batch.static_vars,
            {v: atmos_preds[:, i] for i, v in enumerate(atmos_vars)},
            Metadata(
                lat=lat,
                lon=lon,
                time=new_time_array,
                atmos_levels=atmos_levels,
                rollout_step=batch.metadata.rollout_step + 1,
            ),
        )
