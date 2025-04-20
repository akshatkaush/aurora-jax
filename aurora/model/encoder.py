"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from typing import Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from einops import rearrange

from aurora.area import area, radius_earth
from aurora.batch import Batch
from aurora.model.fourier import FourierExpansion
from aurora.model.patchembed import LevelPatchEmbed
from aurora.model.perceiver import MLP, PerceiverResampler
from aurora.model.posencoding import pos_scale_enc
from aurora.model.util import (
    check_lat_lon_dtype,
)

__all__ = ["Perceiver3DEncoder"]


class Perceiver3DEncoder(nn.Module):
    surf_vars_temp: tuple[str, ...]
    static_vars: tuple[str, ...] | None
    atmos_vars: tuple[str, ...]
    patch_size: int = 4
    latent_levels: int = 8
    embed_dim: int = 1024
    num_heads: int = 16
    head_dim: int = 64
    drop_rate: float = 0.1
    depth: int = 2
    mlp_ratio: float = 4.0
    max_history_size: int = 2
    perceiver_ln_eps: float = 1e-5
    stabilise_level_agg: bool = False

    def setup(self):
        if self.static_vars is not None:
            self.surf_vars = self.surf_vars_temp + self.static_vars
        else:
            self.surf_vars = self.surf_vars_temp

        self.atmos_latents = self.param(
            "atmos_latents",
            nn.initializers.truncated_normal(0.02),
            (self.latent_levels - 1, self.embed_dim),
        )

        self.surf_level_encoding = self.param(
            "surf_level_encoding", nn.initializers.truncated_normal(0.02), (self.embed_dim,)
        )

        # Surface components
        self.surf_mlp = MLP(self.embed_dim, int(self.embed_dim * self.mlp_ratio), self.drop_rate)
        self.surf_norm = nn.LayerNorm(epsilon=1e-5)

        # Learnable embedding to encode the surface level.
        self.pos_embed = nn.Dense(self.embed_dim)
        self.scale_embed = nn.Dense(self.embed_dim)
        self.lead_time_embed = nn.Dense(self.embed_dim)
        self.absolute_time_embed = nn.Dense(self.embed_dim)
        self.atmos_levels_embed = nn.Dense(self.embed_dim)

        # Patch embeddings
        self.surf_token_embeds = LevelPatchEmbed(
            self.surf_vars, self.patch_size, self.embed_dim, self.max_history_size
        )

        self.atmos_token_embeds = LevelPatchEmbed(
            self.atmos_vars, self.patch_size, self.embed_dim, self.max_history_size
        )

        # Learnable pressure level aggregation
        self.level_agg = PerceiverResampler(
            latent_dim=self.embed_dim,
            context_dim=self.embed_dim,
            depth=self.depth,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            drop=self.drop_rate,
            mlp_ratio=self.mlp_ratio,
            ln_eps=self.perceiver_ln_eps,
            ln_k_q=self.stabilise_level_agg,
        )
        _delta = 0.01

        coords = jnp.array(
            [
                # The smallest patches will be at the poles. Just use the north pole.
                [90, 0],
                [90, _delta],
                [90 - _delta, _delta],
                [90 - _delta, 0],
            ],
            dtype=jnp.float64,
        )
        _min_patch_area: float = area(coords)
        _area_earth = 4 * jnp.pi * radius_earth * radius_earth

        self.levels_exp = FourierExpansion(_delta, 1e5)
        self.pos_expansion = FourierExpansion(_delta, 720)
        self.scale_expansion = FourierExpansion(_min_patch_area, _area_earth)
        self.absolute_time_expansion = FourierExpansion(1, 24 * 365.25, assert_range=False)
        self.lead_time_expansion = FourierExpansion(1 / 60, 24 * 7 * 3)

        self.pos_drop = nn.Dropout(self.drop_rate)
        # self.apply(init_weights)

    def aggregate_levels(self, x: jnp.ndarray) -> jnp.ndarray:
        B, _, L, _ = x.shape
        C_A, D = self.atmos_latents.shape
        latents = self.atmos_latents.astype(x.dtype)
        latents = jnp.expand_dims(self.atmos_latents, 1)
        latents = jnp.broadcast_to(latents, (B, C_A, L, D))  # (C_A, D) to (B, C_A, L, D)

        x = jnp.einsum("bcld->blcd", x)
        x = x.reshape(B * L, -1, self.embed_dim)
        latents = jnp.einsum("bcld->blcd", latents).reshape(B * L, -1, self.embed_dim)

        x = self.level_agg(latents, x)
        x = x.reshape(B, L, -1, self.embed_dim)
        return jnp.einsum("blcd->bcld", x)

    def __call__(
        self, batch: Batch, lead_time: int, training: bool, rng: Optional[jax.random.PRNGKey] = None
    ) -> jnp.ndarray:
        # rng = jax.random.PRNGKey(0)
        surf_vars = batch.surf_vars_ordered_keys()
        static_vars = batch.static_vars_ordered_keys()
        atmos_vars = batch.atmos_vars_ordered_keys()
        atmos_levels = batch.metadata.atmos_levels

        # todo make faster
        x_surf = jnp.stack(batch.surf_vars_ordered_values(), axis=2)
        x_static = jnp.stack(batch.static_vars_ordered_values(), axis=2)
        x_atmos = jnp.stack(batch.atmos_vars_ordered_values(), axis=2)

        B, T, _, C, H, W = x_atmos.shape
        assert x_surf.shape[:2] == (B, T), f"Expected shape {(B, T)}, got {x_surf.shape[:2]}."
        if self.static_vars is None:
            assert x_static is None, "Static variables given, but not configured."
        else:
            assert x_static is not None, "Static variables not given."
            x_static = jnp.broadcast_to(x_static, (B, T) + x_static.shape[2:])
            x_surf = jnp.concatenate((x_surf, x_static), axis=2)
            surf_vars = surf_vars + static_vars

        lat, lon = batch.metadata.lat, batch.metadata.lon
        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.astype(jnp.float32), lon.astype(jnp.float32)
        assert lat.shape[0] == H and lon.shape[-1] == W

        # Patch embed the surface level
        x_surf = jnp.transpose(x_surf, (0, 2, 1, 3, 4))  # b t v h w -> b v t h w
        x_surf = self.surf_token_embeds(x_surf, surf_vars)  # (B, L, D)
        dtype = x_surf.dtype

        x_atmos = rearrange(x_atmos, "b t v c h w -> (b c) v t h w")  # b t v c h w -> (b c) v t h w
        x_atmos = self.atmos_token_embeds(x_atmos, atmos_vars)
        x_atmos = jnp.reshape(x_atmos, (B, C, -1, self.embed_dim))  # (b c) l d -> b c l d

        # Add surface level encoding
        x_surf = x_surf + self.surf_level_encoding[None, None, :].astype(dtype)
        # Add Perceiver-like MLP for surface level
        x_surf = x_surf + self.surf_norm(self.surf_mlp(x_surf))

        # Add atmospheric pressure encoding
        atmos_levels_tensor = jnp.array(atmos_levels, dtype=jnp.float32)
        atmos_levels_encode = self.levels_exp(atmos_levels_tensor, self.embed_dim).astype(dtype)
        atmos_levels_embed = self.atmos_levels_embed(atmos_levels_encode)
        atmos_levels_embed = jnp.expand_dims(jnp.expand_dims(atmos_levels_embed, 0), 2)
        x_atmos = x_atmos + atmos_levels_embed  # (B, C_A, L, D)

        # Aggregate over pressure levels.
        x_atmos = self.aggregate_levels(x_atmos)  # (B, C_A, L, D) to (B, C, L, D)

        # Concatenate the surface level with the atmospheric levels
        x_surf_expanded = jnp.expand_dims(x_surf, 1)
        x = jnp.concatenate((x_surf_expanded, x_atmos), axis=1)

        pos_encode, scale_encode = pos_scale_enc(
            self.embed_dim,
            lat,
            lon,
            self.patch_size,
            pos_expansion=self.pos_expansion,
            scale_expansion=self.scale_expansion,
        )

        pos_encode = self.pos_embed(
            jnp.expand_dims(jnp.expand_dims(pos_encode, 0), 0).astype(dtype)
        )
        scale_encode = self.scale_embed(
            jnp.expand_dims(jnp.expand_dims(scale_encode, 0), 0).astype(dtype)
        )
        x = x + pos_encode + scale_encode

        x = jnp.reshape(x, (B, -1, self.embed_dim))
        # Add lead time embedding
        lead_hours = lead_time / 3600
        lead_times = lead_hours * jnp.ones((B,), dtype=dtype)
        lead_time_encode = self.lead_time_expansion(lead_times, self.embed_dim).astype(dtype)
        lead_time_emb = self.lead_time_embed(lead_time_encode)  # (B, D)
        x = x + jnp.expand_dims(lead_time_emb, 1)  # (B, L', D) + (B, 1, D)

        # Add absolute time embedding
        absolute_times = jnp.array([t / 3600 for t in batch.metadata.time])
        absolute_time_encode = self.absolute_time_expansion(absolute_times, self.embed_dim)
        absolute_time_embed = self.absolute_time_embed(absolute_time_encode.astype(dtype))
        x = x + jnp.expand_dims(absolute_time_embed, 1)  # (B, L, D) + (B, 1, D)

        x = self.pos_drop(x, deterministic=True)
        return x
