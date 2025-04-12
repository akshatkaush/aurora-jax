"""Copyright (c) Microsoft Corporation. Licensed under the MIT license.

Code adapted from

    https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py

"""

import itertools
from datetime import timedelta
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn
from jax.experimental import checkify

from aurora.model.film import AdaptiveLayerNorm
from aurora.model.fourier import FourierExpansion
from aurora.model.lora import LoRAMode, LoRARollout

__all__ = ["Swin3DTransformerBackbone"]


class DropPath(nn.Module):
    drop_prob: float = 0.0
    deterministic: Optional[bool] = None
    scale_by_keep: bool = True

    @nn.compact
    def __call__(self, x, deterministic=False, *, rng=None):
        """Apply DropPath to input tensor."""
        # Convert deterministic to Python boolean if it's not already

        # Early return for deterministic mode or zero dropout
        # if deterministic or self.drop_prob == 0.0:
        #     return x

        # if rng is None:
        #     rng = jax.random.PRNGKey(0)
        #     keep_prob = 1.0 - self.drop_prob

        # # Create mask for dropout (batch dimension only)
        # shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        # random_tensor = jax.random.uniform(rng, shape, dtype=x.dtype)
        # binary_tensor = jnp.floor(random_tensor + keep_prob)

        # # Apply scaling without using conditional
        # # This avoids the conditional that was causing the broadcasting issue
        # scaling_factor = 1.0
        # if self.scale_by_keep:
        #     scaling_factor = 1.0 / keep_prob

        # return x * scaling_factor * binary_tensor

        return x


def to_3tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x) * 3 if len(x) == 1 else tuple(x)[:3]
    return (x,) * 3


class MLP(nn.Module):
    in_features: int
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    act_layer: Callable = nn.gelu
    drop: float = 0.0

    @nn.compact
    def __call__(self, x, training: bool = False):
        # Set default features
        hidden = self.hidden_features or self.in_features
        out = self.out_features or self.in_features

        # Forward pass
        x = nn.Dense(hidden)(x)
        x = self.act_layer(x)
        x = nn.Dropout(self.drop)(x, deterministic=not training)
        x = nn.Dense(out)(x)
        x = nn.Dropout(self.drop)(x, deterministic=not training)
        return x


class WindowAttention(nn.Module):
    """Window-based multi-head self-attention (W-MSA) with Flax implementation."""

    dim: int
    window_size: Tuple[int, int, int]
    num_heads: int
    qkv_bias: bool = True
    qk_scale: Optional[float] = None
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    lora_steps: int = 40
    lora_mode: LoRAMode = "single"
    use_lora: bool = False

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        assert (
            self.dim % self.num_heads == 0
        ), f"dim {self.dim} must be divisible by num_heads {self.num_heads}"

        self.qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias, name="qkv")
        self.proj = nn.Dense(self.dim, name="proj")
        self.proj_drop_layer = nn.Dropout(self.proj_drop)
        self.attn_drop_layer = nn.Dropout(self.attn_drop)
        if self.use_lora:
            self.lora_qkv = LoRARollout(
                self.dim,
                self.dim * 3,
                self.lora_r,
                self.lora_alpha,
                self.lora_dropout,
                self.lora_steps,
                self.lora_mode,
                name="lora_qkv",
            )
            self.lora_proj = LoRARollout(
                self.dim,
                self.dim,
                self.lora_r,
                self.lora_alpha,
                self.lora_dropout,
                self.lora_steps,
                self.lora_mode,
                name="lora_proj",
            )
        else:
            # Dummy functions that return zero
            self.lora_qkv = lambda x, step: 0
            self.lora_proj = lambda x, step: 0

    def __call__(self, x, mask=None, rollout_step=0, training=False, rng=None):
        """Run the forward pass of the window-based multi-head self-attention layer.

        Args:
            x: Input features with shape of (nW*B, N, C).
            mask: Attention mask with shape of (nW, ws, ws), where nW is the number of windows,
                and ws is the window size (i.e. total tokens inside the window).
            rollout_step: Current rollout step for LoRA.
            deterministic: Whether to apply dropout.

        Returns:
            Output of shape (nW*B, N, C).
        """
        qkv = self.qkv(x) + self.lora_qkv(x, rollout_step)
        attn_rng, rng = jax.random.split(rng, 2)

        qkv = rearrange(qkv, "B N (qkv H D) -> qkv B H N D", H=self.num_heads, qkv=3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dropout_rate = jax.lax.cond(training, lambda _: self.attn_drop, lambda _: 0.0, operand=None)
        if mask is not None:
            nW = mask.shape[0]
            attention_bias = jnp.where(mask == 0, -jnp.inf, 0.0)
            # B_orig = x.shape[0] // nW

            q = rearrange(q, "(B nW) H N D -> B nW H N D", nW=nW)
            k = rearrange(k, "(B nW) H N D -> B nW H N D", nW=nW)
            v = rearrange(v, "(B nW) H N D -> B nW H N D", nW=nW)

            B, _, H, N, D = q.shape

            attn_weights = jnp.matmul(q, jnp.swapaxes(k, -2, -1))

            scale = self.qk_scale or 1.0 / jnp.sqrt(self.head_dim)
            attn_weights = attn_weights * scale

            attention_bias = attention_bias.reshape(1, nW, 1, N, N)
            attn_weights = attn_weights + attention_bias
            attn_weights = jax.nn.softmax(attn_weights, axis=-1)
            attn_weights = jax.lax.cond(
                jnp.logical_not(training) & (dropout_rate > 0),  # Combined condition
                lambda x: self.attn_drop_layer(
                    x, deterministic=True
                ),  # No need for jnp.logical_not
                lambda x: x,
                attn_weights,
            )
            x = jnp.matmul(attn_weights, v)
            x = rearrange(x, "B nW H N D -> (B nW) H N D")
        else:
            x = jax.lax.cond(
                training,
                # Training branch (concrete dropout rate + deterministic=False)
                lambda: nn.dot_product_attention(
                    query=q,
                    key=k,
                    value=v,
                    dropout_rate=self.attn_drop,  # Use raw dropout value
                    deterministic=False,  # Concrete boolean
                    precision=None,
                    dtype=q.dtype,
                    dropout_rng=attn_rng,
                ),
                # Inference branch (0 dropout + deterministic=True)
                lambda: nn.dot_product_attention(
                    query=q,
                    key=k,
                    value=v,
                    dropout_rate=0.0,  # Concrete float
                    deterministic=True,  # Concrete boolean
                    precision=None,
                    dtype=q.dtype,
                    dropout_rng=attn_rng,
                ),
            )

        x = rearrange(x, "B H N D -> B N (H D)")
        x = self.proj(x) + self.lora_proj(x, rollout_step)

        x = self.proj_drop_layer(x, deterministic=~training)

        return x

    def __repr__(self):
        return (
            f"WindowAttention(dim={self.dim}, "
            f"window_size={self.window_size}, "
            f"num_heads={self.num_heads})"
        )


def window_partition_3d(x: jnp.ndarray, window_size: tuple[int, int, int]) -> jnp.ndarray:
    """JIT-compatible 3D window partitioning with static shape validation."""
    # Extract static dimensions from window_size
    Wc, Wh, Ww = window_size

    # Precompute static output shape parts
    B, C, H, W, D = x.shape
    num_windows_c = C // Wc
    num_windows_h = H // Wh
    num_windows_w = W // Ww

    # Create checkified function
    @checkify.checkify
    def inner_fn(x):
        checkify.check(C % Wc == 0, "Channel dimension error")
        checkify.check(H % Wh == 0, "Height dimension error")
        checkify.check(W % Ww == 0, "Width dimension error")

        x = jnp.reshape(
            x,
            (B, num_windows_c, Wc, num_windows_h, Wh, num_windows_w, Ww, D),
        )
        x = x.transpose(0, 1, 3, 5, 2, 4, 6, 7)
        return jnp.reshape(
            x,
            (-1, Wc, Wh, Ww, D),
        )

    # Execute with error checking
    err, result = inner_fn(x)
    err.throw()
    return result


def window_reverse_3d(windows, window_size, C, H, W):
    """Pure JAX implementation of 3D window reversal.

    Args:
        windows: Input array of shape (num_windows*B, Wc, Wh, Ww, D)
        window_size: Tuple of (window_depth, window_height, window_width)
        C: Original number of channels/depth
        H: Original height
        W: Original width

    Returns:
        Reconstructed array of shape (B, C, H, W, D)
    """
    Wc, Wh, Ww = window_size
    C1, H1, W1 = C // Wc, H // Wh, W // Ww
    B = windows.shape[0] // (C1 * H1 * W1)

    x = windows.reshape(B, C1, H1, W1, Wc, Wh, Ww, -1)
    x = x.transpose(0, 1, 4, 2, 5, 3, 6, 7)
    return x.reshape(B, C, H, W, -1)


def get_two_sided_padding(H_padding: int, W_padding: int) -> Tuple[int, int, int, int]:
    if W_padding > 0:
        padding_left = W_padding // 2
        padding_right = W_padding - padding_left
    else:
        padding_left = padding_right = 0

    if H_padding > 0:
        padding_top = H_padding // 2
        padding_bottom = H_padding - padding_top
    else:
        padding_top = padding_bottom = 0

    return padding_left, padding_right, padding_top, padding_bottom


def get_three_sided_padding(
    C_padding: int, H_padding: int, W_padding: int
) -> Tuple[int, int, int, int, int, int]:
    pad_front = 0
    pad_back = 0
    if C_padding > 0:
        pad_front = C_padding // 2
        pad_back = C_padding - pad_front

    padding_left, padding_right, padding_top, padding_bottom = get_two_sided_padding(
        H_padding, W_padding
    )

    return (
        pad_front,
        pad_back,
        padding_left,
        padding_right,
        padding_top,
        padding_bottom,
    )


# def _concrete_pad_3d(x, pad_size, value):
#     left_pad, right_pad, top_pad, bottom_pad, front_pad, back_pad =
# get_three_sided_padding(*pad_size)

#     pad_width = [
#         (0, 0),                # Batch
#         (left_pad, right_pad), # Channel
#         (top_pad, bottom_pad), # Height
#         (front_pad, back_pad), # Width
#         (0, 0)                 # Depth
#     ]

#     return jnp.pad(x, pad_width, mode='constant', constant_values=value)

# @custom_vjp
# def pad_3d(x: jnp.ndarray,
#            pad_size: tuple[jax.Array, jax.Array, jax.Array],
#            value: float = 0.0) -> jnp.ndarray:
#     # This will be implemented in the _fwd function
#     return io_callback(
#         _concrete_pad_3d,
#         jax.ShapeDtypeStruct(x.shape, x.dtype),  # Output shape/type
#         x,
#         pad_size,
#         value
#     )

# def pad_3d_fwd(x, pad_size, value):
#     """Forward pass with concrete padding calculation."""
#     # Convert tracer arrays to concrete values
#     padded = pad_3d(x, pad_size, value)
#     return padded, (x.shape, pad_size)

# def pad_3d_bwd(residual, grad_output):
#     orig_shape, pad_size = residual
#     c_pad, h_pad, w_pad = pad_size

#     # Calculate padding amounts
#     left_pad, right_pad = c_pad//2, c_pad - c_pad//2
#     top_pad, bottom_pad = h_pad//2, h_pad - h_pad//2
#     front_pad, back_pad = w_pad//2, w_pad - w_pad//2

#     # Slice gradient to original shape
#     grad_input = grad_output[
#         :,  # Batch
#         left_pad:left_pad + orig_shape[1],  # Channel
#         top_pad:top_pad + orig_shape[2],    # Height
#         front_pad:front_pad + orig_shape[3],# Width
#         :   # Depth
#     ]
#     return (grad_input, None, None)

# # Link the custom VJP
# pad_3d.defvjp(pad_3d_fwd, pad_3d_bwd)

# def pad_jitted_function(x, pad_size, count):
#     padded = pad_3d(x, pad_size, count)
#     return padded


def pad_3d(x: jnp.ndarray, pad_size: tuple[int, int, int], value: float = 0.0) -> jnp.ndarray:
    """Pads 5D tensors (B, C, D, H, W) with specified padding on C, D, H dimensions."""
    padding = get_three_sided_padding(*pad_size)
    pad_width = (
        (0, 0),
        (padding[0], padding[0]),
        (padding[4], padding[5]),
        (padding[2], padding[3]),
        (0, 0),
    )
    return jnp.pad(x, pad_width, constant_values=value)


def crop_3d(x: jnp.ndarray, pad_size: tuple[int, int, int]) -> jnp.ndarray:
    """Undoes the `pad_3d` function by cropping the padded values."""
    B, C, H, W, D = x.shape
    Cp, Hp, Wp = pad_size
    pfront, pback, pleft, pright, ptop, pbottom = get_three_sided_padding(Cp, Hp, Wp)
    x = x[:, pfront : C - pback, ptop : H - pbottom, pleft : W - pright, :]
    return x


def get_3d_merge_groups() -> List[Tuple[int, int]]:
    merge_groups_2d = jnp.array([(1, 2), (4, 5), (7, 8)], dtype=jnp.int32)
    offsets = jnp.arange(3, dtype=jnp.int32) * 9
    result = jnp.zeros((9, 2), dtype=jnp.int32)

    result = jax.lax.fori_loop(
        0,
        3,
        lambda i, res: jax.lax.dynamic_update_slice_in_dim(
            res, merge_groups_2d + offsets[i], i * 3, axis=0
        ),
        result,
    )
    return result


def compute_3d_shifted_window_mask(
    C: int,
    H: int,
    W: int,
    ws: Tuple[int, int, int],
    ss: Tuple[int, int, int],
    dtype: jnp.dtype = jnp.bfloat16,
    warped: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX implementation of 3D shifted window attention mask"""

    img_mask = jnp.zeros((1, C, H, W, 1), dtype=dtype)

    # Define slices using original logic
    c_slices = ((0, C - ws[0]), (C - ws[0], C - ss[0]), (C - ss[0], C))
    h_slices = ((0, H - ws[1]), (H - ws[1], H - ss[1]), (H - ss[1], H))
    w_slices = ((0, W - ws[2]), (W - ws[2], W - ss[2]), (W - ss[2], W))

    slices = list(itertools.product(c_slices, h_slices, w_slices))
    cnt = 0

    for (cs, ce), (hs, he), (ws_, we) in slices:
        patch = jnp.full((1, ce - cs, he - hs, we - ws_, 1), cnt, dtype=dtype)
        img_mask = jax.lax.dynamic_update_slice(img_mask, patch, (0, cs, hs, ws_, 0))
        cnt += 1

    if warped:
        for grp1, grp2 in get_3d_merge_groups():
            img_mask = jnp.where(img_mask == grp1, grp2, img_mask)

    pad_size = (
        (ws[0] - C % ws[0]) % ws[0],
        (ws[1] - H % ws[1]) % ws[1],
        (ws[2] - W % ws[2]) % ws[2],
    )
    img_mask = pad_3d(img_mask, pad_size, value=cnt)
    mask_windows = window_partition_3d(img_mask, ws)

    mask_windows = mask_windows.reshape(-1, ws[0] * ws[1] * ws[2])

    # Attention mask calculation (same broadcasting logic)
    attn_mask = mask_windows[:, None, :] - mask_windows[:, :, None]
    attn_mask = jnp.where(attn_mask != 0, -100.0, 0.0)

    return attn_mask, img_mask
    # img_mask = jnp.zeros((1, C, H, W, 1), dtype=dtype)

    # # Pre-compute all slice boundaries
    # c_indices = [(0, C - ws[0]), (C - ws[0], C - ss[0]), (C - ss[0], C)]
    # h_indices = [(0, H - ws[1]), (H - ws[1], H - ss[1]), (H - ss[1], H)]
    # w_indices = [(0, W - ws[2]), (W - ws[2], W - ss[2]), (W - ss[2], W)]

    # def update_img_mask(img_mask, counter):
    #     for c_idx, c_slice in enumerate(c_indices):
    #         for h_idx, h_slice in enumerate(h_indices):
    #             for w_idx, w_slice in enumerate(w_indices):
    #                 # In JAX, we need to use a functional update pattern
    #                 # This is a bit verbose but follows the structure of the original
    #                 for c in c_slice:
    #                     for h in h_slice:
    #                         for w in w_slice:
    #                             img_mask = img_mask.at[0, c, h, w, 0].set(counter)
    #                 counter += 1
    #     return img_mask, counter

    # img_mask, cnt = update_img_mask(img_mask, 0)

    # if warped:
    #     merge_groups = get_3d_merge_groups()

    #     # Apply merges
    #     def apply_merges(mask, groups):
    #         for grp1, grp2 in groups:
    #             mask = jnp.where(mask == grp1, jnp.ones_like(mask) * grp2, mask)
    #         return mask

    #     img_mask = apply_merges(img_mask, merge_groups)

    # pad_size = (ws[0] - C % ws[0], ws[1] - H % ws[1], ws[2] - W % ws[2])
    # pad_size = (pad_size[0] % ws[0], pad_size[1] % ws[1], pad_size[2] % ws[2])
    # img_mask = pad_3d(img_mask, pad_size, cnt)

    # # Window partition
    # mask_windows = window_partition_3d(img_mask, ws)  # (nW*B, ws[0], ws[1], ws[2], 1)
    # mask_windows = jnp.reshape(mask_windows, (-1, ws[0] * ws[1] * ws[2]))

    # # Two patches communicate if they are in the same group
    # attn_mask = jnp.expand_dims(mask_windows, 1) - jnp.expand_dims(mask_windows, 2)

    # # Apply mask values
    # attn_mask = jnp.where(attn_mask != 0, jnp.full_like(attn_mask, -100.0), attn_mask)
    # attn_mask = jnp.where(attn_mask == 0, jnp.full_like(attn_mask, 0.0), attn_mask)

    # return attn_mask, img_mask


class Swin3DTransformerBlock(nn.Module):
    """3D Swin Transformer block with JAX optimizations"""

    dim: int
    num_heads: int
    time_dim: int
    window_size: Tuple[int, int, int] = (2, 7, 7)
    shift_size: Tuple[int, int, int] = (0, 0, 0)
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: float = 0.0
    act_layer: Callable = nn.gelu
    scale_bias: float = 0.0
    lora_steps: int = 40
    lora_mode: str = "single"
    use_lora: bool = False

    def setup(self):
        self.norm1 = AdaptiveLayerNorm(self.dim, self.time_dim, self.scale_bias)

        self.attn = WindowAttention(
            dim=self.dim,
            window_size=self.window_size,
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            proj_drop=self.drop,
            lora_steps=self.lora_steps,
            use_lora=self.use_lora,
            lora_mode=self.lora_mode,
        )

        self.drop_path_layer = DropPath(drop_prob=self.drop_path)
        self.norm2 = AdaptiveLayerNorm(self.dim, self.time_dim, self.scale_bias)

        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = MLP(
            in_features=self.dim,
            hidden_features=mlp_hidden_dim,
            act_layer=self.act_layer,
            drop=self.drop,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        c: jnp.ndarray,
        res: Tuple[int, int, int],
        rollout_step: int,
        warped: bool = True,
        training: bool = False,
        rng: jax.random.PRNGKey = None,
    ) -> jnp.ndarray:
        C, H, W = res
        B, L, D = x.shape
        shortcut = x

        # ws, ss = maybe_adjust_windows(self.window_size, self.shift_size, res)
        ws = []
        ss = []

        # Perform the operation
        for w, s, r in zip(self.window_size, self.shift_size, res):
            if r <= w:
                ws.append(r)
                ss.append(0)
            else:
                ws.append(w)
                ss.append(s)
        x = x.reshape(B, C, H, W, D)

        shifted_x, attn_mask = self._shift_windows(x, ws, ss, warped)
        if rng is not None:
            attention_rng, residual_rng, rng = jax.random.split(rng, 3)
        shifted_x = self._process_windows(
            shifted_x, ws, ss, attn_mask, rollout_step, training, attention_rng
        )

        x = self._merge_windows(shifted_x, ss, (C, H, W), ws)
        x = x.reshape(B, C * H * W, D)

        x = self._apply_residuals(x, shortcut, c, training, residual_rng)
        return x

    def _shift_windows(self, x, ws, ss, warped):
        if not all(s == 0 for s in ss):
            shifted_x = jnp.roll(x, shift=(-ss[0], -ss[1], -ss[2]), axis=(1, 2, 3))
            attn_mask, _ = compute_3d_shifted_window_mask(
                x.shape[1], x.shape[2], x.shape[3], ws, ss, warped=warped
            )
        else:
            shifted_x = x
            attn_mask = None

        return shifted_x, attn_mask

    def _process_windows(self, shifted_x, ws, ss, attn_mask, rollout_step, training, attention_rng):
        pad_size = (
            (-shifted_x.shape[1] % ws[0]),
            (-shifted_x.shape[2] % ws[1]),
            (-shifted_x.shape[3] % ws[2]),
        )
        shifted_x = pad_3d(shifted_x, pad_size)

        # Window partition
        x_windows = window_partition_3d(shifted_x, ws)
        x_windows = x_windows.reshape(-1, ws[0] * ws[1] * ws[2], self.dim)

        # Attention computation
        attn_windows = self.attn(
            x_windows,
            mask=attn_mask,
            rollout_step=rollout_step,
            training=training,
            rng=attention_rng,
        )
        _, pad_C, pad_H, pad_W, _ = shifted_x.shape
        # Window reconstruction
        return window_reverse_3d(
            attn_windows.reshape(-1, ws[0], ws[1], ws[2], self.dim), ws, pad_C, pad_H, pad_W
        )

    def _merge_windows(self, shifted_x, ss, res, ws):
        pad_size = ((-res[0] % ws[0]), (-res[1] % ws[1]), (-res[2] % ws[2]))
        shifted_x = crop_3d(shifted_x, pad_size)
        return jnp.roll(shifted_x, shift=ss, axis=(1, 2, 3)) if any(ss) else shifted_x

    def _apply_residuals(self, x, shortcut, c, training, rng):
        # First residual
        if rng is not None:
            first_residual_rng, second_residual_rng, rng = jax.random.split(rng, 3)
        x = shortcut + self.drop_path_layer(self.norm1(x, c), deterministic=~training)

        # MLP processing
        mlp_out = self.mlp(x)

        # Second residual
        return x + self.drop_path_layer(self.norm2(mlp_out, c), deterministic=~training)


class PatchMerging3D(nn.Module):
    """Patch merging layer maintaining original structure with JAX optimizations."""

    dim: int  # Feature dimension

    def setup(self):
        self.reduction = nn.Dense(2 * self.dim, use_bias=False)
        self.norm = nn.LayerNorm()

    def _merge(self, x: jnp.ndarray, res: tuple[int, int, int]) -> jnp.ndarray:
        C, H, W = res
        B, L, D = x.shape

        assert L == C * H * W, f"Wrong feature size: {L} vs {C}*{H}*{W}={C*H*W}"
        assert H > 1 and W > 1, f"Spatial dims must be >1 (H:{H}, W:{W})"

        x = x.reshape((B, C, H, W, D))
        pad_h = H % 2
        pad_w = W % 2
        x = pad_3d(x, (0, pad_h, pad_w))
        new_H, new_W = x.shape[2], x.shape[3]

        return rearrange(
            x.reshape(B, C, new_H // 2, 2, new_W // 2, 2, D), "B C H h W w D -> B (C H W) (h w D)"
        )

    def __call__(self, x: jnp.ndarray, input_resolution: tuple[int, int, int]) -> jnp.ndarray:
        """Flax forward pass with JAX optimizations:
        - Automatic parameter handling
        - Functional transformations
        - Immutable tensor operations
        """
        x = self._merge(x, input_resolution)
        x = self.norm(x)
        return self.reduction(x)


class PatchSplitting3D(nn.Module):
    """Patch splitting layer optimized for JAX/Flax"""

    dim: int  # Number of input channels

    def setup(self):
        # Dimension checks remain important in JAX
        assert self.dim % 2 == 0, f"dim ({self.dim}) should be divisible by 2."

        # Flax convention: input features come last
        self.lin1 = nn.Dense(self.dim * 2, use_bias=False)
        self.lin2 = nn.Dense(self.dim // 2, use_bias=False)
        self.norm = nn.LayerNorm()

    def _split(self, x, res, crop):
        C, H, W = res
        B, L, D = x.shape

        # Convert values to JAX arrays for checkify
        checkify.check(
            L == C * H * W,
            "Token count mismatch: {} vs {}*{}*{}={}",
            jnp.asarray(L),
            jnp.asarray(C),
            jnp.asarray(H),
            jnp.asarray(W),
            jnp.asarray(C * H * W),
        )
        checkify.check(D % 4 == 0, "Feature dimension {} not divisible by 4", jnp.asarray(D))

        x = x.reshape(B, C, H, W, 2, 2, D // 4)
        x = rearrange(x, "B C H W h w D -> B C (H h) (W w) D")  # (B, C, 2*H, 2*W, D/4)
        x = crop_3d(x, crop)  # Undo padding from `PatchMerging` (if any).
        return x.reshape(B, -1, D // 4)  # (B, C*2H*2W, D/4)

    def __call__(self, x, input_resolution, crop=(0, 0, 0)):
        x = self.lin1(x)
        x = self._split(x, input_resolution, crop)
        x = self.norm(x)
        return self.lin2(x)


class BasicLayer3D(nn.Module):
    """A basic 3D Swin Transformer layer for one stage."""

    dim: int
    depth: int
    num_heads: int
    ws: tuple[int, int, int]
    time_dim: int
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop: float = 0.0
    attn_drop: float = 0.0
    drop_path: float | list[float] = 0.0
    downsample_temp: type[PatchMerging3D] | None = None
    upsample_temp: type[PatchSplitting3D] | None = None
    scale_bias: float = 0.0
    lora_steps: int = 40
    lora_mode: LoRAMode = "single"
    use_lora: bool = False

    def setup(self):
        if self.downsample_temp is not None and self.upsample_temp is not None:
            raise ValueError("Cannot set both `downsample` and `upsample`.")

        self.blocks = [
            Swin3DTransformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                window_size=self.ws,
                shift_size=(
                    (0, 0, 0)
                    if (i % 2 == 0)
                    else (self.ws[0] // 2, self.ws[1] // 2, self.ws[2] // 2)
                ),
                time_dim=self.time_dim,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop,
                attn_drop=self.attn_drop,
                drop_path=(
                    self.drop_path[i] if isinstance(self.drop_path, list) else self.drop_path
                ),
                scale_bias=self.scale_bias,
                use_lora=self.use_lora,
                lora_steps=self.lora_steps,
                lora_mode=self.lora_mode,
            )
            for i in range(self.depth)
        ]

        self.downsample = self.downsample_temp(dim=self.dim) if self.downsample_temp else None
        self.upsample = self.upsample_temp(dim=self.dim) if self.upsample_temp else None

    def __call__(
        self,
        x: jnp.ndarray,
        c: jnp.ndarray,
        res: Tuple[int, int, int],
        crop: Tuple[int, int, int] = (0, 0, 0),
        rollout_step: int = 0,
        training: bool = False,
        rng: jax.random.PRNGKey = None,
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        for blk in self.blocks:
            rng, layer_rng = jax.random.split(rng)
            x = blk(x, c, res, rollout_step=rollout_step, training=training, rng=layer_rng)

        if self.downsample is not None:
            x_scaled = self.downsample(x, res)
            return x_scaled, x
        if self.upsample is not None:
            x_scaled = self.upsample(x, res, crop)
            return x_scaled, x
        return x, None


class Basic3DEncoderLayer(BasicLayer3D):
    """A basic 3D Swin Transformer encoder layer. Used for FSDP, which requires a subclass."""


class Basic3DDecoderLayer(BasicLayer3D):
    """A basic 3D Swin Transformer decoder layer. Used for FSDP, which requires a subclass."""


class Swin3DTransformerBackbone(nn.Module):
    embed_dim: int = 96
    encoder_depths: tuple[int, ...] = (2, 2, 6, 2)
    encoder_num_heads: tuple[int, ...] = (3, 6, 12, 24)
    decoder_depths: tuple[int, ...] = (2, 6, 2, 2)
    decoder_num_heads: tuple[int, ...] = (24, 12, 6, 3)
    window_size_temp: int | tuple[int, int, int] = 7
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.1
    drop_path_rate: float = 0.1
    lora_steps: int = 40
    lora_mode: LoRAMode = "single"
    use_lora: bool = False

    def setup(self):
        self.window_size = to_3tuple(self.window_size_temp)
        self.num_encoder_layers = len(self.encoder_depths)
        self.num_decoder_layers = len(self.decoder_depths)

        self.time_mlp = nn.Sequential([nn.Dense(self.embed_dim), nn.silu, nn.Dense(self.embed_dim)])

        self.lead_time_expansion = FourierExpansion(1 / 60, 24 * 7 * 3)

        checkify.check(
            sum(self.encoder_depths) == sum(self.decoder_depths),
            "Sum of encoder and decoder depths must be equal",
        )
        dpr = jnp.linspace(0, self.drop_path_rate, sum(self.encoder_depths))

        self.encoder_layers = [
            Basic3DEncoderLayer(
                dim=int(self.embed_dim * 2**i_layer),
                depth=self.encoder_depths[i_layer],
                num_heads=self.encoder_num_heads[i_layer],
                ws=self.window_size,
                mlp_ratio=self.mlp_ratio,
                time_dim=self.embed_dim,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[
                    sum(self.encoder_depths[:i_layer]) : sum(self.encoder_depths[: i_layer + 1])
                ],
                downsample_temp=(
                    PatchMerging3D if (i_layer < self.num_encoder_layers - 1) else None
                ),
                use_lora=self.use_lora,
                lora_steps=self.lora_steps,
                lora_mode=self.lora_mode,
            )
            for i_layer in range(self.num_encoder_layers)
        ]

        self.decoder_layers = [
            Basic3DDecoderLayer(
                dim=int(self.embed_dim * 2 ** (self.num_decoder_layers - i_layer - 1)),
                depth=self.decoder_depths[i_layer],
                num_heads=self.decoder_num_heads[i_layer],
                ws=self.window_size,
                mlp_ratio=self.mlp_ratio,
                time_dim=self.embed_dim,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[
                    sum(self.decoder_depths[:i_layer]) : sum(self.decoder_depths[: i_layer + 1])
                ],
                upsample_temp=(
                    PatchSplitting3D if (i_layer < self.num_decoder_layers - 1) else None
                ),
                use_lora=self.use_lora,
                lora_steps=self.lora_steps,
                lora_mode=self.lora_mode,
            )
            for i_layer in range(self.num_decoder_layers)
        ]

    def get_encoder_specs(
        self, patch_res: Tuple[int, int, int]
    ) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
        all_res = [patch_res]
        padded_outs = []
        for _ in range(1, self.num_encoder_layers):
            C, H, W = all_res[-1]
            pad_H, pad_W = H % 2, W % 2
            padded_outs.append((0, pad_H, pad_W))
            all_res.append((C, (H + pad_H) // 2, (W + pad_W) // 2))

        padded_outs.append((0, 0, 0))
        return all_res, padded_outs

    def __call__(
        self,
        x: jnp.ndarray,
        lead_time: float,
        rollout_step: int,
        patch_res: tuple[int, int, int],
        training: bool,
        rng: Optional[jax.random.PRNGKey] = None,
    ) -> jnp.ndarray:
        checkify.check(
            x.shape[1] == patch_res[0] * patch_res[1] * patch_res[2],
            "Input shape does not match patch size.",
        )
        checkify.check(
            patch_res[0] % self.window_size[0] == 0,
            f"Patch height ({patch_res[0]}) must be divisible by ws[0] ({self.window_size[0]})",
        )

        all_enc_res, padded_outs = self.get_encoder_specs(patch_res)

        lead_hours = lead_time / timedelta(hours=1).total_seconds()
        lead_times = lead_hours * jnp.ones(x.shape[0], dtype=jnp.float32)
        time_embed = jnp.asarray(self.lead_time_expansion(lead_times, self.embed_dim))

        # Now use your time_mlp on the precomputed embedding
        c = self.time_mlp(time_embed)

        skips = []

        for i, layer in enumerate(self.encoder_layers):
            rng, layer_rng = jax.random.split(rng)
            x, x_unscaled = layer(
                x, c, all_enc_res[i], rollout_step=rollout_step, training=training, rng=layer_rng
            )
            skips.append(x_unscaled)

        for i, layer in enumerate(self.decoder_layers):
            rng, layer_rng = jax.random.split(rng)
            index = self.num_decoder_layers - i - 1
            x, _ = layer(
                x,
                c,
                all_enc_res[index],
                padded_outs[index - 1],
                rollout_step=rollout_step,
                training=training,
                rng=layer_rng,
            )

            if 0 < i < self.num_decoder_layers - 1:
                x = x + skips[index - 1]
            elif i == self.num_decoder_layers - 1:
                x = jnp.concatenate([x, skips[0]], axis=-1)

        return x
