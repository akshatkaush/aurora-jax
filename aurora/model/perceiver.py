import jax
import jax.numpy as jnp
from einops import rearrange
from flax import linen as nn

__all__ = ["MLP", "PerceiverResampler"]


class MLP(nn.Module):
    """A simple one-hidden-layer MLP."""

    dim: int
    hidden_features: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic=True):
        """Run the MLP."""
        x = nn.Dense(features=self.hidden_features)(x)
        x = jax.nn.gelu(x)
        x = nn.Dense(features=self.dim)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        return x


class PerceiverAttention(nn.Module):
    latent_dim: int
    context_dim: int
    head_dim: int = 64
    num_heads: int = 8
    ln_k_q: bool = False

    def setup(self):
        self.inner_dim = self.head_dim * self.num_heads

        self.to_q = nn.Dense(features=self.inner_dim, use_bias=False)
        self.to_kv = nn.Dense(features=self.inner_dim * 2, use_bias=False)
        self.to_out = nn.Dense(features=self.latent_dim, use_bias=False)

        if self.ln_k_q:
            self.ln_k = nn.LayerNorm(epsilon=1e-5)
            self.ln_q = nn.LayerNorm(epsilon=1e-5)
        else:
            self.ln_k = lambda x: x
            self.ln_q = lambda x: x

    def __call__(self, latents, x):
        """Run the cross-attention module.

        Args:
            latents: Latent features of shape (B, L1, Latent_D)
            x: Context features of shape (B, L2, Context_D)
            deterministic: Whether to use deterministic behavior (unused, for compatibility)

        Returns:
            Latent values of shape (B, L1, Latent_D)
        """
        h = self.num_heads
        q = self.to_q(latents)  # (B, L1, inner_dim)

        kv = self.to_kv(x)  # (B, L2, inner_dim*2)
        k, v = jnp.split(kv, 2, axis=-1)  # Each (B, L2, inner_dim)

        k = self.ln_k(k)
        q = self.ln_q(q)

        # Reshape to separate heads: (B, L, inner_dim) -> (B, h, L, head_dim)
        h = self.num_heads
        q = rearrange(q, "b l (h d) -> b h l d", h=h)
        k = rearrange(k, "b l (h d) -> b h l d", h=h)
        v = rearrange(v, "b l (h d) -> b h l d", h=h)

        out = nn.dot_product_attention(
            q,
            k,
            v,
        )
        # Reshape back: (B, h, L1, head_dim) -> (B, L1, inner_dim)
        out = rearrange(out, "b h l d -> b l (h d)")
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    """Perceiver Resampler module from the Flamingo paper.

    Args:
        latent_dim (int): Dimensionality of the latent features given as input.
        context_dim (int): Dimensionality of the context features also given as input.
        depth (int, optional): Number of attention layers.
        head_dim (int, optional): Attention head dimensionality. Defaults to `64`.
        num_heads (int, optional): Number of heads. Defaults to `16`
        mlp_ratio (float, optional): Dimensionality of the hidden layer divided by that of the
            input for all MLPs. Defaults to `4.0`.
        drop (float, optional): Drop-out rate. Defaults to no drop-out.
        residual_latent (bool, optional): Use residual attention w.r.t. the latent features.
            Defaults to `True`.
        ln_eps (float, optional): Epsilon in the layer normalisation layers. Defaults to
            `1e-5`.
        ln_k_q (bool, optional): Apply an extra layer norm. to the keys and queries of the first
            resampling layer. Defaults to `False`.
    """

    latent_dim: int
    context_dim: int
    depth: int = 1
    head_dim: int = 64
    num_heads: int = 16
    mlp_ratio: float = 4.0
    drop: float = 0.0
    residual_latent: bool = True
    ln_eps: float = 1e-5
    ln_k_q: bool = False

    def setup(self):
        layers = []
        mlp_hidden_dim = int(self.latent_dim * self.mlp_ratio)

        for i in range(self.depth):
            layers.append(
                (
                    PerceiverAttention(
                        latent_dim=self.latent_dim,
                        context_dim=self.context_dim,
                        head_dim=self.head_dim,
                        num_heads=self.num_heads,
                        ln_k_q=self.ln_k_q if i == 0 else False,
                    ),
                    MLP(dim=self.latent_dim, hidden_features=mlp_hidden_dim, dropout=self.drop),
                    nn.LayerNorm(epsilon=self.ln_eps),
                    nn.LayerNorm(epsilon=self.ln_eps),
                )
            )
        self.layers = layers

    def __call__(self, latents, x, deterministic: bool = True):
        """Run the module.

        Args:
            latents (:class:`jnp.ndarray`): Latent features of shape `(B, L1, D1)`.
            x (:class:`jnp.ndarray`): Context features of shape `(B, L2, D1)`.

        Returns:
            jnp.ndarray: Latent features of shape `(B, L1, D1)`.
        """
        for attn, ff, ln1, ln2 in self.layers:
            # We use post-res-norm like in Swin v2 and most Transformer architectures these days.
            # This empirically works better than the pre-norm used in the original Perceiver.
            attn_out = ln1(attn(latents, x))
            # HuggingFace suggests using non-residual attention in Perceiver might work better when
            # the semantics of the query and the output are different:
            #
            #   https://github.com/huggingface/transformers/blob/v4.35.2/src/transformers/models/perceiver/modeling_perceiver.py#L398
            #
            latents = attn_out + latents if self.residual_latent else attn_out
            f_out = ff(latents, deterministic=deterministic)
            latents = ln2(f_out) + latents
        return latents
