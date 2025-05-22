"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import contextlib
import dataclasses
import warnings
from functools import partial
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp

# import torch
from flax import linen as nn

# from huggingface_hub import hf_hub_download
# from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
#     apply_activation_checkpointing,
# )
from aurora.batch import Batch
from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.encoder import Perceiver3DEncoder
from aurora.model.swin3d import Swin3DTransformerBackbone

__all__ = ["Aurora", "AuroraSmall", "AuroraHighRes"]


class Aurora(nn.Module):
    """The Aurora model.

    Defaults to to the 1.3 B parameter configuration.
    """

    surf_vars: Tuple[str, ...] = ("2t", "10u", "10v", "msl")
    static_vars: Tuple[str, ...] = ("lsm", "z", "slt")
    atmos_vars: Tuple[str, ...] = ("z", "u", "v", "t", "q")
    window_size: Tuple[int, int, int] = (2, 6, 12)
    encoder_depths: Tuple[int, ...] = (6, 10, 8)
    encoder_num_heads: Tuple[int, ...] = (8, 16, 32)
    decoder_depths: Tuple[int, ...] = (8, 10, 6)
    decoder_num_heads: Tuple[int, ...] = (32, 16, 8)
    latent_levels: int = 4
    patch_size: int = 4
    embed_dim: int = 512
    num_heads: int = 16
    mlp_ratio: float = 4.0
    drop_path: float = 0.0
    drop_rate: float = 0.0
    enc_depth: int = 1
    dec_depth: int = 1
    dec_mlp_ratio: float = 2.0
    perceiver_ln_eps: float = 1e-5
    max_history_size: int = 2
    timestep: int = 21600
    stabilise_level_agg: bool = False
    use_lora: bool = True
    lora_steps: int = 40
    lora_mode: str = "single"
    surf_stats_temp: Optional[Dict[str, Tuple[float, float]]] = None
    autocast: bool = False

    def setup(self):
        self.surf_stats = self.surf_stats_temp or {}

        if self.surf_stats:
            warnings.warn(
                f"The normalisation statics for the following surface-level variables are manually "
                f"adjusted: {', '.join(sorted(self.surf_stats.keys()))}. "
                f"Please ensure that this is right!",
                stacklevel=2,
            )

        # self.encoder = nn.remat(
        #     Perceiver3DEncoder,
        #     static_argnums=(2, 3),
        # )(
        self.encoder = Perceiver3DEncoder(
            surf_vars_temp=self.surf_vars,
            static_vars=self.static_vars,
            atmos_vars=self.atmos_vars,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            drop_rate=self.drop_rate,
            mlp_ratio=self.mlp_ratio,
            head_dim=self.embed_dim // self.num_heads,
            depth=self.enc_depth,
            latent_levels=self.latent_levels,
            max_history_size=self.max_history_size,
            perceiver_ln_eps=self.perceiver_ln_eps,
            stabilise_level_agg=self.stabilise_level_agg,
        )

        # self.backbone = nn.remat(
        #     Swin3DTransformerBackbone,
        #     static_argnums=
        # )
        self.backbone = Swin3DTransformerBackbone(
            window_size_temp=self.window_size,
            encoder_depths=self.encoder_depths,
            encoder_num_heads=self.encoder_num_heads,
            decoder_depths=self.decoder_depths,
            decoder_num_heads=self.decoder_num_heads,
            embed_dim=self.embed_dim,
            mlp_ratio=self.mlp_ratio,
            drop_path_rate=self.drop_path,
            drop_rate=self.drop_rate,
            use_lora=self.use_lora,
            lora_steps=self.lora_steps,
            lora_mode=self.lora_mode,
        )

        # self.decoder = nn.remat(
        #    Perceiver3DDecoder,
        #     static_argnums=(2, 3, 4, 5),
        # )(
        self.decoder = Perceiver3DDecoder(
            surf_vars=self.surf_vars,
            atmos_vars=self.atmos_vars,
            patch_size=self.patch_size,
            # Concatenation at the backbone end doubles the dim.
            embed_dim=self.embed_dim * 2,
            head_dim=self.embed_dim * 2 // self.num_heads,
            num_heads=self.num_heads,
            depth=self.dec_depth,
            # Because of the concatenation, high ratios are expensive.
            # We use a lower ratio here to keep the memory in check.
            mlp_ratio=self.dec_mlp_ratio,
            perceiver_ln_eps=self.perceiver_ln_eps,
        )

    def __call__(
        self, batch: Batch, training: bool, rng: Optional[jax.random.PRNGKey] = None
    ) -> Batch:
        """Forward pass.

        Args:
            batch (:class:`Batch`): Batch to run the model on.

        Returns:
            :class:`Batch`: Prediction for the batch.
        """
        # batch = batch.type(p.dtype)
        batch = batch.normalise(surf_stats=self.surf_stats)
        batch = batch.crop(patch_size=self.patch_size)

        H, W = batch.spatial_shape
        patch_res = (
            self.encoder.latent_levels,
            H // self.encoder.patch_size,
            W // self.encoder.patch_size,
        )

        # Insert batch and history dimension for static variables.
        B, T = next(iter(batch.surf_vars.values())).shape[:2]
        batch = dataclasses.replace(
            batch,
            static_vars={
                k: jnp.tile(jnp.expand_dims(v, (0, 1)), (B, T, 1, 1))
                for k, v in batch.static_vars.items()
            },
        )

        # numpy_array = jnp.array(batch)

        # Save the NumPy array to a file
        # np.save('../tempData/arrayJax.npy', numpy_array)

        rng, encoder_rng, backbone_rng, decoder_rng = jax.random.split(rng, 4)

        # start = time.time()
        x = self.encoder(
            batch,
            self.timestep,
            training,
            encoder_rng,
        )
        # end = time.time()
        # jax.debug.print(f"Encoder time: {(end - start) * 1000:.2f} ms")

        # start = time.time()
        with jax.default_device(
            jax.devices("gpu")[0]
        ) if self.autocast else contextlib.nullcontext():
            x = self.backbone(
                x,
                self.timestep,
                batch.metadata.rollout_step,
                patch_res,
                training,
                backbone_rng,
            )
        # end = time.time()
        # jax.debug.print(f"Backbone time: {(end - start) * 1000:.2f} ms")
        # print("backbone printing")

        # start = time.time()
        pred = self.decoder(
            x,
            batch,
            patch_res,
            self.timestep,
            training,
            decoder_rng,
        )
        # end = time.time()
        # jax.debug.print(f"Decoder time: {(end - start) * 1000:.2f} ms")

        # Remove batch and history dimension from static variables.
        pred = dataclasses.replace(
            pred,
            static_vars={k: v[0, 0] for k, v in batch.static_vars.items()},
        )

        # Insert history dimension in prediction. The time should already be right.
        pred = dataclasses.replace(
            pred,
            surf_vars={k: v[:, None] for k, v in pred.surf_vars.items()},
            atmos_vars={k: v[:, None] for k, v in pred.atmos_vars.items()},
        )

        pred = pred.unnormalise(surf_stats=self.surf_stats)
        # print("compilation done")
        return pred
        # return x

    # def load_checkpoint(self, repo: str, name: str, strict: bool = True):
    #     """Load a checkpoint from HuggingFace.

    #     Args:
    #         repo (str): Name of the repository of the form `user/repo`.
    #         name (str): Path to the checkpoint relative to the root of the repository, e.g.
    #             `checkpoint.cpkt`.
    #         strict (bool, optional): Error if the model parameters are not exactly equal to the
    #             parameters in the checkpoint. Defaults to `True`.
    #     """
    #     path = hf_hub_download(repo_id=repo, filename=name)
    #     jax_params = self.load_checkpoint_local(path, strict=strict)
    #     return jax_params

    # def load_checkpoint_local(self, path: str, strict: bool = True):
    #     """Load a checkpoint directly from a file.

    #     Args:
    #         path (str): Path to the checkpoint.
    #         strict (bool, optional): Error if the model parameters are not exactly equal to the
    #             parameters in the checkpoint. Defaults to `True`.
    #     """
    #     # Assume that all parameters are either on the CPU or on the GPU.
    #     # Get the first parameter key (e.g., 'encoder')
    #     # param_key = list(params['params'].keys())[0]
    #     # param_value = params['params'][param_key]     # Access the corresponding parameter value
    #     device = jax.devices()[0]

    #     d = torch.load(
    #         path, map_location="cpu" if device.platform == "cpu" else "cuda", weights_only=True
    #     )

    #     # You can safely ignore all cumbersome processing below. We modified the model after we
    #     # trained it. The code below manually adapts the checkpoints, so the checkpoints are
    #     # compatible with the new model.

    #     # Remove possibly prefix from the keys.
    #     for k, v in list(d.items()):
    #         if k.startswith("net."):
    #             del d[k]
    #             d[k[4:]] = v

    #     # Convert the ID-based parametrization to a name-based parametrization.
    #     if "encoder.surf_token_embeds.weight" in d:
    #         weight = d["encoder.surf_token_embeds.weight"]
    #         del d["encoder.surf_token_embeds.weight"]

    #         assert weight.shape[1] == 4 + 3
    #         for i, name in enumerate(("2t", "10u", "10v", "msl", "lsm", "z", "slt")):
    #             d[f"encoder.surf_token_embeds.weights.{name}"] = weight[:, [i]]

    #     if "encoder.atmos_token_embeds.weight" in d:
    #         weight = d["encoder.atmos_token_embeds.weight"]
    #         del d["encoder.atmos_token_embeds.weight"]

    #         assert weight.shape[1] == 5
    #         for i, name in enumerate(("z", "u", "v", "t", "q")):
    #             d[f"encoder.atmos_token_embeds.weights.{name}"] = weight[:, [i]]

    #     if "decoder.surf_head.weight" in d:
    #         weight = d["decoder.surf_head.weight"]
    #         bias = d["decoder.surf_head.bias"]
    #         del d["decoder.surf_head.weight"]
    #         del d["decoder.surf_head.bias"]

    #         assert weight.shape[0] == 4 * self.patch_size**2
    #         assert bias.shape[0] == 4 * self.patch_size**2
    #         weight = weight.reshape(self.patch_size**2, 4, -1)
    #         bias = bias.reshape(self.patch_size**2, 4)

    #         for i, name in enumerate(("2t", "10u", "10v", "msl")):
    #             d[f"decoder.surf_heads.{name}.weight"] = weight[:, i]
    #             d[f"decoder.surf_heads.{name}.bias"] = bias[:, i]

    #     if "decoder.atmos_head.weight" in d:
    #         weight = d["decoder.atmos_head.weight"]
    #         bias = d["decoder.atmos_head.bias"]
    #         del d["decoder.atmos_head.weight"]
    #         del d["decoder.atmos_head.bias"]

    #         assert weight.shape[0] == 5 * self.patch_size**2
    #         assert bias.shape[0] == 5 * self.patch_size**2
    #         weight = weight.reshape(self.patch_size**2, 5, -1)
    #         bias = bias.reshape(self.patch_size**2, 5)

    #         for i, name in enumerate(("z", "u", "v", "t", "q")):
    #             d[f"decoder.atmos_heads.{name}.weight"] = weight[:, i]
    #             d[f"decoder.atmos_heads.{name}.bias"] = bias[:, i]

    #     # Check if the history size is compatible and adjust weights if necessary.
    #     current_history_size = d["encoder.surf_token_embeds.weights.2t"].shape[2]
    #     if self.max_history_size > current_history_size:
    #         self.adapt_checkpoint_max_history_size(d)
    #     elif self.max_history_size < current_history_size:
    #         raise AssertionError(
    #             f"Cannot load checkpoint with `max_history_size` {current_history_size} "
    #             f"into model with `max_history_size` {self.max_history_size}."
    #         )

    #     jax_params = self.convert_pytorch_to_jax(d)
    #     return jax_params

    # def adapt_checkpoint_max_history_size(self, checkpoint: dict[str, torch.Tensor]) -> None:
    #     """Adapt a checkpoint with smaller `max_history_size` to a model with a larger
    #     `max_history_size` than the current model.

    #     If a checkpoint was trained with a larger `max_history_size` than the current model,
    #     this function will assert fail to prevent loading the checkpoint. This is to
    #     prevent loading a checkpoint which will likely cause the checkpoint to degrade is
    #     performance.

    #     This implementation copies weights from the checkpoint to the model and fills zeros
    #     for the new history width dimension. It mutates `checkpoint`.
    #     """
    #     for name, weight in list(checkpoint.items()):
    #         # We only need to adapt the patch embedding in the encoder.
    #         enc_surf_embedding = name.startswith("encoder.surf_token_embeds.weights.")
    #         enc_atmos_embedding = name.startswith("encoder.atmos_token_embeds.weights.")
    #         if enc_surf_embedding or enc_atmos_embedding:
    #             # This shouldn't get called with current logic but leaving here
    #                                       for future proofing
    #             # and in cases where its called outside current context.
    #             if not (weight.shape[2] <= self.max_history_size):
    #                 raise AssertionError(
    #                     f"Cannot load checkpoint with `max_history_size` {weight.shape[2]} "
    #                     f"into model with `max_history_size` {self.max_history_size}."
    #                 )

    #             # Initialize the new weight tensor.
    #             new_weight = torch.zeros(
    #                 (weight.shape[0], 1, self.max_history_size, weight.shape[3], weight.shape[4]),
    #                 device=weight.device,
    #                 dtype=weight.dtype,
    #             )
    #             # Copy the existing weights to the new tensor by duplicating the
    #                                           histories provided
    #             # into any new history dimensions. The rest remains at zero.
    #             new_weight[:, :, : weight.shape[2]] = weight

    #             checkpoint[name] = new_weight

    # def configure_activation_checkpointing(self):
    #     """Configure activation checkpointing.

    #     This is required in order to compute gradients without running out of memory.
    #     """
    #     apply_activation_checkpointing(self, check_fn=lambda x: isinstance(x, BasicLayer3D))

    # def convert_pytorch_to_jax(self, state_dict):
    #     """
    #     Convert PyTorch state dictionary to JAX-compatible format.

    #     Args:
    #         state_dict (dict): PyTorch state dictionary with parameter names and tensors.

    #     Returns:
    #         dict: JAX-compatible state dictionary with the same keys and JAX arrays.
    #     """
    #     jax_state_dict = {}
    #     for key, value in state_dict.items():
    #         # Convert PyTorch tensors to JAX arrays
    #         jax_state_dict[key] = jnp.array(value.cpu().numpy())
    #     return jax_state_dict


AuroraSmall = partial(
    Aurora,
    encoder_depths=(2, 6, 2),
    encoder_num_heads=(4, 8, 16),
    decoder_depths=(2, 6, 2),
    decoder_num_heads=(16, 8, 4),
    embed_dim=256,
    num_heads=8,
    use_lora=False,
)

AuroraHighRes = partial(
    Aurora,
    patch_size=10,
    encoder_depths=(6, 8, 8),
    decoder_depths=(8, 8, 6),
)
