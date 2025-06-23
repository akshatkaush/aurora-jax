"""PyTorch checkpoint loading utilities for Aurora model.

This module contains utilities for loading PyTorch checkpoints into Aurora models.
Since Aurora is now implemented in JAX, this separate module handles the PyTorch-specific
checkpoint loading and adaptation logic.
"""

import torch
from typing import Dict, Optional
from huggingface_hub import hf_hub_download


class CheckpointLoader:
    """Utility class for loading PyTorch checkpoints into Aurora models."""
    
    def __init__(self, patch_size: int, max_history_size: int):
        """Initialize the checkpoint loader.
        
        Args:
            patch_size (int): Patch size used in the model.
            max_history_size (int): Maximum history size of the target model.
        """
        self.patch_size = patch_size
        self.max_history_size = max_history_size
    
    def load_checkpoint(self, repo: str, name: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Load a checkpoint from HuggingFace.

        Args:
            repo (str): Name of the repository of the form `user/repo`.
            name (str): Path to the checkpoint relative to the root of the repository, e.g.
                `checkpoint.cpkt`.
            device (str): Device to load the checkpoint on. Defaults to "cpu".
                
        Returns:
            Dict[str, torch.Tensor]: The loaded and adapted checkpoint dictionary.
        """
        path = hf_hub_download(repo_id=repo, filename=name)
        return self.load_checkpoint_local(path, device=device)

    def load_checkpoint_local(self, path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Load a checkpoint directly from a file.

        Args:
            path (str): Path to the checkpoint.
            device (str): Device to load the checkpoint on. Defaults to "cpu".
            
        Returns:
            Dict[str, torch.Tensor]: The loaded and adapted checkpoint dictionary.
        """
        d = torch.load(path, map_location=device, weights_only=True)

        # You can safely ignore all cumbersome processing below. We modified the model after we
        # trained it. The code below manually adapts the checkpoints, so the checkpoints are
        # compatible with the new model.

        # Remove possibly prefix from the keys.
        for k, v in list(d.items()):
            if k.startswith("net."):
                del d[k]
                d[k[4:]] = v

        # Convert the ID-based parametrization to a name-based parametrization.
        if "encoder.surf_token_embeds.weight" in d:
            weight = d["encoder.surf_token_embeds.weight"]
            del d["encoder.surf_token_embeds.weight"]

            assert weight.shape[1] == 4 + 3
            for i, name in enumerate(("2t", "10u", "10v", "msl", "lsm", "z", "slt")):
                d[f"encoder.surf_token_embeds.weights.{name}"] = weight[:, [i]]

        if "encoder.atmos_token_embeds.weight" in d:
            weight = d["encoder.atmos_token_embeds.weight"]
            del d["encoder.atmos_token_embeds.weight"]

            assert weight.shape[1] == 5
            for i, name in enumerate(("z", "u", "v", "t", "q")):
                d[f"encoder.atmos_token_embeds.weights.{name}"] = weight[:, [i]]

        if "decoder.surf_head.weight" in d:
            weight = d["decoder.surf_head.weight"]
            bias = d["decoder.surf_head.bias"]
            del d["decoder.surf_head.weight"]
            del d["decoder.surf_head.bias"]

            assert weight.shape[0] == 4 * self.patch_size**2
            assert bias.shape[0] == 4 * self.patch_size**2
            weight = weight.reshape(self.patch_size**2, 4, -1)
            bias = bias.reshape(self.patch_size**2, 4)

            for i, name in enumerate(("2t", "10u", "10v", "msl")):
                d[f"decoder.surf_heads.{name}.weight"] = weight[:, i]
                d[f"decoder.surf_heads.{name}.bias"] = bias[:, i]

        if "decoder.atmos_head.weight" in d:
            weight = d["decoder.atmos_head.weight"]
            bias = d["decoder.atmos_head.bias"]
            del d["decoder.atmos_head.weight"]
            del d["decoder.atmos_head.bias"]

            assert weight.shape[0] == 5 * self.patch_size**2
            assert bias.shape[0] == 5 * self.patch_size**2
            weight = weight.reshape(self.patch_size**2, 5, -1)
            bias = bias.reshape(self.patch_size**2, 5)

            for i, name in enumerate(("z", "u", "v", "t", "q")):
                d[f"decoder.atmos_heads.{name}.weight"] = weight[:, i]
                d[f"decoder.atmos_heads.{name}.bias"] = bias[:, i]

        # Check if the history size is compatible and adjust weights if necessary.
        current_history_size = d["encoder.surf_token_embeds.weights.2t"].shape[2]
        if self.max_history_size > current_history_size:
            self.adapt_checkpoint_max_history_size(d)
        elif self.max_history_size < current_history_size:
            raise AssertionError(
                f"Cannot load checkpoint with `max_history_size` {current_history_size} "
                f"into model with `max_history_size` {self.max_history_size}."
            )

        return d

    def adapt_checkpoint_max_history_size(self, checkpoint: Dict[str, torch.Tensor]) -> None:
        """Adapt a checkpoint with smaller `max_history_size` to a model with a larger
        `max_history_size` than the current model.

        If a checkpoint was trained with a larger `max_history_size` than the current model,
        this function will assert fail to prevent loading the checkpoint. This is to
        prevent loading a checkpoint which will likely cause the checkpoint to degrade is
        performance.

        This implementation copies weights from the checkpoint to the model and fills zeros
        for the new history width dimension. It mutates `checkpoint`.
        """
        for name, weight in list(checkpoint.items()):
            # We only need to adapt the patch embedding in the encoder.
            enc_surf_embedding = name.startswith("encoder.surf_token_embeds.weights.")
            enc_atmos_embedding = name.startswith("encoder.atmos_token_embeds.weights.")
            if enc_surf_embedding or enc_atmos_embedding:
                # This shouldn't get called with current logic but leaving here for future proofing
                # and in cases where its called outside current context.
                if not (weight.shape[2] <= self.max_history_size):
                    raise AssertionError(
                        f"Cannot load checkpoint with `max_history_size` {weight.shape[2]} "
                        f"into model with `max_history_size` {self.max_history_size}."
                    )

                # Initialize the new weight tensor.
                new_weight = torch.zeros(
                    (weight.shape[0], 1, self.max_history_size, weight.shape[3], weight.shape[4]),
                    device=weight.device,
                    dtype=weight.dtype,
                )
                # Copy the existing weights to the new tensor by duplicating the histories provided
                # into any new history dimensions. The rest remains at zero.
                new_weight[:, :, : weight.shape[2]] = weight

                checkpoint[name] = new_weight


def load_pytorch_checkpoint(
    repo_or_path: str, 
    name: Optional[str] = None, 
    patch_size: int = 4, 
    max_history_size: int = 2,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """Convenience function to load a PyTorch checkpoint.
    
    Args:
        repo_or_path (str): Either a HuggingFace repo ID (if name is provided) or local file path.
        name (str, optional): Checkpoint filename if loading from HuggingFace. If None, 
            repo_or_path is treated as a local file path.
        patch_size (int): Patch size used in the model. Defaults to 4.
        max_history_size (int): Maximum history size of the target model. Defaults to 2.
        device (str): Device to load the checkpoint on. Defaults to "cpu".
        
    Returns:
        Dict[str, torch.Tensor]: The loaded and adapted checkpoint dictionary.
    """
    loader = CheckpointLoader(patch_size=patch_size, max_history_size=max_history_size)
    
    if name is not None:
        # Load from HuggingFace
        return loader.load_checkpoint(repo_or_path, name, device=device)
    else:
        # Load from local file
        return loader.load_checkpoint_local(repo_or_path, device=device) 