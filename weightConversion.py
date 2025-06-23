from pathlib import Path
import jax
import jax.numpy as jnp
import xarray as xr
import orbax.checkpoint as ocp
from orbax.checkpoint import utils as orbax_utils
import os

from aurora import AuroraSmall, Batch, Metadata


def load_individual_checkpoint(checkpoint_path):
    """Load individual checkpoint using StandardCheckpointer."""
    checkpointer = ocp.StandardCheckpointer()
    abs_path = os.path.abspath(checkpoint_path)
    return checkpointer.restore(abs_path)


def main():
    """Main function to combine existing individual checkpoints into one."""
    
    print("Loading existing individual checkpoints...")
    
    # Load individual checkpoints
    try:
        backbone_checkpoint = load_individual_checkpoint("checkpointBackbone")
        encoder_checkpoint = load_individual_checkpoint("checkpointEncoder") 
        decoder_checkpoint = load_individual_checkpoint("checkpointDecoder")
        
        # Extract the actual parameters (they are nested under the section name)
        backbone_params = backbone_checkpoint["backbone"]
        encoder_params = encoder_checkpoint["encoder"]
        decoder_params = decoder_checkpoint["decoder"]
        
        print("✓ Individual checkpoints loaded successfully")
        
        # Print parameter counts for verification
        backbone_leaves, _ = jax.tree_util.tree_flatten(backbone_params)
        encoder_leaves, _ = jax.tree_util.tree_flatten(encoder_params)
        decoder_leaves, _ = jax.tree_util.tree_flatten(decoder_params)
        
        print(f"  Backbone: {len(backbone_leaves)} parameter arrays")
        print(f"  Encoder: {len(encoder_leaves)} parameter arrays") 
        print(f"  Decoder: {len(decoder_leaves)} parameter arrays")
        
    except Exception as e:
        print(f"✗ Error loading individual checkpoints: {e}")
        return
    
    # Combine into final parameters structure
    final_params = {
        "backbone": backbone_params,
        "encoder": encoder_params,
        "decoder": decoder_params
    }
    
    print("\nCombining parameters...")
    
    # Calculate total parameters
    total_leaves = len(backbone_leaves) + len(encoder_leaves) + len(decoder_leaves)
    total_params = sum(leaf.size for leaf in backbone_leaves) + \
                   sum(leaf.size for leaf in encoder_leaves) + \
                   sum(leaf.size for leaf in decoder_leaves)
    
    print(f"Combined checkpoint will have:")
    print(f"  Total parameter arrays: {total_leaves}")
    print(f"  Total parameters: {total_params:,}")
    
    # Save the combined parameters
    print("\nSaving combined parameters...")
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(
        "/home1/a/akaush/aurora/checkpointAllParams",
        final_params,
        force=True,
    )
    
    print("✅ Parameter combination completed successfully!")
    print(f"Combined parameters saved to: /home1/a/akaush/aurora/checkpointAllParams")
    print("\nThe combined checkpoint now contains exactly the same parameters")
    print("as the individual checkpoints - they should match perfectly!")


if __name__ == "__main__":
    main()