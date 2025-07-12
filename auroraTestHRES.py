from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
import xarray as xr
from jax.tree_util import tree_leaves
from save_batches import save_batch_npz

from aurora import AuroraSmall, rollout
from aurora.IterableDataset import HresT0SequenceDataset
from aurora.score import weighted_mae, weighted_rmse


# =============================================================================
# CONFIGURATION AND PATHS
# =============================================================================

# Dataset paths
DATASET_PATH = Path("dataset")
ZARR_PATH = "hresDataset/hres_t0_2021-2022mid.zarr"
OUTPUT_FOLDER = "../tempData"

# Model checkpoint paths
CHECKPOINT_ENCODER = str(Path("checkpointEncoder").resolve())
CHECKPOINT_BACKBONE = str(Path("checkpointBackbone").resolve())
CHECKPOINT_DECODER = str(Path("checkpointDecoder").resolve())

# Finetuned model checkpoint paths
FINETUNED_ENCODER = str(Path("../tempData/singleStepEncoder").resolve())
FINETUNED_BACKBONE = str(Path("../tempData/singleStepBackbone").resolve())
FINETUNED_DECODER = str(Path("../tempData/singleStepDecoder").resolve())

# Output paths
METRICS_CSV_PATH = "all_metrics.csv"
PLOT_OUTPUT_PATH = "outputs/differenceOneStepTrained.png"

# Model parameters
STEPS = 2
RANDOM_SEED = 0
TIME_INDEX = 2
LEVEL_INDEX = 0
USE_FINETUNED_MODEL = False  # Set to True to use finetuned model checkpoints


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_weighted_rmse(pred, batch_true):
    """Compute latitude-weighted RMSE for surface and atmospheric variables

    Args:
        pred (Batch): Prediction batch.
        batch_true (Batch): Ground truth batch.

    Returns:
        tuple: (surf_rmse, atmos_rmse, surf_mae, atmos_mae) dictionaries
    """
    # Compute surface variable metrics
    surf_rmse = {}
    surf_mae = {}
    for key in pred.surf_vars:
        pred_var = pred.surf_vars[key][0, 0]
        true_var = batch_true.surf_vars[key][0, 0]
        surf_rmse[key] = weighted_rmse(pred_var, true_var, pred.metadata.lat)
        surf_mae[key] = weighted_mae(pred_var, true_var, pred.metadata.lat)

    # Compute atmospheric variable metrics
    atmos_rmse = {}
    atmos_mae = {}
    for key in pred.atmos_vars:
        for l_idx, level in enumerate(pred.metadata.atmos_levels):
            pred_var = pred.atmos_vars[key][0, 0, l_idx]
            true_var = batch_true.atmos_vars[key][0, 0, l_idx]
            atmos_rmse[f"{key}_{level}"] = weighted_rmse(pred_var, true_var, pred.metadata.lat)
            atmos_mae[f"{key}_{level}"] = weighted_mae(pred_var, true_var, pred.metadata.lat)

    # Save metrics to CSV
    _save_metrics_to_csv(surf_rmse, surf_mae, atmos_rmse, atmos_mae)

    return surf_rmse, atmos_rmse, surf_mae, atmos_mae


def _save_metrics_to_csv(surf_rmse, surf_mae, atmos_rmse, atmos_mae):
    """Save computed metrics to CSV file"""
    rows = []
    
    # Add surface variable metrics
    for k in surf_rmse:
        rows.append({"variable": k, "rmse": surf_rmse[k], "mae": surf_mae[k]})

    # Add atmospheric variable metrics
    for k in atmos_rmse:
        rows.append({"variable": k, "rmse": atmos_rmse[k], "mae": atmos_mae[k]})

    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_CSV_PATH, index=False)
    print(f"All metrics saved to {METRICS_CSV_PATH!r}")


def plot_all_vars(
    batch, t_idx: int = 1, level_idx: int = 0, out_path: str = "outputs/all_vars.png"
):
    """
    Plot every variable in batch.surf_vars (row 0) and batch.atmos_vars (row 1)
    at time-step t_idx, showing only pressure level level_idx for atmos_vars.
    Saves to out_path.
    """
    lons = np.asarray(batch.metadata.lon)
    lats = np.asarray(batch.metadata.lat)
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    aspect = (lons.max() - lons.min()) / (lats.max() - lats.min())

    surf_keys = list(batch.surf_vars.keys())
    atmos_keys = list(batch.atmos_vars.keys())
    n_surf, n_atmos = len(surf_keys), len(atmos_keys)
    n_cols = max(n_surf, n_atmos)

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8), constrained_layout=True)

    def to_np(x):
        return np.asarray(x)

    # Plot surface variables
    for j, name in enumerate(surf_keys):
        data = to_np(batch.surf_vars[name][0, t_idx])
        ax = axes[0, j]
        im = ax.imshow(
            data,
            origin="upper",
            extent=extent,
            aspect=aspect,
        )
        ax.set_title(f"{name}")
        ax.axis("off")
    for j in range(n_surf, n_cols):
        axes[0, j].axis("off")

    # Plot atmospheric variables
    levels = batch.metadata.atmos_levels
    for j, name in enumerate(atmos_keys):
        arr = to_np(batch.atmos_vars[name][0, t_idx])
        data = arr[level_idx]
        ax = axes[1, j]
        im = ax.imshow(
            data,
            origin="upper",  # north is up
            extent=extent,
            aspect=aspect,
        )
        ax.set_title(f"{name} @ {levels[level_idx]} hPa")
        ax.axis("off")
    for j in range(n_atmos, n_cols):
        axes[1, j].axis("off")

    # Add colorbar
    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.04,
        pad=0.02,
        label="variable value",
    )

    # Save figure
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out_p}")
    plt.show()


# =============================================================================
# MODEL SETUP AND LOADING
# =============================================================================

def load_model_checkpoints(use_finetuned=False):
    """Load model checkpoints from specified paths
    
    Args:
        use_finetuned (bool): Whether to use finetuned model checkpoints
        
    Returns:
        dict: Dictionary containing encoder, backbone, and decoder parameters
    """
    if use_finetuned:
        # Load finetuned model checkpoints
        params_encoder = ocp.StandardCheckpointer().restore(FINETUNED_ENCODER)
        params_backbone = ocp.StandardCheckpointer().restore(FINETUNED_BACKBONE)
        params_decoder = ocp.StandardCheckpointer().restore(FINETUNED_DECODER)
    else:
        # Load original model checkpoints
        params_encoder = ocp.StandardCheckpointer().restore(CHECKPOINT_ENCODER)["encoder"]
        params_backbone = ocp.StandardCheckpointer().restore(CHECKPOINT_BACKBONE)["backbone"]
        params_decoder = ocp.StandardCheckpointer().restore(CHECKPOINT_DECODER)["decoder"]

    params = {
        "encoder": params_encoder,
        "backbone": params_backbone,
        "decoder": params_decoder,
    }
    
    return params


def prepare_batch_for_model(batch, model, device_platform="gpu"):
    """Prepare batch for model processing
    
    Args:
        batch: Input batch
        model: Aurora model
        device_platform (str): Device platform to use
        
    Returns:
        Batch: Prepared batch
    """
    # Get device and prepare batch
    device = jax.devices(device_platform)[0]
    p = tree_leaves(batch)[0] if hasattr(batch, '__iter__') else None
    
    if p is not None:
        batch = batch.type(p.dtype)
    
    batch = batch.crop(model.patch_size)
    batch = batch.to(device)
    
    return batch


# =============================================================================
# MAIN EXECUTION FLOW
# =============================================================================

def main():
    """Main execution function"""
    print("Starting Aurora HRES evaluation...")
    
    # Load static variables dataset
    print("Loading static variables dataset...")
    static_vars_ds = xr.open_dataset(DATASET_PATH / "static.nc", engine="netcdf4")
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_ds = HresT0SequenceDataset(ZARR_PATH, mode="val", shuffle=False, seed=RANDOM_SEED, steps=STEPS)
    batch, out_batch_list = next(iter(val_ds))
    
    # Initialize model
    print("Initializing model...")
    model = AuroraSmall(use_lora=False)
    
    # Load model parameters
    print("Loading model checkpoints...")
    params = load_model_checkpoints(use_finetuned=USE_FINETUNED_MODEL)
    params = jax.device_put(params, device=jax.devices("gpu")[0])
    
    # Prepare batch for model
    print("Preparing batch for model...")
    rng = jax.random.PRNGKey(RANDOM_SEED)
    p = tree_leaves(params)[0]
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(jax.devices(p.device.platform)[p.device.host_id])
    
    # Run model predictions
    print("Running model predictions...")
    preds = [
        pred.to(jax.devices("cpu")[0])
        for pred in rollout(model, batch, steps=STEPS, params=params, training=False, rng=rng)
    ]
    
    # Prepare ground truth batches
    print("Preparing ground truth batches...")
    out_batch_list[0] = out_batch_list[0].crop(model.patch_size)
    out_batch_list[1] = out_batch_list[1].crop(model.patch_size)
    
    # Move parameters to CPU for final processing
    params = jax.device_put(params, device=jax.devices("cpu")[0])
    
    # Compute metrics
    print("Computing weighted RMSE and MAE...")
    surf_rmse, atmos_rmse, surf_mae, atmos_mae = compute_weighted_rmse(preds[0], out_batch_list[0])
    
    # Save results
    print("Saving results...")
    save_batch_npz(out_batch_list[0], OUTPUT_FOLDER, "truth value")
    save_batch_npz(preds[0], OUTPUT_FOLDER, "jax_values_hres")
    
    # Generate plots
    print("Generating plots...")
    plot_all_vars(preds[0], t_idx=TIME_INDEX, level_idx=LEVEL_INDEX, out_path=PLOT_OUTPUT_PATH)
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
