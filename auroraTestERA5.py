from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
import xarray as xr
from jax.tree_util import tree_leaves

from aurora import AuroraSmall, Batch, Metadata, rollout
from aurora.score import mae_loss_fn, weighted_mae, weighted_rmse


# =============================================================================
# CONFIGURATION AND PATHS
# =============================================================================

# Dataset paths
DATASET_PATH = Path("dataset")
STATIC_DATA_FILE = "static.nc"
SURFACE_DATA_FILE = "2023-01-01-surface-level.nc"
ATMOSPHERIC_DATA_FILE = "2023-01-01-atmospheric.nc"

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
PLOT_OUTPUT_PATH = "outputs/difference.png"

# Model parameters
STEPS = 2
RANDOM_SEED = 0
TIME_INDEX = 2
LEVEL_INDEX = 0
GAMMA = 0.5
USE_FINETUNED_MODEL = True  # Set to True to use finetuned model checkpoints

# Data selection parameters
DATA_TIME_INDEX = 1  # Select this time index in the downloaded data

# Surface weights for loss computation
SURFACE_WEIGHTS = {"2t": 1.0, "10u": 1.0, "10v": 1.0, "msl": 1.0}

# Atmospheric weights for loss computation (13 levels)
ATMOSPHERIC_WEIGHTS = {
    "t": jnp.ones(13) * 0.2,
    "u": jnp.ones(13) * 0.2,
    "v": jnp.ones(13) * 0.2,
    "q": jnp.ones(13) * 0.2,
    "z": jnp.ones(13) * 0.2,
}


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
# DATA LOADING AND BATCH CREATION
# =============================================================================

def load_datasets():
    """Load static, surface, and atmospheric datasets from netCDF files
    
    Returns:
        tuple: (static_vars_ds, surf_vars_ds, atmos_vars_ds) datasets
    """
    static_vars_ds = xr.open_dataset(DATASET_PATH / STATIC_DATA_FILE, engine="netcdf4")
    surf_vars_ds = xr.open_dataset(DATASET_PATH / SURFACE_DATA_FILE, engine="netcdf4")
    atmos_vars_ds = xr.open_dataset(DATASET_PATH / ATMOSPHERIC_DATA_FILE, engine="netcdf4")
    
    return static_vars_ds, surf_vars_ds, atmos_vars_ds


def create_input_batch(static_vars_ds, surf_vars_ds, atmos_vars_ds, i=None):
    """Create input batch for model inference
    
    Args:
        static_vars_ds: Static variables dataset
        surf_vars_ds: Surface variables dataset  
        atmos_vars_ds: Atmospheric variables dataset
        i (int): Time index to use (defaults to DATA_TIME_INDEX)
        
    Returns:
        Batch: Input batch for model
    """
    if i is None:
        i = DATA_TIME_INDEX
        
    batch = Batch(
        surf_vars={
            # First select time points `i` and `i - 1`. Afterwards, `[None]` inserts a
            # batch dimension of size one.
            "2t": jnp.array(surf_vars_ds["t2m"].values[[i - 1, i]][None]),
            "10u": jnp.array(surf_vars_ds["u10"].values[[i - 1, i]][None]),
            "10v": jnp.array(surf_vars_ds["v10"].values[[i - 1, i]][None]),
            "msl": jnp.array(surf_vars_ds["msl"].values[[i - 1, i]][None]),
        },
        static_vars={
            # The static variables are constant, so we just get them for the first time.
            "z": jnp.array(static_vars_ds["z"].values[0]),
            "slt": jnp.array(static_vars_ds["slt"].values[0]),
            "lsm": jnp.array(static_vars_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": jnp.array(atmos_vars_ds["t"].values[[i - 1, i]][None]),
            "u": jnp.array(atmos_vars_ds["u"].values[[i - 1, i]][None]),
            "v": jnp.array(atmos_vars_ds["v"].values[[i - 1, i]][None]),
            "q": jnp.array(atmos_vars_ds["q"].values[[i - 1, i]][None]),
            "z": jnp.array(atmos_vars_ds["z"].values[[i - 1, i]][None]),
        },
        metadata=Metadata(
            lat=jnp.array(surf_vars_ds.latitude.values, dtype=jnp.float32),
            lon=jnp.array(surf_vars_ds.longitude.values, dtype=jnp.float32),
            # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
            # `datetime.datetime`s. Note that this needs to be a tuple of length one:
            # one value for every batch element.
            time=(
                jnp.array(
                    surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[i].timestamp(),
                    dtype=jnp.int64,
                ),
            ),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
        ),
    )
    
    return batch


def create_ground_truth_batch(static_vars_ds, surf_vars_ds, atmos_vars_ds, t_idx=None):
    """Create ground truth batch for evaluation
    
    Args:
        static_vars_ds: Static variables dataset
        surf_vars_ds: Surface variables dataset
        atmos_vars_ds: Atmospheric variables dataset
        t_idx (int): Time index to use (defaults to TIME_INDEX)
        
    Returns:
        Batch: Ground truth batch
    """
    if t_idx is None:
        t_idx = TIME_INDEX
        
    batch_true = Batch(
        surf_vars={
            "2t": jnp.array(surf_vars_ds["t2m"].values[[t_idx]][None]),
            "10u": jnp.array(surf_vars_ds["u10"].values[[t_idx]][None]),
            "10v": jnp.array(surf_vars_ds["v10"].values[[t_idx]][None]),
            "msl": jnp.array(surf_vars_ds["msl"].values[[t_idx]][None]),
        },
        static_vars={
            "z": jnp.array(static_vars_ds["z"].values[0]),
            "slt": jnp.array(static_vars_ds["slt"].values[0]),
            "lsm": jnp.array(static_vars_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": jnp.array(atmos_vars_ds["t"].values[[t_idx]][None]),
            "u": jnp.array(atmos_vars_ds["u"].values[[t_idx]][None]),
            "v": jnp.array(atmos_vars_ds["v"].values[[t_idx]][None]),
            "q": jnp.array(atmos_vars_ds["q"].values[[t_idx]][None]),
            "z": jnp.array(atmos_vars_ds["z"].values[[t_idx]][None]),
        },
        metadata=Metadata(
            lat=jnp.array(surf_vars_ds.latitude.values),
            lon=jnp.array(surf_vars_ds.longitude.values),
            time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[t_idx],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
        ),
    )
    
    return batch_true


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
        params_encoder = ocp.StandardCheckpointer().restore(CHECKPOINT_ENCODER)
        params_backbone = ocp.StandardCheckpointer().restore(CHECKPOINT_BACKBONE)
        params_decoder = ocp.StandardCheckpointer().restore(CHECKPOINT_DECODER)
        
        # Extract the nested parameters for original checkpoints
        params_encoder = params_encoder["encoder"]
        params_backbone = params_backbone["backbone"]
        params_decoder = params_decoder["decoder"]

    params = {
        "encoder": params_encoder,
        "backbone": params_backbone,
        "decoder": params_decoder,
    }
    
    return params


def prepare_batch_for_model(batch, params, model):
    """Prepare batch for model processing
    
    Args:
        batch: Input batch
        params: Model parameters
        model: Aurora model
        
    Returns:
        Batch: Prepared batch
    """
    p = tree_leaves(params)[0]
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(jax.devices(p.device.platform)[p.device.host_id])
    
    return batch


# =============================================================================
# MAIN EXECUTION FLOW
# =============================================================================

def main():
    """Main execution function"""
    print("Starting Aurora ERA5 evaluation...")
    
    # Load datasets
    print("Loading datasets...")
    static_vars_ds, surf_vars_ds, atmos_vars_ds = load_datasets()
    
    # Create input batch
    print("Creating input batch...")
    batch = create_input_batch(static_vars_ds, surf_vars_ds, atmos_vars_ds)
    
    # Initialize model
    print("Initializing model...")
    model = AuroraSmall(use_lora=False)
    
    # Load model parameters
    print("Loading model checkpoints...")
    params = load_model_checkpoints(use_finetuned=USE_FINETUNED_MODEL)
    params = jax.device_put(params, device=jax.devices("gpu")[0])
    
    # Prepare batch for model
    print("Preparing batch for model...")
    batch = prepare_batch_for_model(batch, params, model)
    
    # Run model predictions
    print("Running model predictions...")
    rng = jax.random.PRNGKey(RANDOM_SEED)
    preds = [
        pred.to(jax.devices("cpu")[0])
        for pred in rollout(model, batch, steps=STEPS, params=params, training=False, rng=rng)
    ]
    
    # Move parameters to CPU for final processing
    params = jax.device_put(params, device=jax.devices("cpu")[0])
    
    # Create ground truth batch
    print("Creating ground truth batch...")
    batch_true = create_ground_truth_batch(static_vars_ds, surf_vars_ds, atmos_vars_ds)
    batch_true = batch_true.crop(model.patch_size)
    
    # Compute loss
    print("Computing loss...")
    loss = mae_loss_fn(preds[0], batch_true, SURFACE_WEIGHTS, ATMOSPHERIC_WEIGHTS, gamma=GAMMA)
    print(f"Loss: {loss:.2f}")
    
    # Compute metrics
    print("Computing weighted RMSE and MAE...")
    surf_rmse, atmos_rmse, surf_mae, atmos_mae = compute_weighted_rmse(preds[0], batch_true)
    
    # Generate plots
    print("Generating plots...")
    plot_all_vars(preds[0], t_idx=TIME_INDEX, level_idx=LEVEL_INDEX, out_path=PLOT_OUTPUT_PATH)
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
