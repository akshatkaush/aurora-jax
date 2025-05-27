from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
import pandas as pd
import xarray as xr
from jax.tree_util import tree_leaves
from orbax.checkpoint import utils as ocp_utils

from aurora import AuroraSmall, Batch, Metadata, rollout
from aurora.score import mae_loss_fn, weighted_mae, weighted_rmse
from aurora.IterableDataset import HresT0SequenceDataset
from save_batches import save_batch_npz


def compute_weighted_rmse(pred, batch_true):
    """Compute latitude-weighted RMSE for surface and atmospheric variables

    Args:
        pred (Batch): Prediction batch.
        batch (Batch): Ground truth batch.

    Returns:
        rmse: Latitude weighted root mean squared error
    """
    surf_rmse = {}
    surf_mae = {}
    for key in pred.surf_vars:
        pred_var = pred.surf_vars[key][0, 0]
        true_var = batch_true.surf_vars[key][0, 0]
        surf_rmse[key] = weighted_rmse(pred_var, true_var, pred.metadata.lat)
        surf_mae[key] = weighted_mae(pred_var, true_var, pred.metadata.lat)

    atmos_rmse = {}
    atmos_mae = {}
    for key in pred.atmos_vars:
        for l_idx, level in enumerate(pred.metadata.atmos_levels):
            pred_var = pred.atmos_vars[key][0, 0, l_idx]
            true_var = batch_true.atmos_vars[key][0, 0, l_idx]
            atmos_rmse[f"{key}_{level}"] = weighted_rmse(pred_var, true_var, pred.metadata.lat)
            atmos_mae[f"{key}_{level}"] = weighted_mae(pred_var, true_var, pred.metadata.lat)

    # print("RMSE for surface variables:")
    # for key, rmse in surf_rmse.items():
    #     print(f"  {key}: {rmse:.2f}")
    # print("RMSE for atmospheric variables:")
    # for key, rmse in atmos_rmse.items():
    #     print(f"  {key}: {rmse:.2f}")

    rows = []
    for k in surf_rmse:
        rows.append({"variable": k, "rmse": surf_rmse[k], "mae": surf_mae[k]})

    for k in atmos_rmse:
        rows.append({"variable": k, "rmse": atmos_rmse[k], "mae": atmos_mae[k]})

    csv_path = "all_metrics.csv"
    df = pd.DataFrame(rows, columns=["variable", "rmse", "mae"])
    df.to_csv(csv_path, index=False)
    print(f"All metrics saved to {csv_path!r}")

    return surf_rmse, atmos_rmse, surf_mae, atmos_mae


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

    fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.04,
        pad=0.02,
        label="variable value",
    )

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {out_p}")
    plt.show()


download_path = Path("datasetEnviousScratch")
static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")

zarr_path = "/home1/a/akaush/aurora/hresDataset/hres_t0_2021-2022mid.zarr"

val_ds = HresT0SequenceDataset(
    zarr_path,
    mode="val", 
    shuffle=False,    
    seed=0,          
    steps=2
)

batch, out_batch_list = next(iter(val_ds))

model = AuroraSmall(use_lora=False)

key = jax.random.key(0)

# params_encoder = ocp.StandardCheckpointer().restore("/home1/a/akaush/aurora/checkpoints")
# params_backbone = ocp.StandardCheckpointer().restore(
#     "/home1/a/akaush/aurora/checkpointsTillBackbone"
# )
# params_decoder = ocp.StandardCheckpointer().restore("/home1/a/akaush/aurora/checkpointsTillDecoder")
params_encoder = ocp.StandardCheckpointer().restore("/home1/a/akaush/tempData/singleStepEncoder")
params_backbone = ocp.StandardCheckpointer().restore(
    "/home1/a/akaush/tempData/singleStepBackbone"
)
params_decoder = ocp.StandardCheckpointer().restore("/home1/a/akaush/tempData/singleStepDecoder")

params = {
    "encoder": params_encoder['encoder'],
    "backbone": params_backbone['backbone'],
    "decoder": params_decoder['decoder'],
}
params = jax.device_put(params, device=jax.devices("gpu")[0])

rng = jax.random.PRNGKey(0)
p = tree_leaves(params)[0]
batch = batch.type(p.dtype)
batch = batch.crop(model.patch_size)
batch = batch.to(jax.devices(p.device.platform)[p.device.host_id])

preds = [
    pred.to(jax.devices("cpu")[0])
    for pred in rollout(model, batch, steps=2, params=params, training=False, rng=rng)
]

out_batch_list[0]= out_batch_list[0].crop(model.patch_size)
out_batch_list[1]= out_batch_list[1].crop(model.patch_size)

params = jax.device_put(params, device=jax.devices("cpu")[0])

t_idx = 2

# batch_true = batch_true.crop(model.patch_size)
# surf_weights = {"2t": 1.0, "10u": 1.0, "10v": 1.0, "msl": 1.0}
# atmos_weights = {
#     "t": jnp.ones(13) * 0.2,
#     "u": jnp.ones(13) * 0.2,
#     "v": jnp.ones(13) * 0.2,
#     "q": jnp.ones(13) * 0.2,
#     "z": jnp.ones(13) * 0.2,
# }
# # loss_fn = jax.jit(mae_loss_fn)
# loss_fn = mae_loss_fn
# loss = loss_fn(preds[0], batch_true, surf_weights, atmos_weights, gamma=0.5)
# print(f"Loss: {loss:.2f}")
surf_rmse, atmos_rmse, surf_mae, atmos_mae = compute_weighted_rmse(preds[0], out_batch_list[0])

output_folder = '../tempData'
save_batch_npz(out_batch_list[0], output_folder, "truth value")
plot_all_vars(preds[0], t_idx=2, level_idx=0, out_path="outputs/differenceTwoStepTrained.png")
