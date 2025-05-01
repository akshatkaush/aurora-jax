from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
import xarray as xr
from jax.tree_util import tree_leaves

from aurora import AuroraSmall, Batch, Metadata, rollout


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
surf_vars_ds = xr.open_dataset(download_path / "2023-01-01-surface-level.nc", engine="netcdf4")
atmos_vars_ds = xr.open_dataset(download_path / "2023-01-01-atmospheric.nc", engine="netcdf4")

i = 1  # Select this time index in the downloaded data.

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
                surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[1].timestamp(),
                dtype=jnp.int64,
            ),
        ),
        atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
    ),
)
model = AuroraSmall(use_lora=False)

params_encoder = ocp.StandardCheckpointer().restore("/home1/a/akaush/aurora/checkpoints")
params_backbone = ocp.StandardCheckpointer().restore(
    "/home1/a/akaush/aurora/checkpointsTillBackbone"
)
params_decoder = ocp.StandardCheckpointer().restore("/home1/a/akaush/aurora/checkpointsTillDecoder")
params = {
    "encoder": params_encoder["encoder"],
    "backbone": params_backbone["backbone"],
    "decoder": params_decoder["decoder"],
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

params = jax.device_put(params, device=jax.devices("cpu")[0])

plot_all_vars(preds[1], t_idx=0, level_idx=0, out_path="outputs/all_vars_jax.png")
