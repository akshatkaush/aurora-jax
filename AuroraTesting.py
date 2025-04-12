from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import xarray as xr

from aurora import AuroraSmall, Batch, Metadata, rollout

download_path = Path("dataset")
static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
surf_vars_ds = xr.open_dataset(download_path / "2023-01-01-surface-level.nc", engine="netcdf4")
atmos_vars_ds = xr.open_dataset(download_path / "2023-01-01-atmospheric.nc", engine="netcdf4")

i = 1  # Select this time index in the downloaded data.
jax.config.update("jax_enable_x64", True)

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
        lat=jnp.array(surf_vars_ds.latitude.values),
        lon=jnp.array(surf_vars_ds.longitude.values),
        # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
        # `datetime.datetime`s. Note that this needs to be a tuple of length one:
        # one value for every batch element.
        time=(
            jnp.array(
                surf_vars_ds.valid_time.values.astype("datetime64[s]").astype(float)[i],
                dtype=jnp.float64,
            ),
        ),
        atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
    ),
)
model = AuroraSmall(use_lora=False)
rng = jax.random.PRNGKey(0)
params_encoder = ocp.StandardCheckpointer().restore("/home1/a/akaush/aurora/checkpoints")

params_backbone = ocp.StandardCheckpointer().restore(
    "/home1/a/akaush/aurora/checkpointsTillBackbone"
)

params = {
    "encoder": params_encoder["encoder"],
    "backbone": params_backbone["backbone"],
}
params = jax.device_put(params, device=jax.devices("gpu")[0])

preds = [
    pred.to("cpu")
    for pred in rollout(model, batch, steps=2, params=params, training=False, rng=rng)
]

params = jax.device_put(params, device=jax.devices("cpu")[0])

# fig, ax = plt.subplots(2, 2, figsize=(12, 6.5))

# for i in range(ax.shape[0]):
#     pred = preds[i]

#     ax[i, 0].imshow(pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50)
#     ax[i, 0].set_ylabel(str(pred.metadata.time[0]))
#     if i == 0:
#         ax[i, 0].set_title("Aurora Prediction")
#     ax[i, 0].set_xticks([])
#     ax[i, 0].set_yticks([])

#     ax[i, 1].imshow(surf_vars_ds["t2m"][2 + i].values - 273.15, vmin=-50, vmax=50)
#     if i == 0:
#         ax[i, 1].set_title("ERA5")
#     ax[i, 1].set_xticks([])
#     ax[i, 1].set_yticks([])
#     plt.tight_layout()
#     plt.savefig("aurora_comparison.png", bbox_inches="tight", dpi=300)
#     plt.close()


# # import jax
# # print(jax.default_backend())  # Should print 'gpu'
# # print(jax.devices())
