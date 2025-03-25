from pathlib import Path

import jax
import jax.numpy as jnp
import xarray as xr

from aurora import AuroraSmall, Batch, Metadata, rollout


def transform_key(key):
    """
    Given a dot-separated key from `params`, split it into parts and rename segments
    to match the structure of template_params.
    """
    parts = key.split(".")
    new_parts = []
    i = 0
    while i < len(parts):
        part = parts[i]
        # Group layers if encountered, e.g., "layers", "0", "1" -> "layers_0_1"
        if (
            part == "layers"
            and i + 2 < len(parts)
            and parts[i + 1].isdigit()
            and parts[i + 2].isdigit()
        ):
            new_parts.append(f"layers_{parts[i+1]}_{parts[i+2]}")
            i += 3
            continue
        # Rename "net.0" to "Dense_0" and "net.2" to "Dense_1"
        if part == "net" and i + 1 < len(parts) and parts[i + 1] in {"0", "2"}:
            new_parts.append("Dense_0" if parts[i + 1] == "0" else "Dense_1")
            i += 2
            continue
        # Combine "weights" with the following segment, e.g., "weights", "q" -> "weights_q"
        if part == "weights" and i + 1 < len(parts):
            new_parts.append("weights_" + parts[i + 1])
            i += 2
            continue
        # For the final segment: change "weight" to "kernel"
        if i == len(parts) - 1:
            if part == "weight":
                new_parts.append("kernel")
            else:
                new_parts.append(part)
            i += 1
            continue

        new_parts.append(part)
        i += 1

    return new_parts


def unflatten_and_transform(params):
    """
    Converts the flat dictionary `params` (with dot-separated keys)
    into a nested dictionary (pytree) matching the template_params structure.
    Only keys that start with 'encoder' are kept.

    For keys that become 'kernel' (transformed from 'weight'),
    the value is transposed (if it is a 2D array) to convert from the original
    convention to the template convention.
    """
    new_dict = {}
    for flat_key, value in params.items():
        # Only process keys that start with 'encoder'
        if not flat_key.startswith("encoder"):
            continue

        new_key_parts = transform_key(flat_key)
        d = new_dict
        for part in new_key_parts[:-1]:
            d = d.setdefault(part, {})

        # If the key is now "kernel" and the value is a 2D jax array, transpose it.
        if new_key_parts[-1] == "kernel" and hasattr(value, "ndim") and value.ndim == 2:
            value = jnp.transpose(value)

        d[new_key_parts[-1]] = value
    new_dict["encoder"]["level_agg"]["layers_0_2"]["scale"] = new_dict["encoder"]["level_agg"][
        "layers_0_2"
    ].pop("kernel")
    new_dict["encoder"]["level_agg"]["layers_0_3"]["scale"] = new_dict["encoder"]["level_agg"][
        "layers_0_3"
    ].pop("kernel")
    new_dict["encoder"]["surf_norm"]["scale"] = new_dict["encoder"]["surf_norm"].pop("kernel")
    return new_dict


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
        time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[i],),
        atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
    ),
)


model = AuroraSmall(use_lora=False)
rng = jax.random.PRNGKey(0)

variables = model.init(rng, batch)
template_params = variables["params"]
params = model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
params = {key: value for key, value in params.items() if key.startswith("encoder")}

params = unflatten_and_transform(params)
# jnp.save('model_params.npy', result)

assert jax.tree_structure(template_params) == jax.tree_structure(params)

assert jax.tree_util.tree_all(
    jax.tree_map(lambda x, y: x.shape == y.shape, template_params, params)
)

params = jax.device_put(params, device=jax.devices("gpu")[0])

# ploting the predictions
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
