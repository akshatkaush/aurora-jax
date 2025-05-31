from pathlib import Path

import jax
import jax.numpy as jnp
import xarray as xr

from aurora import AuroraSmall, Batch, Metadata


def assign_pt_params_to_jax_backbone(jax_params, pt_params):
    """
    Assign PyTorch parameters to JAX parameters with appropriate transposition.
    Only processes parameters that start with 'backbone' and returns only the backbone parameters.

    Args:
        jax_params: JAX model parameters
        pt_params: PyTorch model parameters

    Returns:
        Updated JAX backbone parameters with values from PyTorch parameters
    """
    # Extract the backbone part of the JAX parameters
    backbone_params = jax_params["decoder"]

    # Flatten the backbone parameters (without paths)
    backbone_leaves, backbone_treedef = jax.tree_util.tree_flatten(backbone_params)

    # Get the PyTorch backbone parameters
    pt_backbone_params = {k: v for k, v in pt_params.items() if k.startswith("decoder")}
    pt_backbone_keys = sorted(pt_backbone_params.keys())

    # Convert PyTorch parameters to numpy arrays
    pt_arrays = []
    for key in pt_backbone_keys:
        param = pt_backbone_params[key]
        pt_array = param
        pt_arrays.append(pt_array)

    # Update JAX leaves with PyTorch values
    updated_leaves = []
    for _, (leaf, pt_array) in enumerate(zip(backbone_leaves, pt_arrays)):
        # Check if transposition is needed (2D case)
        if len(leaf.shape) == 2 and len(pt_array.shape) == 2:
            # Transpose the parameter
            pt_array = pt_array.T

        # Create a new JAX array with the PyTorch values
        updated_leaf = jax.numpy.array(pt_array, dtype=leaf.dtype)
        updated_leaves.append(updated_leaf)

    # Reconstruct the backbone parameters with updated leaves
    updated_backbone_params = jax.tree_util.tree_unflatten(backbone_treedef, updated_leaves)
    updated_params = {"decoder": updated_backbone_params}
    return updated_params


def extract_jax_params(template_params):
    """Extract a dictionary mapping formatted key path to (shape, dtype) for JAX params."""
    jax_leaves, _ = jax.tree_util.tree_flatten_with_path(template_params)
    jax_dict = {}
    # Sort keys by converting the tuple to a string.
    sorted_leaves = sorted(jax_leaves, key=lambda x: "/".join(str(key) for key in x[0]))
    for path, leaf in sorted_leaves:
        # Format the path as "['key1']/['key2']/..."
        key_str = "/".join(f"['{key}']" for key in path)
        jax_dict[key_str] = (leaf.shape, str(leaf.dtype))
    return jax_dict


def extract_pt_params(params):
    """Extract a dictionary mapping formatted key path to (shape, dtype) for PyTorch params."""
    pt_dict = {}
    for key in sorted(params.keys()):
        param = params[key]
        # Split the dot-separated key and format it as "['part']"
        key_parts = key.split(".")
        key_str = "/".join(f"['{k}']" for k in key_parts)
        # Remove the "torch." prefix from dtype string.
        pt_dict[key_str] = (tuple(param.shape), str(param.dtype).replace("torch.", ""))
    return pt_dict


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


model = AuroraSmall(use_lora=True)
rng = jax.random.PRNGKey(0)

variables = model.init(rng, batch, training=False, rng=rng)
template_params = variables["params"]
params = model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")

# params = {key: value for key, value in params.items() if key.startswith("encoder")}
# print("JAX parameters:")
# for key, (shape, dtype) in extract_jax_params(template_params).items():
#     print(f"Path: {key}, Shape: {shape}, dtype: {dtype}")

print("\nPyTorch parameters:")
for key, (shape, dtype) in extract_pt_params(params).items():
    print(f"Path: {key}, Shape: {shape}, dtype: {dtype}")

# final_backbone_params = assign_pt_params_to_jax_backbone(template_params, params)
# assert jax.tree_structure(final_backbone_params["decoder"]) == jax.tree_structure(
#     template_params["decoder"]
# )

# assert jax.tree_util.tree_all=p[(
#     jax.tree_map(
#         lambda x, y: x.shape == y.shape,
#         final_backbone_params["decoder"],
#         template_params["decoder"],
#     )
# )


# checkpointer = ocp.PyTreeCheckpointer()
# save_args = orbax_utils.save_args_from_target(final_backbone_params)
# checkpointer.save(
#     "/home1/a/akaush/aurora/checkpointsTillDecoder",
#     final_backbone_params,
#     save_args=save_args,
#     force=True,
# )
