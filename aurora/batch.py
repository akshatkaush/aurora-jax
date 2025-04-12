"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from functools import partial
from pathlib import Path
from typing import Callable, List

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from jax import device_put
from scipy.interpolate import RegularGridInterpolator as RGI
from typing_extensions import dataclass_transform

from aurora.normalisation import (
    normalise_atmos_var,
    normalise_surf_var,
    unnormalise_atmos_var,
    unnormalise_surf_var,
)

__all__ = ["Metadata", "Batch"]


@dataclass_transform(field_specifiers=(jdc.Static,))
def pytree_dataclass(cls):
    return jdc.pytree_dataclass(cls)


@pytree_dataclass
class Metadata:
    lat: jnp.ndarray
    lon: jnp.ndarray
    time: jdc.Static[tuple[float, ...]]
    atmos_levels: tuple[int | float, ...]
    rollout_step: int = 0


@pytree_dataclass
class Batch:
    """A batch of data.

    Args:
        surf_vars (dict[str, :class:`torch.Tensor`]): Surface-level variables with shape
            `(b, t, h, w)`.
        static_vars (dict[str, :class:`torch.Tensor`]): Static variables with shape `(h, w)`.
        atmos_vars (dict[str, :class:`torch.Tensor`]): Atmospheric variables with shape
            `(b, t, c, h, w)`.
        metadata (:class:`Metadata`): Metadata associated to this batch.
    """

    surf_vars: dict[str, jnp.ndarray]
    static_vars: dict[str, jnp.ndarray]
    atmos_vars: dict[str, jnp.ndarray]
    metadata: jdc.Static[Metadata]

    # todo remove hardcode
    _surf_vars_order: jdc.Static[tuple[str, ...]] = ("2t", "10u", "10v", "msl")
    _static_vars_order: jdc.Static[tuple[str, ...]] = ("z", "slt", "lsm")
    _atmos_vars_order: jdc.Static[tuple[str, ...]] = ("t", "u", "v", "q", "z")

    # todo use aux in tree_flatten and tree_unflatten
    # Get ordered keys methods
    def surf_vars_ordered_keys(self):
        return self._surf_vars_order

    def static_vars_ordered_keys(self):
        return self._static_vars_order

    def atmos_vars_ordered_keys(self):
        return self._atmos_vars_order

    # Get ordered items methods
    def surf_vars_ordered_values(self):
        return [self.surf_vars[k] for k in self._surf_vars_order]

    def static_vars_ordered_values(self):
        return [self.static_vars[k] for k in self._static_vars_order]

    def atmos_vars_ordered_values(self):
        return [self.atmos_vars[k] for k in self._atmos_vars_order]

    @property
    def spatial_shape(self) -> tuple[int, int]:
        """Get the spatial shape from an arbitrary surface-level variable."""
        return next(iter(self.surf_vars.values())).shape[-2:]

    def normalise(self, surf_stats: dict[str, tuple[float, float]]) -> "Batch":
        """Normalise all variables in the batch.

        Args:
            surf_stats (dict[str, tuple[float, float]]): For these surface-level variables, adjust
                the normalisation to the given tuple consisting of a new location and scale.

        Returns:
            :class:`.Batch`: Normalised batch.
        """
        return Batch(
            surf_vars={
                k: normalise_surf_var(v, k, stats=surf_stats) for k, v in self.surf_vars.items()
            },
            static_vars={
                k: normalise_surf_var(v, k, stats=surf_stats) for k, v in self.static_vars.items()
            },
            atmos_vars={
                k: normalise_atmos_var(v, k, self.metadata.atmos_levels)
                for k, v in self.atmos_vars.items()
            },
            metadata=self.metadata,
        )

    def unnormalise(self, surf_stats: dict[str, tuple[float, float]]) -> "Batch":
        """Unnormalise all variables in the batch.

        Args:
            surf_stats (dict[str, tuple[float, float]]): For these surface-level variables, adjust
                the normalisation to the given tuple consisting of a new location and scale.

        Returns:
            :class:`.Batch`: Unnormalised batch.
        """
        return Batch(
            surf_vars={
                k: unnormalise_surf_var(v, k, stats=surf_stats) for k, v in self.surf_vars.items()
            },
            static_vars={
                k: unnormalise_surf_var(v, k, stats=surf_stats) for k, v in self.static_vars.items()
            },
            atmos_vars={
                k: unnormalise_atmos_var(v, k, self.metadata.atmos_levels)
                for k, v in self.atmos_vars.items()
            },
            metadata=self.metadata,
        )

    def crop(self, patch_size: int) -> "Batch":
        """Crop the variables in the batch to patch size `patch_size`."""
        h, w = self.spatial_shape

        if w % patch_size != 0:
            raise ValueError("Width of the data must be a multiple of the patch size.")

        if h % patch_size == 0:
            return self
        elif h % patch_size == 1:
            return Batch(
                surf_vars={k: v[..., :-1, :] for k, v in self.surf_vars.items()},
                static_vars={k: v[..., :-1, :] for k, v in self.static_vars.items()},
                atmos_vars={k: v[..., :-1, :] for k, v in self.atmos_vars.items()},
                metadata=Metadata(
                    lat=self.metadata.lat[:-1],
                    lon=self.metadata.lon,
                    atmos_levels=self.metadata.atmos_levels,
                    time=self.metadata.time,
                    rollout_step=self.metadata.rollout_step,
                ),
            )
        else:
            raise ValueError(
                f"There can at most be one latitude too many, "
                f"but there are {h % patch_size} too many."
            )

    def _fmap(self, f: Callable[[jnp.ndarray], jnp.ndarray]) -> "Batch":
        return Batch(
            surf_vars={k: f(v) for k, v in self.surf_vars.items()},
            static_vars={k: f(v) for k, v in self.static_vars.items()},
            atmos_vars={k: f(v) for k, v in self.atmos_vars.items()},
            metadata=Metadata(
                lat=f(self.metadata.lat),
                lon=f(self.metadata.lon),
                atmos_levels=self.metadata.atmos_levels,
                time=self.metadata.time,
                rollout_step=self.metadata.rollout_step,
            ),
        )

    def to(self, device: str) -> "Batch":
        """Move the batch to another device."""
        device_force = jax.devices("gpu")[0] if jax.devices("gpu") else jax.devices("cpu")[0]
        return self._fmap(lambda x: device_put(x, device_force))

    def type(self, dtype) -> "Batch":
        """Convert everything to type `dtype`."""
        return self._fmap(lambda x: x.astype(dtype))

    def regrid(self, res: float) -> "Batch":
        """Regrid the batch to a `res` degrees resolution.

        This results in `float32` data on the CPU.

        This function is not optimised for either speed or accuracy. Use at your own risk.
        """

        shape = (round(180 / res) + 1, round(360 / res))
        lat_new = jnp.linspace(90, -90, shape[0])
        lon_new = jnp.linspace(0, 360, shape[1], endpoint=False)
        interpolate_res = partial(
            interpolate,
            lat=self.metadata.lat,
            lon=self.metadata.lon,
            lat_new=lat_new,
            lon_new=lon_new,
        )

        return Batch(
            surf_vars={k: interpolate_res(v) for k, v in self.surf_vars.items()},
            static_vars={k: interpolate_res(v) for k, v in self.static_vars.items()},
            atmos_vars={k: interpolate_res(v) for k, v in self.atmos_vars.items()},
            metadata=Metadata(
                lat=lat_new,
                lon=lon_new,
                atmos_levels=self.metadata.atmos_levels,
                time=self.metadata.time,
                rollout_step=self.metadata.rollout_step,
            ),
        )

    def to_netcdf(self, path: str | Path) -> None:
        """Write the batch to a file.

        This requires `xarray` and `netcdf4` to be installed.
        """
        try:
            import xarray as xr
        except ImportError as e:
            raise RuntimeError("`xarray` must be installed.") from e

        ds = xr.Dataset(
            {
                **{
                    f"surf_{k}": (("batch", "history", "latitude", "longitude"), _np(v))
                    for k, v in self.surf_vars.items()
                },
                **{
                    f"static_{k}": (("latitude", "longitude"), _np(v))
                    for k, v in self.static_vars.items()
                },
                **{
                    f"atmos_{k}": (("batch", "history", "level", "latitude", "longitude"), _np(v))
                    for k, v in self.atmos_vars.items()
                },
            },
            coords={
                "latitude": _np(self.metadata.lat),
                "longitude": _np(self.metadata.lon),
                "time": list(self.metadata.time),
                "level": list(self.metadata.atmos_levels),
                "rollout_step": self.metadata.rollout_step,
            },
        )
        ds.to_netcdf(path)

    @classmethod
    def from_netcdf(cls, path: str | Path) -> "Batch":
        """Load a batch from a file."""
        try:
            import xarray as xr
        except ImportError as e:
            raise RuntimeError("`xarray` must be installed.") from e

        ds = xr.load_dataset(path, engine="netcdf4")

        surf_vars: List[str] = []
        static_vars: List[str] = []
        atmos_vars: List[str] = []

        for k in ds:
            if k.startswith("surf_"):
                surf_vars.append(k.removeprefix("surf_"))
            elif k.startswith("static_"):
                static_vars.append(k.removeprefix("static_"))
            elif k.startswith("atmos_"):
                atmos_vars.append(k.removeprefix("atmos_"))

        return Batch(
            surf_vars={k: jnp.array(ds[f"surf_{k}"].values) for k in surf_vars},
            static_vars={k: jnp.array(ds[f"static_{k}"].values) for k in static_vars},
            atmos_vars={k: jnp.array(ds[f"atmos_{k}"].values) for k in atmos_vars},
            metadata=Metadata(
                lat=jnp.array(ds.latitude.values),
                lon=jnp.array(ds.longitude.values),
                time=tuple(ds.time.values.astype("datetime64[s]").tolist()),
                atmos_levels=tuple(ds.level.values),
                rollout_step=int(ds.rollout_step.values),
            ),
        )


def _np(x: jnp.ndarray) -> np.ndarray:
    return jax.device_get(x)


def interpolate(
    v: jnp.ndarray,
    lat: jnp.ndarray,
    lon: jnp.ndarray,
    lat_new: jnp.ndarray,
    lon_new: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate a variable `v` with latitudes `lat` and longitudes `lon` to new latitudes
    `lat_new` and new longitudes `lon_new`."""
    # Perform the interpolation in double precision.
    v_np = np.asarray(v.astype(jnp.float32))
    lat_np = np.asarray(lat.astype(jnp.float32))
    lon_np = np.asarray(lon.astype(jnp.float32))
    lat_new_np = np.asarray(lat_new.astype(jnp.float32))
    lon_new_np = np.asarray(lon_new.astype(jnp.float32))

    interpolated = interpolate_numpy(
        v_np,
        lat_np,
        lon_np,
        lat_new_np,
        lon_new_np,
    )
    return jnp.array(interpolated).astype(jnp.float32)


def interpolate_numpy(
    v: jnp.ndarray,
    lat: jnp.ndarray,
    lon: jnp.ndarray,
    lat_new: jnp.ndarray,
    lon_new: jnp.ndarray,
) -> jnp.ndarray:
    """Like :func:`.interpolate`, but for NumPy tensors."""

    # Implement periodic longitudes in `lon`.
    assert (np.diff(lon) > 0).all()
    lon = np.concatenate((lon[-1:] - 360, lon, lon[:1] + 360))

    # Merge all batch dimensions into one.
    batch_shape = v.shape[:-2]
    v = v.reshape(-1, *v.shape[-2:])

    # Loop over all batch elements.
    vs_regridded = []
    for vi in v:
        # Implement periodic longitudes in `vi`.
        vi = np.concatenate((vi[:, -1:], vi, vi[:, :1]), axis=1)

        rgi = RGI(
            (lat, lon),
            vi,
            method="linear",
            bounds_error=False,  # Allow out of bounds, for the latitudes.
            fill_value=None,  # Extrapolate latitudes if they are out of bounds.
        )
        lat_new_grid, lon_new_grid = np.meshgrid(
            lat_new,
            lon_new,
            indexing="ij",
            sparse=True,
        )
        vs_regridded.append(rgi((lat_new_grid, lon_new_grid)))

    # Recreate the batch dimensions.
    v_regridded = np.stack(vs_regridded, axis=0)
    v_regridded = v_regridded.reshape(*batch_shape, lat_new.shape[0], lon_new.shape[0])

    return v_regridded
