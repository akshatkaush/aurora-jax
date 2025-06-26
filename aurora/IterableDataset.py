import random

import jax.numpy as jnp
import xarray as xr
from torch.utils.data import IterableDataset, get_worker_info

from aurora.batch import Batch, Metadata

# HRES t0 keys translation map
surf_map = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}
atmos_map = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}


def collate_aurora_batches(batch_list):
    """
    Optimized collate function for multi-GPU training.
    Efficiently stacks individual Aurora samples into proper batches.
    
    Args:
        batch_list: List of (input_batch, target_batches) tuples
        
    Returns:
        Tuple of (stacked_input_batch, list_of_stacked_target_batches)
    """
    num_samples = len(batch_list)
    if num_samples == 0:
        raise ValueError("Empty batch_list provided")
    
    # Extract input batches and target batches lists
    input_batches = [item[0] for item in batch_list]
    target_batches_list = [item[1] for item in batch_list]
    
    # Get reference structures from first sample
    first_input = input_batches[0]
    first_targets = target_batches_list[0]
    
    # Stack surf_vars efficiently using jnp.stack instead of concatenate
    stacked_surf_vars = {}
    for key in first_input.surf_vars.keys():
        # Stack along new first dimension (batch dimension)
        surf_arrays = [b.surf_vars[key] for b in input_batches]
        stacked_surf_vars[key] = jnp.stack(surf_arrays, axis=0)
    
    # Stack atmos_vars efficiently  
    stacked_atmos_vars = {}
    for key in first_input.atmos_vars.keys():
        # Stack along new first dimension (batch dimension)
        atmos_arrays = [b.atmos_vars[key] for b in input_batches]
        stacked_atmos_vars[key] = jnp.stack(atmos_arrays, axis=0)
    
    # Create stacked input batch
    stacked_input = Batch(
        surf_vars=stacked_surf_vars,
        static_vars=first_input.static_vars,  # Static vars are the same across samples
        atmos_vars=stacked_atmos_vars,
        metadata=first_input.metadata,  # Metadata is the same across samples
    )
    
    # Stack target batches for each rollout step
    num_rollout_steps = len(first_targets)
    stacked_target_batches = []
    
    for step_idx in range(num_rollout_steps):
        # Get all target batches for this rollout step
        target_step_batches = [target_list[step_idx] for target_list in target_batches_list]
        
        # Stack surf_vars for this step
        stacked_step_surf = {}
        for key in target_step_batches[0].surf_vars.keys():
            surf_arrays = [b.surf_vars[key] for b in target_step_batches]
            stacked_step_surf[key] = jnp.stack(surf_arrays, axis=0)
        
        # Stack atmos_vars for this step
        stacked_step_atmos = {}
        for key in target_step_batches[0].atmos_vars.keys():
            atmos_arrays = [b.atmos_vars[key] for b in target_step_batches]
            stacked_step_atmos[key] = jnp.stack(atmos_arrays, axis=0)
        
        # Create stacked target batch for this step
        stacked_step_batch = Batch(
            surf_vars=stacked_step_surf,
            static_vars=target_step_batches[0].static_vars,
            atmos_vars=stacked_step_atmos,
            metadata=target_step_batches[0].metadata,
        )
        stacked_target_batches.append(stacked_step_batch)
    
    return stacked_input, stacked_target_batches


class HresT0SequenceDataset(IterableDataset):
    """
    Yields (input_batch, target_batch):
      - input_batch: 2 timesteps at [i-2, i-1]
      - target_batch: 1 timestep at [i]
    """

    def __init__(
        self,
        zarr_path: str,
        mode: str = "train",
        shuffle: bool = True,
        seed: int | None = None,
        steps: int = 1,
    ):
        ds_full = xr.open_zarr(zarr_path, consolidated=True, chunks={"time": 1})
        if mode == "train":
            ds = ds_full.sel(time=slice("2020-01-01", "2021-12-31"))
        else:
            ds = ds_full.sel(time=slice("2022-01-01", "2022-12-31"))
        self.ds = ds[list(surf_map.values()) + list(atmos_map.values())]

        static_ds = xr.open_dataset("/scratch/akaush/dataset/static.nc")
        self.static_vars = {
            "z": jnp.array(static_ds["z"].values[0]),
            "slt": jnp.array(static_ds["slt"].values[0]),
            "lsm": jnp.array(static_ds["lsm"].values[0]),
        }

        self.lat = jnp.array(self.ds.latitude.values, dtype=jnp.float32)
        self.lon = jnp.array(self.ds.longitude.values, dtype=jnp.float32)
        self.levels = tuple(int(Plevels) for Plevels in self.ds.level.values)
        self.times = self.ds.time.values

        self.shuffle = shuffle
        self.seed = seed
        self.rollout_steps = steps

        # pre-compute all valid indices once
        max_start = len(self.times) - (self.rollout_steps - 1)
        self._idxs = list(range(2, max_start))

    def __iter__(self):
        worker = get_worker_info()
        if self.shuffle:
            if self.seed is None:
                rnd = random.Random()
            else:
                rnd = random.Random(self.seed + (worker.id if worker else 0))
            idxs = self._idxs.copy()
            rnd.shuffle(idxs)
        else:
            idxs = self._idxs

        for i in idxs:
            surf_in = {
                key: jnp.array(self.ds[var].isel(time=[i - 2, i - 1]).fillna(0).values[None])
                for key, var in surf_map.items()
            }
            atmos_in = {
                key: jnp.array(self.ds[var].isel(time=[i - 2, i - 1]).fillna(0).values[None])
                for key, var in atmos_map.items()
            }
            ts_in = int(self.times[i - 1].astype("datetime64[s]").tolist().timestamp())
            meta_in = Metadata(
                lat=self.lat,
                lon=self.lon,
                time=(jnp.array(ts_in, dtype=jnp.int64),),
                atmos_levels=self.levels,
            )
            in_batch = Batch(
                surf_vars=surf_in,
                static_vars=self.static_vars,
                atmos_vars=atmos_in,
                metadata=meta_in,
            )

            out_batch_list = []
            temp_i = i
            for _ in range(self.rollout_steps):
                surf_out = {
                    key: jnp.array(self.ds[var].isel(time=[temp_i]).fillna(0).values[None])
                    for key, var in surf_map.items()
                }
                atmos_out = {
                    key: jnp.array(self.ds[var].isel(time=[temp_i]).fillna(0).values[None])
                    for key, var in atmos_map.items()
                }
                ts_out = int(self.times[temp_i].astype("datetime64[s]").tolist().timestamp())
                meta_out = Metadata(
                    lat=self.lat,
                    lon=self.lon,
                    time=(jnp.array(ts_out, dtype=jnp.int64),),
                    atmos_levels=self.levels,
                )
                out_batch = Batch(
                    surf_vars=surf_out,
                    static_vars=self.static_vars,
                    atmos_vars=atmos_out,
                    metadata=meta_out,
                )

                out_batch_list.append(out_batch)
                temp_i = temp_i + 1

            yield in_batch, out_batch_list
