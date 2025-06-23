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
    Collate function to stack individual Aurora samples into proper batches.
    For n GPUs, this creates batch size n from n individual samples.
    
    Args:
        batch_list: List of (input_batch, target_batches) tuples
        
    Returns:
        Tuple of (stacked_input_batch, list_of_stacked_target_batches)
    """
    num_samples = len(batch_list)
    if num_samples == 0:
        raise ValueError("Empty batch_list provided")
    
    # Stack input batches
    input_batches = [item[0] for item in batch_list]
    target_batches_list = [item[1] for item in batch_list]
    
    # Stack surf_vars
    stacked_surf_vars = {}
    for key in input_batches[0].surf_vars.keys():
        stacked_surf_vars[key] = jnp.concatenate([b.surf_vars[key] for b in input_batches], axis=0)
    
    # Stack atmos_vars  
    stacked_atmos_vars = {}
    for key in input_batches[0].atmos_vars.keys():
        stacked_atmos_vars[key] = jnp.concatenate([b.atmos_vars[key] for b in input_batches], axis=0)
    
    # Use static_vars and metadata from first sample (they should be identical)
    stacked_input = Batch(
        surf_vars=stacked_surf_vars,
        static_vars=input_batches[0].static_vars,
        atmos_vars=stacked_atmos_vars,
        metadata=input_batches[0].metadata,
    )
    
    # Stack target batches for each rollout step
    stacked_target_batches = []
    for step_idx in range(len(target_batches_list[0])):
        target_step_batches = [target_list[step_idx] for target_list in target_batches_list]
        
        # Stack surf_vars for this step
        stacked_step_surf = {}
        for key in target_step_batches[0].surf_vars.keys():
            stacked_step_surf[key] = jnp.concatenate([b.surf_vars[key] for b in target_step_batches], axis=0)
        
        # Stack atmos_vars for this step
        stacked_step_atmos = {}
        for key in target_step_batches[0].atmos_vars.keys():
            stacked_step_atmos[key] = jnp.concatenate([b.atmos_vars[key] for b in target_step_batches], axis=0)
        
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
