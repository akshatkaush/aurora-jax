#!/usr/bin/env python3
import xarray as xr

# ——— CONFIG —————————————————————————————————————————————————————————————
MAIN_ZARR = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"
STATIC_ZARR = "gs://weatherbench2/datasets/static/hres_t0_static.zarr"

OUT_MAIN = "hresDataset/hres_t0_2021-2022mid.zarr"
OUT_STATIC = "hresDataset/hres_t0_static.zarr"

# exactly the vars your code maps:
SURF_VARS = [
    "2m_temperature",  # → "2t"
    "10m_u_component_of_wind",  # → "10u"
    "10m_v_component_of_wind",  # → "10v"
    "mean_sea_level_pressure",  # → "msl"
]
ATMOS_VARS = [
    "temperature",  # → "t"
    "u_component_of_wind",  # → "u"
    "v_component_of_wind",  # → "v"
    "specific_humidity",  # → "q"
    "geopotential",  # → "z"
]
WANTED = SURF_VARS + ATMOS_VARS

TIME_SLICE = slice("2020-01-01", "2022-12-31")

# ——— download & trim main t=0 store —————————————————————————————————————
print(f"→ Opening main t=0 store: {MAIN_ZARR}")
ds = xr.open_dataset(
    MAIN_ZARR,
    engine="zarr",
    backend_kwargs={
        "consolidated": True,
        "storage_options": {"token": "anon"},
    },
    chunks={"time": 1},
)

print(f"→ Subsetting time {TIME_SLICE.start} → {TIME_SLICE.stop} and vars {WANTED}")
ds_trim = ds.sel(time=TIME_SLICE)[WANTED]

print(f"→ Writing trimmed main store to ./{OUT_MAIN}")
ds_trim.to_zarr(
    OUT_MAIN,
    mode="w",
    consolidated=True,
    zarr_format=2,
)
print("done with hres")

# ——— download static store (z, slt, lsm) ————————————————————————————————
print(f"→ Opening static store: {STATIC_ZARR}")
ds_static = xr.open_dataset(
    STATIC_ZARR,
    engine="zarr",
    backend_kwargs={
        "consolidated": True,
        "storage_options": {"token": "anon"},
    },
)

print("→ Extracting static vars ['z','slt','lsm']")
ds_static_sub = ds_static[["z", "slt", "lsm"]]

print(f"→ Writing static store to ./{OUT_STATIC}")
ds_static_sub.to_zarr(
    OUT_STATIC,
    mode="w",
    consolidated=True,
    zarr_format=2,
)

print("✅ Done!")
