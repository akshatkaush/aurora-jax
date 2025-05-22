# save_batches.py

import os
from datetime import datetime

import numpy as np


def npify(x):
    # torch.Tensor → np.ndarray, jax.ndarray → np.ndarray
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    else:
        return np.array(x)


def save_batch_npz(batch, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)

    def path(name):
        return os.path.join(out_dir, f"{prefix}_{name}.npz")

    for k, arr in batch.surf_vars.items():
        np.savez_compressed(path(f"surf_{k}"), data=npify(arr))

    for k, arr in batch.static_vars.items():
        np.savez_compressed(path(f"static_{k}"), data=npify(arr))

    for k, arr in batch.atmos_vars.items():
        np.savez_compressed(path(f"atm_{k}"), data=npify(arr))

    timestamps = []
    for t in batch.metadata.time:
        if isinstance(t, datetime):
            timestamps.append(int(t.timestamp()))
        else:
            timestamps.append(int(np.array(t).item()))

    np.savez_compressed(
        path("meta"),
        lat=npify(batch.metadata.lat),
        lon=npify(batch.metadata.lon),
        levels=np.array(batch.metadata.atmos_levels, dtype=np.int32),
        time=np.array(timestamps, dtype=np.int64),
    )
