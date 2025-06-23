from typing import Dict

import jax.numpy as jnp
import numpy as np
import xarray as xr

from aurora import Batch

export = ["weighted_rmse", "weighted_mae", "mae_loss_fn", "compute_weighted_acc"]


def weighted_rmse_batch(pred: Batch, truth: Batch) -> jnp.ndarray:
    """
    Compute a single scalar RMSE over both surf_vars and atmos_vars in a Batch,
    with latitude weighting.

    Args:
        pred: Batch of predictions
        truth: Batch of ground-truths
    Returns:
        Scalar jnp.ndarray: latitude-weighted RMSE over all vars
    """
    lat = pred.metadata.lat  # → (h,)
    w_lat = jnp.cos(jnp.deg2rad(lat))
    w_lat = w_lat / jnp.mean(w_lat)  # ensure mean(w_lat)==1

    # 2) Prepare broadcastable weight views
    #   for surf_vars: (1,1,h,1) to match (b,t,h,w)
    w_surf = w_lat[None, None, :, None]
    #   for atmos_vars: (1,1,1,h,1) to match (b,t,c,h,w)
    w_atmos = w_lat[None, None, None, :, None]

    sum_wse = 0.0  # accumulate sum of weighted squared errors
    count = 0  # accumulate total element count

    # 3) Surface variables
    for k, p_arr in pred.surf_vars.items():
        t_arr = truth.surf_vars[k]  # → (b,t,h,w)
        err2 = (p_arr - t_arr) ** 2  # squared error
        wse = err2 * w_surf  # broadcasts to (b,t,h,w)
        sum_wse += jnp.sum(wse)
        count += wse.size

    # 4) Atmospheric variables
    for k, p_arr in pred.atmos_vars.items():
        t_arr = truth.atmos_vars[k]  # → (b,t,c,h,w)
        err2 = (p_arr - t_arr) ** 2
        wse = err2 * w_atmos  # → (b,t,c,h,w)
        sum_wse += jnp.sum(wse)
        count += wse.size

    # 5) Compute RMSE
    mse = sum_wse / count
    return jnp.sqrt(mse)


def weighted_rmse(pred: Batch, truth: Batch, lat):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.

    Args:
        pred (np.ndarray): prediction array
        truth (np.ndarray): Truth.
        lat (np.ndarray): one dimension Latitude array
    Returns:
        rmse: Latitude weighted root mean squared error
    """
    error = pred - truth
    weights = np.cos(np.deg2rad(lat))
    weights /= weights.mean()
    weights = weights.reshape(-1, 1)
    return np.sqrt(((error) ** 2 * weights).mean())


def weighted_mae(pred: Batch, truth: Batch, lat):
    """
    Compute the MAE with latitude weighting from two xr.DataArrays.
    Args:
        pred (np.ndarray): prediction array
        truth (np.ndarray): Truth.
        lat (np.ndarray): one dimension Latitude array
    Returns:
        mae: Latitude weighted root mean absolute error
    """
    error = pred - truth
    weights = np.cos(np.deg2rad(lat))
    weights /= weights.mean()
    weights = weights.reshape(-1, 1)
    mae = (np.abs(error) * weights).mean()
    return mae


def mae_loss_fn(
    pred: Batch,
    batch: Batch,
    surf_weights: Dict[str, float],
    atmos_weights: Dict[str, jnp.ndarray],
    gamma: float,
    alpha: float = 1 / 4,
    beta: float = 1.0,
):
    """
    Vectorized MAE loss for surf_vars shape [B,1,H,W] and
    atmos_vars shape [B,1,C,H,W].
    """

    # --- SURFACE part ---
    # stack over variables k -> [B, V_S, H, W]
    surf_preds = jnp.stack([pred.surf_vars[k][:, 0] for k in surf_weights], axis=1)
    surf_trues = jnp.stack([batch.surf_vars[k][:, 0] for k in surf_weights], axis=1)
    diff_s = jnp.abs(surf_preds - surf_trues)  # [B, V_S, H, W]

    # weight vector [V_S]
    w_s = jnp.array([surf_weights[k] for k in surf_weights])  # [V_S]

    # mean over H,W -> [B, V_S], dot with w_s -> [B]
    surf_term = jnp.dot(jnp.mean(diff_s, axis=(2, 3)), w_s)  # [B]
    surf_loss = alpha * surf_term  # [B]

    # --- ATMOSPHERIC part ---
    # stack over variables k -> [B, V_A, C, H, W]
    atm_preds = jnp.stack([pred.atmos_vars[k][:, 0] for k in atmos_weights], axis=1)
    atm_trues = jnp.stack([batch.atmos_vars[k][:, 0] for k in atmos_weights], axis=1)
    diff_a = jnp.abs(atm_preds - atm_trues)  # [B, V_A, C, H, W]

    # weight matrix [V_A, C]
    w_a = jnp.stack([atmos_weights[k] for k in atmos_weights], axis=0)  # [V_A, C]

    # apply per‐level weights and sum over C,H,W -> [B, V_A]
    weighted = diff_a * w_a[None, :, :, None, None]
    summed = jnp.sum(weighted, axis=(2, 3, 4))  # [B, V_A]

    # normalise by (C*H*W), then sum over V_A -> [B]
    C, H, W = diff_a.shape[2:]
    atm_term = jnp.sum(summed / (C * H * W), axis=1)  # [B]
    atm_loss = beta * atm_term  # [B]

    # --- combine & average ---
    V_S = len(surf_weights)
    V_A = len(atmos_weights)
    per_example = gamma / (V_S + V_A) * (surf_loss + atm_loss)  # [B]
    return jnp.mean(per_example)  # scalar


# todo need to complete this function
def compute_weighted_acc(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the ACC with latitude weighting from two xr.DataArrays.
    WARNING: Does not work if datasets contain NaNs

    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        acc: Latitude weighted acc
    """
    clim = da_true.mean("time")
    try:
        t = np.intersect1d(da_fc.time, da_true.time)
        fa = da_fc.sel(time=t) - clim
    except AttributeError:
        t = da_true.time.values
        fa = da_fc - clim
    a = da_true.sel(time=t) - clim

    weights_lat = np.cos(np.deg2rad(da_fc.lat))
    weights_lat /= weights_lat.mean()
    w = weights_lat

    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()

    acc = np.sum(w * fa_prime * a_prime) / np.sqrt(np.sum(w * fa_prime**2) * np.sum(w * a_prime**2))
    return acc7hauz
