from typing import Dict

import jax.numpy as jnp
import numpy as np
import xarray as xr

from aurora import Batch

export = ["weighted_rmse", "weighted_mae", "mae_loss_fn", "compute_weighted_acc"]


def weighted_rmse(pred, truth, lat):
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


def weighted_mae(pred, truth, lat):
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
    Compute the loss function with latitude weighting from two xr.DataArrays.
    Args:
        pred (Batch): prediction array
        truth (Batch): Truth.
        lat (np.ndarray): one dimension Latitude array
    Returns:
        loss: aurora loss function
    """
    V_S = len(surf_weights)
    V_A = len(atmos_weights)

    surf_terms = []
    for k, w_s in surf_weights.items():
        pred_s = pred.surf_vars[k][:, 1]  # [B, H, W]
        true_s = batch.surf_vars[k][:, 1]  # [B, H, W]
        diff = jnp.abs(pred_s - true_s)  # [B, H, W]
        surf_terms.append(w_s * jnp.mean(diff, axis=(1, 2)))  # [B]
    surf_loss = alpha * sum(surf_terms)

    atmos_terms = []
    for k, w_a in atmos_weights.items():
        pred_a = pred.atmos_vars[k][:, 1]  # [B, C, H, W]
        true_a = batch.atmos_vars[k][:, 1]  # [B, C, H, W]
        diff = jnp.abs(pred_a - true_a)  # [B]
        weighted = diff * w_a[None, :, None, None]
        C, H, W = diff.shape[1], diff.shape[2], diff.shape[3]
        normed = jnp.sum(weighted, axis=(1, 2, 3)) / (C * H * W)  # [B]
        atmos_terms.append(normed)
    atm_loss = beta * sum(atmos_terms)

    per_example = gamma / (V_S + V_A) * (surf_loss + atm_loss)  # [B]
    return jnp.mean(per_example)


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
    return acc
