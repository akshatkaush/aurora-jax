

import jax.numpy as jnp

__all__ = ["area", "compute_patch_areas", "radius_earth"]

radius_earth = 6378137 / 1000
"""float: Radius of the earth in kilometers."""


def area(polygon: jnp.ndarray) -> jnp.ndarray:
    """Compute the area of a polygon specified by latitudes and longitudes in degrees.

    Args:
        polygon (:class:`jax.numpy.ndarray`): Polygon of the shape
        `(*b, n, 2)` where `b` is an optional
            multidimensional batch size, `n` is the number of points of
            the polygon, and 2
            concatenates first latitudes and then longitudes.
            The polygon does not have to be closed.

    Returns:
        :class:`jax.numpy.ndarray`: Area in square kilometers.
    """
    # Be sure to close the loop.
    polygon = jnp.concatenate((polygon, polygon[..., -1:, :]), axis=-2)

    area = jnp.zeros(polygon.shape[:-2], dtype=polygon.dtype)
    n = polygon.shape[-2]  # Number of points of the polygon

    rad = jnp.deg2rad  # Convert degrees to radians.

    if n > 2:
        for i in range(n):
            i_lower = i
            i_middle = (i + 1) % n
            i_upper = (i + 2) % n

            lon_lower = polygon[..., i_lower, 1]
            lat_middle = polygon[..., i_middle, 0]
            lon_upper = polygon[..., i_upper, 1]

            area = area + (rad(lon_upper) - rad(lon_lower)) * jnp.sin(rad(lat_middle))

    area *= radius_earth * radius_earth / 2

    return jnp.abs(area)


def expand_matrix(matrix: jnp.ndarray) -> jnp.ndarray:
    """Expand matrix by adding one row and one column to each side, using
    linear interpolation.

    Args:
        matrix (:class:`jax.numpy.ndarray`): Matrix to expand.

    Returns:
        :class:`jax.numpy.ndarray`: `matrix`, but with two extra rows and two extra columns.
    """
    # Add top and bottom rows.
    matrix = jnp.concatenate(
        (
            2 * matrix[0:1] - matrix[1:2],
            matrix,
            2 * matrix[-1:] - matrix[-2:-1],
        ),
        axis=0,
    )

    # Add left and right columns.
    matrix = jnp.concatenate(
        (
            2 * matrix[:, 0:1] - matrix[:, 1:2],
            matrix,
            2 * matrix[:, -1:] - matrix[:, -2:-1],
        ),
        axis=1,
    )

    return matrix


def compute_patch_areas(lat: jnp.ndarray, lon: jnp.ndarray) -> jnp.ndarray:
    """A pair of latitude and longitude matrices defines a
    number non-intersecting patches on the Earth.

    Args:
        lat (:class:`jax.numpy.ndarray`): Latitude matrix. Must be decreasing along rows.
        lon (:class:`jax.numpy.ndarray`): Longitude matrix. Must be increasing along columns.

    Returns:
        :class:`jax.numpy.ndarray`: Areas in square kilometer.
    """
    if not (lat.ndim == lon.ndim == 2):
        raise ValueError("`lat` and `lon` must both be matrices.")
    if lat.shape != lon.shape:
        raise ValueError("`lat` and `lon` must have the same shape.")

    # Check that the latitude matrix is decreasing in the appropriate way.
    if not jnp.all(lat[1:] - lat[:-1] <= 0):
        raise ValueError("`lat` must be decreasing along rows.")

    # Check that the longitude matrix is increasing in the appropriate way.
    if not jnp.all(lon[:, 1:] - lon[:, :-1] >= 0):
        raise ValueError("`lon` must be increasing along columns.")

    # Enlarge the latitude and longitude matrices for the midpoint computation.
    lat = expand_matrix(lat)
    lon = expand_matrix(lon)

    # Latitudes cannot expand beyond the poles.
    lat = jnp.clip(lat, -90, 90)

    # Calculate midpoints between entries in lat/lon for symmetry of resulting areas.
    lat_midpoints = (lat[:-1, :-1] + lat[:-1, 1:] + lat[1:, :-1] + lat[1:, 1:]) / 4
    lon_midpoints = (lon[:-1, :-1] + lon[:-1, 1:] + lon[1:, :-1] + lon[1:, 1:]) / 4

    # Determine squares and return their areas.
    top_left = jnp.stack((lat_midpoints[1:, :-1], lon_midpoints[1:, :-1]), axis=-1)
    top_right = jnp.stack((lat_midpoints[1:, 1:], lon_midpoints[1:, 1:]), axis=-1)
    bottom_left = jnp.stack((lat_midpoints[:-1, :-1], lon_midpoints[:-1, :-1]), axis=-1)
    bottom_right = jnp.stack((lat_midpoints[:-1, 1:], lon_midpoints[:-1, 1:]), axis=-1)

    polygon = jnp.stack((top_left, top_right, bottom_right, bottom_left), axis=-2)

    return area(polygon)
