from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "get_earth_distances",
]


def get_earth_distances(
    lat_lng1: NDArray[Any],
    lat_lng2: NDArray[Any],
    earth_radius: float = 6371,
) -> NDArray[Any]:
    """Calculate pair-wise distances between geographical coordinates
    (i.e. the haversine formula). The arguments are assumed to be
    numpy arrays of shape `(n_points, 2)`, where the two columns
    represent latitudes and longitudes, respectively.

    if `lat_lng1.shape = (m, 2)` and `lat_lng2.shape = (n, 2)`, then
    the output is a numpy array of shape `(m, n)`.

    Args:
        lat_lng1 (NDArray[Any]): first array of coordinates
        lat_lng2 (NDArray[Any]): second array of coordinates
        earth_radius (float, optional): radius of Earth. Defaults to 6371.

    Returns:
        NDArray[Any]: earth distances
    """

    radians1 = lat_lng1 * np.pi / 180
    radians2 = lat_lng2 * np.pi / 180

    lat1 = radians1[:, [0]]
    lat2 = radians2[:, [0]]

    cos_lat1 = np.cos(lat1)
    cos_lat2 = np.cos(lat2)

    differences = radians2[np.newaxis, :, :] - radians1[:, np.newaxis, :]

    d_lat = differences[:, :, 0]
    d_lng = differences[:, :, 1]

    haversine = np.sin(d_lat / 2) ** 2 + cos_lat1.dot(cos_lat2.T) * np.sin(d_lng / 2) ** 2

    distances: NDArray[Any] = 2 * earth_radius * np.arcsin(haversine**0.5)

    return distances
