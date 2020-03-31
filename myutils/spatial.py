import numpy as np

__all__ = [
    'get_earth_distances',
]


def get_earth_distances(lat_lng1: np.ndarray, lat_lng2: np.ndarray) -> np.ndarray:
    """Calculate pair-wise distances between geographical coordinates
    (i.e. the haversine formula). The arguments are assumed to be
    numpy arrays of shape `(n_points, 2)`, where the two columns
    represent latitudes and longitudes, respectively.

    if `lat_lng1.shape = (m, 2)` and `lat_lng2.shape = (n, 2)`, then
    the output is a numpy array of shape `(m, n)`.

    :param lat_lng1: first array of coordinates
    :param lat_lng2: second array of coordinates
    :return: earth distances
    """

    r = 6371  # earth radius in km

    radians1 = lat_lng1 * np.pi / 180
    radians2 = lat_lng2 * np.pi / 180

    lat1 = radians1[:, [0]]
    lat2 = radians2[:, [0]]

    cos_lat1 = np.cos(lat1)
    cos_lat2 = np.cos(lat2)

    differences = radians2[np.newaxis, :, :] - radians1[:, np.newaxis, :]

    d_lat = differences[:, :, 0]
    d_lng = differences[:, :, 1]

    h = np.sin(d_lat / 2) ** 2 + cos_lat1.dot(cos_lat2.T) * np.sin(d_lng / 2) ** 2

    return 2 * r * np.arcsin(h ** 0.5)
