import numpy as np

from myutils.spatial import get_earth_distances


def test_spatial():
    lat_lng1 = np.array([[0, 0]])
    lat_lng2 = np.array([[0, 180], [90, 0]])
    distances_real = np.array([20015.086796020572, 10007.543398010284])

    distances = get_earth_distances(lat_lng1, lat_lng2)

    assert distances.shape == (lat_lng1.shape[0], lat_lng2.shape[0])
    assert (distances.flatten() == distances_real).all()
