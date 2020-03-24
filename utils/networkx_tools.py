import networkx as nx
import numpy as np
from scipy import sparse

__all__ = [
    'DistanceConversionError',
    'calculate_distances',
]


class NetworkxUtilsBaseException(Exception):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class DistanceCalculationError(NetworkxUtilsBaseException):
    def __init__(self, n_edges: int) -> None:
        msg = f"Did not reach the graph diameter after {n_edges} (number of edges) iterations."

        super().__init__(msg)


class DistanceConversionError(NetworkxUtilsBaseException):
    def __init__(self, n_errors: int) -> None:
        msg = (f"Found {n_errors} non-integer values while converting matrix values. "
               f"This is not your fault, but currently unresolved. "
               f"See the docstring to `calculate_distances` or set `check_result=False`.")

        super().__init__(msg)


def calculate_distances(g: nx.Graph, check_result: bool = True) -> np.ndarray:
    """Calculate all distances in an unweighted undirected `networkx` graph (uses `numpy`
    `scipy` for speed, sparse matrices for memory efficiency).

    Note. In order to calculate non-zero minimums of sparse matrices this function performs
    multiple integer inversions (i.e. `1 / (1 / int)`). This sometimes results in non-integer
    values when dealing with large numbers. The function will raise an exception in these
    cases if `check_result` is True (default).

    :param g: networkx Graph
    :param check_result: bool
    :return: numpy ndarray with integers
    """

    adj_mx = nx.adjacency_matrix(g)
    
    distances = [adj_mx]
    temp_mx = adj_mx
    nonzero_before = adj_mx.count_nonzero()
    nonzero_after = None

    for i in range(2, g.number_of_edges()):
        temp_mx = temp_mx.dot(adj_mx)
        nonzero_after = temp_mx.count_nonzero()

        if nonzero_after == nonzero_before:
            break

        distances.append(temp_mx.astype(bool).astype(int) / i)
        nonzero_before = nonzero_after
        
    if nonzero_before != nonzero_after:
        raise DistanceCalculationError(g.number_of_edges())
        
    distances = [
        sparse.vstack([mx[i] for mx in distances])
        for i in range(adj_mx.shape[0])
    ]
    
    distances = (1 / np.array([
        mx.max(axis=0).todense().A[0]
        for mx in distances
    ]))

    distances_int = distances.astype(int)

    if check_result is True:
        n_errors = np.where(distances != distances_int)[0].shape[0]

        if n_errors > 0:
            raise DistanceConversionError(n_errors)

    for i in range(distances_int.shape[0]):
        distances_int[i, i] = 0
    
    return distances_int
