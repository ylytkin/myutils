import numpy as np
from scipy import sparse

__all__ = [
    'normalize_vectors',
]


def normalize_vectors(mx: sparse.csr_matrix, axis: int) -> sparse.csr_matrix:
    """Performs normalization of vectors (i.e. divide each vector
    by its corresponding Euclidean norm).

    Parameter `axis` can be 0 (column-vectors) or 1 (row-vectors)

    :param mx: sparse matrix
    :param axis: 0 or 1
    :return: sparse matrix
    """

    if axis not in {0, 1}:
        raise ValueError('Axis must be either 0 or 1.')

    mx = mx.copy().astype(np.float64)

    mx_norms = mx.copy()
    mx_norms.data **= 2
    mx_norms = mx_norms.sum(axis=axis).A.flatten() ** 0.5
    mx_norms = mx_norms[mx.nonzero()[1 - axis]]

    mx.data /= mx_norms

    return mx
