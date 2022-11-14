import numpy as np
from scipy import sparse

from myutils.sparse import normalize_vectors


def test_sparse():
    mx = np.random.randint(0, 2, size=(15, 20))
    sparse_mx = sparse.csr_matrix(mx)

    for axis in {0, 1}:
        mx_with_normalized_vectors = mx / np.linalg.norm(mx, axis=axis, keepdims=True)
        mx_with_normalized_vectors[np.isnan(mx_with_normalized_vectors)] = 0

        sparse_mx_with_normalized_vectors = normalize_vectors(sparse_mx, axis=axis)

        assert (sparse_mx_with_normalized_vectors == mx_with_normalized_vectors).all()
