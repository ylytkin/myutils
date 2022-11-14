import numpy as np
from scipy import sparse

__all__ = [
    "normalize_vectors",
    "pairwise_jaccard_coefficient",
]


def normalize_vectors(matrix: sparse.csr_matrix, axis: int) -> sparse.csr_matrix:
    """Performs normalization of vectors (i.e. divide each vector
    by its corresponding Euclidean norm).

    Parameter `axis` can be 0 (column-vectors) or 1 (row-vectors)

    :param matrix: sparse matrix
    :param axis: 0 or 1
    :return: sparse matrix
    """

    if axis not in {0, 1}:
        raise ValueError("Axis must be either 0 or 1.")

    matrix = matrix.copy().astype(np.float64)

    matrix_norms = matrix.copy()
    matrix_norms.data **= 2
    matrix_norms = matrix_norms.sum(axis=axis).A.flatten() ** 0.5
    matrix_norms = matrix_norms[matrix.nonzero()[1 - axis]]

    matrix.data /= matrix_norms

    return matrix


def pairwise_jaccard_coefficient(
    matrix_1: sparse.csr_matrix,
    matrix_2: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """Calculate pairwise Jaccard coefficients between rows of two matrices.

    Args:
        matrix_1 (sparse.csr_matrix): first matrix
        matrix_2 (sparse.csr_matrix): second matrix

    Returns:
        sparse.csr_matrix: matrix of Jaccard coefficients
    """

    intersections_matrix = matrix_1.dot(matrix_2.T)

    row, col = intersections_matrix.nonzero()
    intersections = intersections_matrix.data

    matrix_1_sizes = matrix_1.sum(axis=1).A.flatten()
    matrix_2_sizes = matrix_2.sum(axis=1).A.flatten()

    unions = matrix_1_sizes[row] + matrix_2_sizes[col] - intersections

    jaccard_coefficient = intersections / unions

    jaccard_coefficient_matrix = sparse.csr_matrix(
        (jaccard_coefficient, (row, col)),
        shape=intersections_matrix.shape,
    )

    return jaccard_coefficient_matrix
