# rsa_metric.py
import numpy as np
from scipy.spatial.distance import cosine

def rsa_similarity(matrix_a: np.ndarray, matrix_b: np.ndarray) -> float:
    """
    Computes RSA similarity between two embedding matrices.
    - matrix_a: (T x D)
    - matrix_b: (T x D)
    Returns value in [-1, 1].
    """
    assert matrix_a.shape == matrix_b.shape

    T = matrix_a.shape[0]
    sims_a = []
    sims_b = []

    for i in range(T):
        for j in range(i + 1, T):
            sims_a.append(1 - cosine(matrix_a[i], matrix_a[j]))
            sims_b.append(1 - cosine(matrix_b[i], matrix_b[j]))

    sims_a = np.array(sims_a)
    sims_b = np.array(sims_b)

    return np.corrcoef(sims_a, sims_b)[0, 1]
