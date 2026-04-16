import numpy as np

def verify_solution(A, x, b):
    return np.allclose(np.array(A) @ np.array(x), np.array(b), atol=1e-5)

def verify_inverse(A, A_inv):
    if A_inv is None:
        return False
    A = np.array(A)
    A_inv = np.array(A_inv)
    I = np.eye(len(A))
    return np.allclose(A @ A_inv, I, atol=1e-5)

def verify_determinant(A, det_custom):
    det_np = np.linalg.det(np.array(A))
    return np.allclose(det_custom, det_np)

def verify_rank(A, rank_custom):
    rank_np = np.linalg.matrix_rank(np.array(A))
    return rank_custom == rank_np