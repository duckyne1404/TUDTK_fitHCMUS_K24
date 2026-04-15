"""
Singular Value Decomposition (SVD).

Decomposes an m x n matrix A into: A = U * Sigma * V^T

Where:
    U: m x m orthogonal matrix (left singular vectors)
    Sigma: m x n diagonal-rectangular matrix (singular values)
    V: n x n orthogonal matrix (right singular vectors)

Based on eigendecomposition of A^T * A.
"""

from utils import (
    matrix_multiply, transpose, normalize, shape, identity_matrix,
    matrix_subtract, scalar_multiply_matrix
)
from diagonalization import diagonalize
from QR import project

EPSILON = 1e-10


def _mat_vec_mul(A, v):
    """
    Multiply matrix A by vector v.
    
    Args:
        A: (m, n) matrix
        v: (n,) vector
    
    Returns: A * v
    """
    return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def _complete_orthonormal_basis(existing_cols, dim):
    """
    Extend a set of orthonormal vectors to a complete orthonormal basis using Gram-Schmidt.
    
    Args:
        existing_cols: list of orthonormal vectors (should already be orthonormal)
        dim: dimension of target space
    
    Returns: list of orthonormal vectors (length dim)
    
    Algorithm:
        1. Keep existing orthonormal vectors
        2. For each standard basis vector e_i:
           - If we already have dim vectors, stop
           - Gram-Schmidt orthogonalize e_i against existing basis
           - If result is non-zero, add to basis
    """
    basis = []

    # Keep existing columns
    for v in existing_cols:
        w = v[:]
        for u in basis:
            # Subtract projection onto u
            w = [w[i] - project(w, u)[i] for i in range(dim)]
        if sum(x * x for x in w) > EPSILON:
            basis.append(normalize(w))

    # Fill in with standard basis vectors if needed
    for idx in range(dim):
        if len(basis) == dim:
            break
        e = [0.0 for _ in range(dim)]
        e[idx] = 1.0
        w = e[:]
        for u in basis:
            w = [w[i] - project(w, u)[i] for i in range(dim)]
        if sum(x * x for x in w) > EPSILON:
            basis.append(normalize(w))

    return basis


def _extend_basis_preserving(existing_cols, dim):
    """
    Extend an already orthonormal basis without modifying existing vectors.

    Args:
        existing_cols: orthonormal vectors to keep fixed
        dim: target basis size

    Returns: list of orthonormal vectors (length dim)
    """
    basis = [v[:] for v in existing_cols]

    for idx in range(dim):
        if len(basis) == dim:
            break
        e = [0.0 for _ in range(dim)]
        e[idx] = 1.0
        w = e[:]
        for u in basis:
            w = [w[i] - project(w, u)[i] for i in range(dim)]
        if sum(x * x for x in w) > EPSILON:
            basis.append(normalize(w))

    return basis


def svd_decomposition(
    A,
    eigenvalue_method='qr_shift',
    eigenvector_method='inverse_iteration'
):
    """
    Compute Singular Value Decomposition of A.
    
    Args:
        A: m x n matrix
        eigenvalue_method: method for computing eigenvalues
            - 'qr_shift': QR with Rayleigh shift
            - 'qr': Standard QR
            - 'power_method': Power iteration + deflation
        eigenvector_method: method for computing eigenvectors
            - 'inverse_iteration' (DEFAULT)
            - 'power_method': Power iteration
            - 'eigenvector_from_lambda': RREF fallback
    
    Returns: (U, Sigma, V)
             U: m x m orthogonal matrix (left singular vectors as COLUMNS)
             Sigma: m x n diagonal-rectangular matrix with singular values on diagonal
             V: n x n orthogonal matrix (right singular vectors as COLUMNS)
    
    Such that: A ≈ U * Sigma * V^T
    
    Algorithm:
        1. Compute W = A^T * A (n x n)
        2. Diagonalize W to get V and eigenvalues
        3. Eigenvalues of W are squares of singular values: sigma_i = sqrt(lambda_i)
        4. Singular values might be negative (numerical error); clamp to 0
        5. Compute U columns: u_i = (1/sigma_i) * A * v_i
        6. Complete U to orthonormal basis if needed (for m > n)
        7. Complete V to orthonormal basis if needed 

    Formularly:
        A^T * A = V * D * V^T where D = diag(sigma_1^2, ..., sigma_n^2)
        
        For each non-zero singular value sigma_i:
            u_i = A * v_i / sigma_i
        
        The columns of U and V are orthonormal, forming left and right
        singular vector matrices.
    """
    m = len(A)
    n = len(A[0])

    # Compute W = A^T * A
    W = matrix_multiply(transpose(A), A)

    # Diagonalize W with selected methods
    V, D = diagonalize(
        W,
        eigenvalue_method=eigenvalue_method,
        eigenvector_method=eigenvector_method
    )

    # Extract eigenvalues and compute singular values
    lambdas = [D[i][i] for i in range(n)]
    # Clamp negative values to 0
    lambdas = [max(0.0, lam) for lam in lambdas]
    singular_values = [lam ** 0.5 for lam in lambdas]

    # Construct Sigma matrix (m x n rectangular)
    Sigma = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(min(m, n)):
        Sigma[i][i] = singular_values[i]

    V_cols = transpose(V)

    # Use the leading right-singular directions directly from diagonalization
    # to preserve sigma_i <-> v_i pairing before basis completion.
    r = min(m, n)
    paired_v_cols = []
    for i in range(r):
        sigma = singular_values[i]
        if sigma <= EPSILON:
            break
        w = V_cols[i][:]
        for u in paired_v_cols:
            w = [w[k] - project(w, u)[k] for k in range(n)]
        if sum(x * x for x in w) > EPSILON:
            paired_v_cols.append(normalize(w))

    full_v_cols = _extend_basis_preserving(paired_v_cols, n)
    V = transpose(full_v_cols)

    # Compute U columns: u_i = (1/sigma_i) * A * v_i
    u_cols = []
    for i in range(len(paired_v_cols)):
        sigma = singular_values[i]
        v_i = paired_v_cols[i]
        Av = _mat_vec_mul(A, v_i)
        u_i = [x / sigma for x in Av]
        for u_prev in u_cols:
            u_i = [u_i[k] - project(u_i, u_prev)[k] for k in range(m)]
        if sum(x * x for x in u_i) > EPSILON:
            u_cols.append(normalize(u_i))

    # Complete U basis if needed without changing paired leading vectors.
    full_u_cols = _extend_basis_preserving(u_cols, m)
    U = transpose(full_u_cols)

    return U, Sigma, V
