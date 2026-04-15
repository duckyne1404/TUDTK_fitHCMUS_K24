"""
QR Decomposition and Eigenvalue Finding Algorithms.

This module provides QR factorization and two methods for computing eigenvalues:
- eig_qr: Standard QR algorithm
- eig_qrshift: QR algorithm with Rayleigh quotient shift (faster convergence)
"""

from utils import (
    identity_matrix, normalize, matrix_add, matrix_subtract,
    matrix_multiply, scalar_multiply_matrix, transpose
)

EPSILON = 1e-10

def _dot(u, v):
    """
    Compute dot product of two vectors.
    
    Args:
        u: (n,) vector
        v: (n,) vector
    
    Returns: u . v = sum(u[i] * v[i])
    """
    return sum(u[i] * v[i] for i in range(len(u)))


def project(u, v):
    """
    Project vector u onto vector v.
    
    Args:
        u: (n,) vector
        v: (n,) vector
    
    Formula: proj_v(u) = (u . v) / (v . v) * v
    
    Returns: Scalar projection of u onto v
    """
    dot_uv = _dot(u, v)
    dot_vv = _dot(v, v)
    if abs(dot_vv) < EPSILON:
        raise ValueError("Cannot project onto the zero vector")
    scalar = dot_uv / dot_vv
    return [scalar * v[i] for i in range(len(v))]

def gram_schmidt(A):
    """
    Orthonormalize columns of a matrix using Gram-Schmidt process.
    
    Args:
        A: an n x n matrix (square)
           Each row of A is a vector in R^n
           We orthonormalize its columns
    
    Returns: Q (n x n matrix where columns are orthonormal vectors)
    """
    A_T = transpose(A)
    basis = []

    for j in range(len(A_T)):
        v = A_T[j]
        w = v[:]

        for qi in basis:
            coeff = _dot(v, qi)
            for k in range(len(w)):
                w[k] -= coeff * qi[k]

        basis.append(normalize(w))

    return transpose(basis)


def qr_decomposition(A):
    """
    Decompose matrix A into Q * R form.
    
    Args:
        A: an n x n matrix (square)

    Returns: Q (orthonormal columns), R (upper triangular)
             Such that A = Q * R
    
    Algorithm:
        1. Compute orthonormal basis Q using Gram-Schmidt
        2. Compute R = Q^T * A (upper triangular)
    """
    n = len(A)
    Q = gram_schmidt(A)
    A_T = transpose(A)
    Q_T = transpose(Q)
    R = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        qi = Q_T[i]
        for j in range(i, n):
            aj = A_T[j]
            R[i][j] = _dot(qi, aj)

    return Q, R


def makesimilar(A):
    """
    Perform one QR iteration: A_new = R * Q where A = Q * R.
    Args:
        A: n x n matrix
        
    Returns: B = R * Q
    """
    Q, R = qr_decomposition(A)
    B = matrix_multiply(R, Q)
    return B

def eig_qr(A, max_iters=2000):
    """
    Compute eigenvalues of A using standard QR algorithm (no shift).

    Args:
        A: n x n symmetric matrix
        max_iters: maximum number of iterations
    
    Returns: (eigenvalues, iterations_taken)
             eigenvalues: list of n eigenvalues
             iterations_taken: number of QR iterations until convergence

    Algorithm:
        A_0 = Q_0 * R_0
        A_1 = R_0 * Q_0
        A_k = Q_k * R_k
        A_{k+1} = R_k * Q_k ...
        Stops when A_k is sufficiently upper triangular (i.e., bottom-right element converges). The diagonal elements are eigenvalues.
    """
    B = makesimilar(A)
    iters = 0
    leig = B[-1][-1]
    diff = 1
    
    while diff > EPSILON and iters < max_iters:
        B = makesimilar(B)
        iters += 1
        diff = abs(B[-1][-1] - leig)
        leig = B[-1][-1]

    return [B[i][i] for i in range(len(B))], iters


def eig_qrshift(A, shindex=-1, max_iters=2000):
    """
    Compute eigenvalues of A using QR algorithm with shift.

    Args:
        A: n x n symmetric matrix
        shindex: which diagonal element to use as shift (default -1 = bottom-right)
        max_iters: maximum number of iterations
    
    Returns: (eigenvalues, iterations_taken)
             eigenvalues: list of n eigenvalues
             iterations_taken: number of QR iterations until convergence
    
    Algorithm:
        Choose shift s = A_k[shindex, shindex] at each iteration k
        C_k = B_k - s*I = Q_k * R_k
        B_{k+1} = R_k * Q_k + s*I
        
        Repeat until convergence.
        
        The shift accelerates convergence because during QR iterations,
        the bottom-right corner tends to converge to the smallest eigenvalue.
        Shifting by this value essentially removes it, allowing QR to focus
        on computing other eigenvalues.
    """
    B = makesimilar(A)
    iters = 0
    leigh = B[shindex][shindex]
    diff = 1
    
    while diff > EPSILON and iters < max_iters:
        I = identity_matrix(len(B))
        shift = scalar_multiply_matrix(I, leigh)
        C = matrix_subtract(B, shift)  # B_old - s*I = C = Q * R
        B = matrix_add(makesimilar(C), shift)  # B_new = R * Q + s*I
        iters += 1
        diff = abs(B[shindex][shindex] - leigh)
        leigh = B[shindex][shindex]

    return [B[i][i] for i in range(len(B))], iters
