"""
Diagonalization: Finding Eigenvalues and Eigenvectors -> A = P * D * P^T 

EIGENVALUE METHODS:
    - 'qr_shift' (default): QR algorithm with Rayleigh quotient shift
    - 'qr': Standard QR algorithm
    - 'power_method': Power iteration + deflation

EIGENVECTOR METHODS (for each found eigenvalue):
    - 'inverse_iteration' (default): Inverse iteration
    - 'power_method': Power iteration
    - 'eigenvector_from_lambda': RREF + back-substitution

MATRIX CONVENTION:
    All matrices are stored in row-major format.
    Each row is a vector in R^n.
"""

import random
import math
from utils import (
    identity_matrix, normalize, matrix_add, matrix_subtract,
    matrix_multiply, scalar_multiply_matrix, transpose, shape
)
from QR import eig_qr, eig_qrshift, project

EPSILON = 1e-10

# ---------------- POWER ITERATION & DEFLATION ----------------

def power_iteration(A, iterations=1000, epsilon=1e-9):
    """
    Power iteration to find the dominant eigenvalue and eigenvector.    
    Args:
        A: n x n symmetric matrix
        iterations: max iterations (default 1000)
        epsilon: convergence tolerance (default 1e-9)
    
    Returns: (eigenvalue, eigenvector)
             eigenvalue: largest (by magnitude) eigenvalue
             eigenvector: corresponding normalized eigenvector

    Algorithm:
        v_0 = random vector
        repeat:
            v_k+1 = A * v_k / ||A * v_k||
        until v_k+1 ≈ v_k (convergence or ping-pong detected)

    Note: Converges to dominant (largest magnitude) eigenvalue.
          Includes ping-pong detection to handle repeated eigenvalues.
    """
    if not A or len(A) != len(A[0]):
        raise ValueError("Matrix must be square and non-empty")
    
    n = len(A)
    v = [random.uniform(-1, 1) for _ in range(n)]
    v = normalize(v)
    v_history = [v]

    for i in range(iterations):
        # A * v
        Av = [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]
        eigenvalue = sum(v[i] * Av[i] for i in range(len(v)))
        
        norm_Av = math.sqrt(sum(x * x for x in Av))
        if norm_Av < 1e-15:
            return 0.0, v
            
        v_next = [x / norm_Av for x in Av]

        # Check standard convergence
        convergence = sum(v[i] * v_next[i] for i in range(len(v)))
        if 1 - abs(convergence) < epsilon:
            return eigenvalue, v_next

        # Check ping-pong (oscillation between two vectors)
        if len(v_history) >= 2:
            ping_pong_check = sum(v_next[i] * v_history[-2][i] for i in range(len(v_next)))
            if 1 - abs(ping_pong_check) < epsilon:
                return norm_Av, v_next

        v = v_next
        v_history.append(v)
        if len(v_history) > 2:
            v_history.pop(0)

    return eigenvalue, v


def deflation(A, lam, v):
    """
    Deflate a matrix by removing the contribution of one eigenpair.

    Args:
        A: n x n matrix
        lam: eigenvalue to remove
        v: corresponding eigenvector (must be normalized)
    
    Returns: A_deflated = A - lambda * v * v^T
    
    After deflation, the next power iteration on A_deflated
    will find the next eigenvalue.
    """
    n = len(A)
    vvT = [[v[i] * v[j] for j in range(n)] for i in range(n)]
    scaled = [[lam * vvT[i][j] for j in range(n)] for i in range(n)]
    return matrix_subtract(A, scaled)


# ---------------- INVERSE ITERATION & RREF SOLVER ----------------

def _solve_linear_system(A, b):
    """
    Solve A * x = b using Gaussian elimination with partial pivoting.
    
    Args:
        A: n x n matrix
        b: n x 1 vector
    
    Returns: x (solution vector)    
    """
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        pivot = col
        while pivot < n and abs(M[pivot][col]) < EPSILON:
            pivot += 1

        if pivot == n:
            raise ValueError("Linear system is singular or ill-conditioned")

        if pivot != col:
            M[col], M[pivot] = M[pivot], M[col]

        pivot_value = M[col][col]
        for j in range(col, n + 1):
            M[col][j] /= pivot_value

        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            for j in range(col, n + 1):
                M[r][j] -= factor * M[col][j]

    return [M[i][n] for i in range(n)]


def _inverse_iteration(A, lam, seed_index=0, steps=8):
    """
    Inverse iteration: refine an eigenvalue estimate to find eigenvector.
        
    Args:
        A: n x n matrix
        lam: approximate eigenvalue estimate
        seed_index: which coordinate to start with
        steps: iteration refinement steps
    
    Returns: Normalized eigenvector corresponding to lam
    
    Algorithm:
        For each refinement step:
            1. Shift A: (A - lam*I)
            2. Solve (A - lam*I) * x = x_old for new x
            3. Normalize x
        
        The shift value is adjusted with small deltas to avoid
        singular matrices and improve robustness.
    """
    n = len(A)
    x = [0.0 for _ in range(n)]
    x[seed_index % n] = 1.0

    for _ in range(steps):
        best_candidate = None
        for delta in (1e-8, -1e-8, 1e-6, -1e-6, 1e-4, -1e-4):
            shifted = matrix_subtract(
                A,
                scalar_multiply_matrix(identity_matrix(n), lam + delta)
            )
            try:
                y = _solve_linear_system([row[:] for row in shifted], x)
                best_candidate = y
                break
            except ValueError:
                continue

        if best_candidate is None:
            raise ValueError("Inverse iteration failed to solve shifted system")

        x_new = normalize(best_candidate)
        if sum(abs(x_new[i] - x[i]) for i in range(n)) < 1e-8:
            x = x_new
            break
        x = x_new

    return x


def solve_rref(A):
    """
    Reduce a matrix to row-echelon form (RREF) and track pivot columns.
    
    Args:
        A: m x n matrix
    
    Returns: (echelon_form, pivot_column_indices)
    """
    m, n = shape(A)
    row = 0
    pivot_cols = []
    
    for col in range(n):
        if row >= m:
            break
        
        # Find pivot row
        pivot_row = row
        while pivot_row < m and abs(A[pivot_row][col]) < EPSILON:
            pivot_row += 1
        
        if pivot_row == m:
            continue  # no pivot

        A[row], A[pivot_row] = A[pivot_row], A[row]
        pivot_cols.append(col)

        pivot_value = A[row][col]
        for j in range(col, n):
            A[row][j] /= pivot_value
        
        for r in range(row + 1, m):
            factor = A[r][col]
            for j in range(col, n):
                A[r][j] -= factor * A[row][j]
        
        row += 1
    
    return A, pivot_cols


def _back_substitute_null_vector(U, pivot_cols):
    """
    Find a solution to U * x = 0 given an upper triangular matrix U and pivot columns.
    
    Args:
        U: m x n upper triangular matrix
        pivot_cols: list of pivot column indices
    
    Returns: (n,) vector x such that U * x = 0
    """
    n = len(U[0])
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n) if c not in pivot_set]

    if not free_cols:
        raise ValueError("No free variable found when solving (A - lambda*I)x = 0")

    x = [0.0 for _ in range(n)]
    x[free_cols[0]] = 1.0

    for r in range(len(pivot_cols) - 1, -1, -1):
        c = pivot_cols[r]
        s = 0.0
        for j in range(c + 1, n):
            s += U[r][j] * x[j]
        x[c] = -s

    return normalize(x)


def eigenvector_from_lambda(A, lam):
    """
    Find eigenvector for eigenvalue lam using inverse iteration + RREF fallback.
    
    Args:
        A: n x n matrix
        lam: eigenvalue estimate
    
    Returns: Normalized eigenvector
    
    Algorithm:
        1. Try inverse iteration first
        2. Fallback to RREF + back-substitution if inverse iteration fails
    """
    n = len(A)
    try:
        return _inverse_iteration(A, lam, seed_index=0)
    except ValueError:
        I = identity_matrix(n)
        shifted = matrix_subtract(A, scalar_multiply_matrix(I, lam))
        echelon, pivot_cols = solve_rref([row[:] for row in shifted])
        return _back_substitute_null_vector(echelon, pivot_cols)

def diagonalize(A, eigenvalue_method='qr_shift', eigenvector_method='inverse_iteration'):
    """
    Diagonalize a symmetric matrix into P * D * P^T = A, where D is diagonal.
    
    SUPPORTS MULTIPLE METHODS:
    
    Eigenvalue Methods:
        - 'qr_shift': QR with Rayleigh shift
        - 'qr': Standard QR algorithm
        - 'power_method': Power iteration + deflation
    
    Eigenvector Methods:
        - 'inverse_iteration'
        - 'power_method': From power iteration itself
        - 'eigenvector_from_lambda': Robust fallback via RREF

    Args:
        A: n x n symmetric matrix
        eigenvalue_method: which method to use for eigenvalues
        eigenvector_method: which method to use for eigenvectors
    
    Returns: (P, D)
             P: n x n matrix whose columns are eigenvectors (sorted by eigenvalue magnitude)
             D: n x n diagonal matrix with eigenvalues on diagonal
    
    Examples:
        # Default (fastest): QR shift + inverse iteration
        P, D = diagonalize(A)
        
        # Power method for comparison/benchmarking
        P, D = diagonalize(A, eigenvalue_method='power_method',
                              eigenvector_method='power_method')
        
        # Mixed: QR eigenvalues, power eigenvectors
        P, D = diagonalize(A, eigenvalue_method='qr_shift',
                              eigenvector_method='power_method')
    """
    n = len(A)

    # Keep eigenvalue/eigenvector pairing consistent for every method combination.
    if eigenvalue_method == 'power_method' and eigenvector_method == 'power_method':
        eigenpairs = []
        A_work = [row[:] for row in A]
        for _ in range(n):
            lam, v = power_iteration(A_work)
            eigenpairs.append((lam, normalize(v)))
            A_work = deflation(A_work, lam, v)
        eigenpairs.sort(key=lambda pair: pair[0], reverse=True)
        eigenvalues = [pair[0] for pair in eigenpairs]
        vectors = [pair[1] for pair in eigenpairs]
    else:
        if eigenvalue_method == 'qr_shift':
            eigenvalues, _ = eig_qrshift(A)
        elif eigenvalue_method == 'qr':
            eigenvalues, _ = eig_qr(A)
        elif eigenvalue_method == 'power_method':
            eigenvalues = []
            A_work = [row[:] for row in A]
            for _ in range(n):
                lam, v = power_iteration(A_work)
                eigenvalues.append(lam)
                A_work = deflation(A_work, lam, v)
        else:
            raise ValueError(
                f"Unknown eigenvalue_method: {eigenvalue_method}. "
                "Choose from: 'qr_shift', 'qr', 'power_method'"
            )

        eigenvalues = sorted(eigenvalues, reverse=True)
        vectors = []

        if eigenvector_method == 'inverse_iteration':
            for i, lam in enumerate(eigenvalues):
                try:
                    vectors.append(_inverse_iteration(A, lam, seed_index=i))
                except ValueError:
                    vectors.append(eigenvector_from_lambda(A, lam))
        elif eigenvector_method == 'eigenvector_from_lambda':
            for lam in eigenvalues:
                vectors.append(eigenvector_from_lambda(A, lam))
        elif eigenvector_method == 'power_method':
            raise ValueError(
                "eigenvector_method='power_method' requires "
                "eigenvalue_method='power_method' to preserve eigenpair consistency"
            )
        else:
            raise ValueError(
                f"Unknown eigenvector_method: {eigenvector_method}. "
                "Choose from: 'inverse_iteration', 'power_method', 'eigenvector_from_lambda'"
            )

    # Form P and D matrices
    P = transpose(vectors)
    D = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        D[i][i] = eigenvalues[i]

    return P, D
