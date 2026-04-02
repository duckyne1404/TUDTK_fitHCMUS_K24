from utils import *
from QR import eig_qr, eig_qrshift

EPSILON = 1e-10

def solve_rref(A):
    m, n = shape(A)
    row = 0
    pivot_cols = []
    for col in range(n):
        if row >= m:
            break
        
        # find pivot row
        pivot_row = row
        while pivot_row < m and abs(A[pivot_row][col]) < EPSILON:
            pivot_row += 1
        
        if pivot_row == m:
            continue  # no pivot in this column, move to next column
        # swap current row with pivot row
        A[row], A[pivot_row] = A[pivot_row], A[row]
        pivot_cols.append(col)
        # normalize the pivot row
        pivot_value = A[row][col]
        for j in range(col, n):
            A[row][j] /= pivot_value
        
        # eliminate below
        for r in range(row + 1, m):
            factor = A[r][col]
            for j in range(col, n):
                A[r][j] -= factor * A[row][j]
        
        row += 1
    
    return A, pivot_cols


def _solve_linear_system(A, b):
    # Solve A x = b by Gaussian elimination on the augmented matrix.
    # A: n x n matrix, b: length-n vector
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
    # A: n x n matrix stored row-major
    # lam: approximate eigenvalue
    # return: normalized approximate eigenvector
    n = len(A)
    x = [0.0 for _ in range(n)]
    x[seed_index % n] = 1.0

    for _ in range(steps):
        best_candidate = None
        for delta in (1e-8, -1e-8, 1e-6, -1e-6, 1e-4, -1e-4):
            shifted = matrix_subtract(A, scalar_multiply_matrix(identity_matrix(n), lam + delta))
            try:
                y = _solve_linear_system([row[:] for row in shifted], x)
                best_candidate = y
                break
            except ValueError:
                continue

        if best_candidate is None:
            raise ValueError("Inverse iteration failed to solve the shifted system")

        x_new = normalize(best_candidate)
        if sum(abs(x_new[i] - x[i]) for i in range(n)) < 1e-8:
            x = x_new
            break
        x = x_new

    return x


def _back_substitute_null_vector(U, pivot_cols):
    # U: row-echelon matrix of (A - lambda I), row-major
    # x: one non-zero solution of Ux = 0
    n = len(U[0])
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n) if c not in pivot_set]

    if not free_cols:
        raise ValueError("No free variable found when solving (A - lambda I)x = 0")

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
    # A: n x n matrix stored row-major
    # lam: one eigenvalue estimate
    # return: one normalized eigenvector x in R^n
    n = len(A)
    try:
        return _inverse_iteration(A, lam, seed_index=0)
    except ValueError:
        I = identity_matrix(n)
        shifted = matrix_subtract(A, scalar_multiply_matrix(I, lam))
        echelon, pivot_cols = solve_rref([row[:] for row in shifted])
        return _back_substitute_null_vector(echelon, pivot_cols)


def diagonalize(A, use_shift=False):
    # A: n x n matrix stored row-major (each row is one vector in R^n)
    # use_shift: if True, use eig_qrshift (with Rayleigh quotient shift) for faster eigenvalue convergence
    #            if False, use standard eig_qr (default)
    # eigenvalues: sorted descending
    # P: n x n matrix whose columns are eigenvectors
    # D: n x n diagonal matrix, D[i][i] = eigenvalues[i]
    if use_shift:
        eigenvalues, _ = eig_qrshift(A)
    else:
        eigenvalues, _ = eig_qr(A)
    eigenvalues = sorted(eigenvalues, reverse=True)

    vectors = []
    for i, lam in enumerate(eigenvalues):
        try:
            vectors.append(_inverse_iteration(A, lam, seed_index=i))
        except ValueError:
            vectors.append(eigenvector_from_lambda(A, lam))
    P = transpose(vectors)

    n = len(eigenvalues)
    D = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        D[i][i] = eigenvalues[i]

    return P, D