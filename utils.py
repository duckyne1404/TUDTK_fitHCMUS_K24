import math

# just to validate that the input is a non-empty rectangular matrix and return its shape
def _validate_non_empty_rectangular_matrix(matrix, name="matrix"):
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not all(isinstance(row, list) for row in matrix):
        raise ValueError(f"{name} must be a matrix (list of lists)")

    n_cols = len(matrix[0])
    if n_cols == 0:
        raise ValueError(f"{name} must have at least one column")

    for row in matrix:
        if len(row) != n_cols:
            raise ValueError(f"{name} must be rectangular")

    return len(matrix), n_cols

# just to validate that the input is a square matrix and return its size
def _validate_square_matrix(matrix, name="matrix"):
    if not isinstance(matrix, list):
        raise ValueError(f"{name} must be a matrix (list of lists)")
    if not all(isinstance(row, list) for row in matrix):
        raise ValueError(f"{name} must be a matrix (list of lists)")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError(f"{name} must be square (n x n)")

    return n

# return A^T
def transpose(A):
    m, n = _validate_non_empty_rectangular_matrix(A, "A")
    AT = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(m):
        for j in range(n):
            AT[j][i] = A[i][j]
    return AT

# return C = A * B
def matrix_multiply(A, B):
    m, n = _validate_non_empty_rectangular_matrix(A, "A")
    b_rows, p = _validate_non_empty_rectangular_matrix(B, "B")
    
    if n != b_rows:
        raise ValueError("Number of columns in A must be equal to number of rows in B")
    
    C = [[0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

# return Identity matrix of size n
def identity_matrix(n):
    if n < 0:
        raise ValueError("Matrix size must be non-negative")

    I = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        I[i][i] = 1.0
    return I

# return normalized vector
def normalize(vector):
    norm = math.sqrt(sum(x ** 2 for x in vector))
    if norm < 1e-10:
        raise ValueError("Cannot normalize a zero vector")
    return [x / norm for x in vector]

# get shape of a matrix
def shape(matrix):
    return _validate_non_empty_rectangular_matrix(matrix, "matrix")

# calculate determinant of a nxn matrix using Laplace O(n!) -> super slow
def laplace_get_determinant(M):
    n = _validate_square_matrix(M, "M")
    if n == 0:
        return 1
    if n == 1: return M[0][0]
    if n == 2: return M[0][0]*M[1][1] - M[0][1]*M[1][0]
    
    det = 0
    for j in range(n):
        minor = [row[:j] + row[j+1:] for row in M[1:]] # choose row 1
        det += ((-1)**j) * M[0][j] * laplace_get_determinant(minor)
    return det

# calculate determinant of a nxn matrix using Gaussian elimination O(n^3) -> much faster
def gauss_get_determinant(M):
    n = _validate_square_matrix(M, "M")
    if n == 0:
        return 1

    A = [row[:] for row in M]
    det = 1
    for i in range(n):
        pivot = i
        while pivot < n and abs(A[pivot][i]) < 1e-10:
            pivot += 1
        if pivot == n:
            return 0  # det = 0 if we have a zero column
        
        if pivot != i:
            A[i], A[pivot] = A[pivot], A[i]  # Swap rows
            det *= -1  # swap rows changes the sign of the determinant
        
        det *= A[i][i]
        
        # eliminate below
        for j in range(i + 1, n):
            factor = A[j][i] / A[i][i]
            for k in range(i, n):
                A[j][k] -= factor * A[i][k]
    
    return det