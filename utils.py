import math

def _validate_non_empty_rectangular_matrix(matrix, name="matrix"):
    """
    Validate that input is a non-empty rectangular matrix.
    Input: matrix (m x n, m rows, n columns, each row is a vector in R^n)
    Returns: (m, n) tuple
    """
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

def _validate_square_matrix(matrix, name="matrix"):
    """
    Validate that input is a square matrix.
    Shape: n x n (n rows, n columns)
    Returns: n (size)
    """
    if not isinstance(matrix, list):
        raise ValueError(f"{name} must be a matrix (list of lists)")
    if not all(isinstance(row, list) for row in matrix):
        raise ValueError(f"{name} must be a matrix (list of lists)")

    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError(f"{name} must be square (n x n)")

    return n

def transpose(A):
    """
    Transpose a matrix.
    Input: A (m, n)
    Returns: A^T (n x m)
    """
    m, n = _validate_non_empty_rectangular_matrix(A, "A")
    AT = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(m):
        for j in range(n):
            AT[j][i] = A[i][j]
    return AT

def matrix_add(A, B):
    """
    Add two matrices of the same shape.
    Input: A (m, n) + B (m, n)
    Returns: C (m, n) = A (m, n) + B (m, n)
    """
    m, n = _validate_non_empty_rectangular_matrix(A, "A")
    b_rows, b_cols = _validate_non_empty_rectangular_matrix(B, "B")

    if m != b_rows or n != b_cols:
        raise ValueError("Matrices A and B must have the same dimensions for addition")

    C = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            C[i][j] = A[i][j] + B[i][j]
    return C

def matrix_subtract(A, B):
    """
    Subtract two matrices of the same shape.
    Input: A (m, n) - B (m, n)
    Returns: C (m, n) = A (m, n) - B (m, n)
    """
    m, n = _validate_non_empty_rectangular_matrix(A, "A")
    b_rows, b_cols = _validate_non_empty_rectangular_matrix(B, "B")
    
    if m != b_rows or n != b_cols:
        raise ValueError("Matrices A and B must have the same dimensions for subtraction")
    
    C = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            C[i][j] = A[i][j] - B[i][j]
    return C

def matrix_multiply(A, B):
    """
    Dot product of two matrices.
    Input: A (m, n) * B (n, p)
    Returns: C (m, p) = A (m, n) * B (n, p)
    """
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

def scalar_multiply_matrix(A, scalar):
    """
    Multiply a matrix by a scalar.
    Input: A (m, n) * scalar
    Returns: C (m, n) = A (m, n) * scalar
    """
    m, n = _validate_non_empty_rectangular_matrix(A, "A")
    C = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            C[i][j] = A[i][j] * scalar
    return C

def identity_matrix(n):
    """
    Create an identity matrix.    
    Input: n (size of the square matrix)
    Returns: I (n x n identity matrix)
    """
    if n < 0:
        raise ValueError("Matrix size must be non-negative")

    I = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        I[i][i] = 1.0
    return I

def normalize(vector):
    """
    Normalize a vector to unit length.
    Input: unnormalized vector v (list of numbers)
    Returns: v / ||v||
    """
    norm = math.sqrt(sum(x ** 2 for x in vector))
    if norm < 1e-32:
        raise ValueError("Cannot normalize a zero vector")
    return [x / norm for x in vector]

def shape(matrix):
    """
    Get the shape of a matrix.
    Input: matrix (list of lists)
    Returns: (m, n) tuple (number of rows, number of columns)
    """
    return _validate_non_empty_rectangular_matrix(matrix, "matrix")

def laplace_get_determinant(M):
    """
    Calculate determinant using Laplace expansion.
    Complexity: O(n!) - VERY SLOW
    Input: M (n, n) square matrix
    Returns: det(M)
    """
    n = _validate_square_matrix(M, "M")
    if n == 0:
        return 1
    if n == 1: return M[0][0]
    if n == 2: return M[0][0]*M[1][1] - M[0][1]*M[1][0]
    
    det = 0
    for j in range(n):
        minor = [row[:j] + row[j+1:] for row in M[1:]]
        det += ((-1)**j) * M[0][j] * laplace_get_determinant(minor)
    return det

def gauss_get_determinant(M):
    """
    Calculate determinant using Gaussian elimination.
    Complexity: O(n^3)
    Input: M (n, n) square matrix
    Returns: det(M)
    """
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
