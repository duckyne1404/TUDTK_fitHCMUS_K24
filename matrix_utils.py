import math
import random

def print_matrix(M, name, precision = 4):
    print(name)
    for row in M:
        print("[ " + "  ".join(f"{x: .{precision}f}" for x in row) + "]")
    print()

def transpose(A):
    rows = len(A)
    cols = len(A[0])

    AT = [[0 for _ in range(rows)] for _ in range(cols)]

    for i in range (rows):
        for j in range(cols):
            AT[j][i] = A[i][j]
    
    return AT

def matrix_multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])

    if (cols_A != rows_B):
        raise ValueError("Kích thước ma trận không khớp để thực hiện phép nhân ma trận")

    C = [[0 for _ in range (cols_B)] for _ in range (rows_A)]

    for i in range (rows_A):
        for j in range (cols_B):
            for k in range (cols_A):
                C[i][j] += A[i][k] * B[k][j]
    
    return C

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def vector_norm(v):
    return math.sqrt(sum(x**2 for x in v))

def normalize(v):
    n = vector_norm(v)
    if (n == 0):
        raise ValueError("Cannot normalize zero vector")
    return [x / n for x in v]


def matrix_multiply_vector(A, v):
    if len(A) == 0:
        raise ValueError("Ma trận A rỗng")

    cols = len(A[0])

    if len(v) != cols:
        raise ValueError("Số cột của ma trận phải bằng kích thước vector")
    return [dot_product(row, v) for row in A]

def distance(v1, v2):
    if len(v1) != len(v2):
        raise ValueError("Số không gian 2 vector khác nhau")

    sum_sq = sum((a - b) ** 2 for a, b in zip(v1, v2))
    return math.sqrt(sum_sq)

def generate_symmetric_matrix(n):
    """Tạo ma trận đối xứng để đảm bảo hội tụ thực"""
    A = [[random.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            val = (A[i][j] + A[j][i]) / 2
            A[i][j] = A[j][i] = val
    return A

def outer_product(v1, v2):
    n = len(v1)
    m = len(v2)
    
    return [[v1[i] * v2[j] for j in range(m)] for i in range(n)]

def matrix_subtract(A, B):
    rows = len(A)
    cols = len(A[0])

    C = [[0]*cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            C[i][j] = A[i][j] - B[i][j]

    return C
