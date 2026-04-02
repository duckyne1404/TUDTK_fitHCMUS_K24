from utils import identity_matrix, normalize, matrix_add, matrix_subtract, matrix_multiply, scalar_multiply_matrix, transpose

def _dot(u, v):
    return sum(u[i] * v[i] for i in range(len(u)))


def project(u, v):
    # project u onto v
    # return (u . v) / (v . v) * v
    dot_uv = _dot(u, v)
    dot_vv = _dot(v, v)
    if abs(dot_vv) < 1e-10:
        raise ValueError("Cannot project onto the zero vector")
    scalar = dot_uv / dot_vv
    return [scalar * v[i] for i in range(len(v))]

def gram_schmidt(A):
    # A: m x n matrix stored row-major, where each row is one vector in R^n
    # A_T = transpose(A): n x m matrix, so each row of A_T is one original column vector of A
    # basis: orthonormal vectors obtained from Gram-Schmidt
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
    # A: n x n matrix stored (ma trận hàng)
    # Q: n x n matrix whose columns are orthonormal vectors
    # R: n x n upper-triangular matrix such that A = Q * R
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

# A = Q * R -> return B = R * Q
def makesimilar(A):
    # A_0 = Q_0 * R_0
    # A_1 = R_0 * Q_0
    # A_1 = Q_1 * R_1
    # A_2 = R_1 * Q_1
    # ...
    # A_k = R_{k-1} * Q_{k-1}
    # A_k = Q_k * R_k
    # Eventually A_k will converge to an upper triangular matrix with the eigenvalues of A on the diagonal.
    # A: n x n matrix stored (ma trận hàng)
    # Q: n x n orthonormal-column matrix
    # R: n x n upper-triangular matrix
    # B: n x n similar matrix, B = R * Q
    Q, R = qr_decomposition(A)
    B = matrix_multiply(R, Q)
    return B

# return (eigenvalues, iters) where eigeinvalues is a list of the eigenvalues of A and iters is the number of iterations until convergence
# time complexity: kinda slow (calculate precise later)
def eig_qr(A):
    # A: n x n matrix stored row-major
    # B: n x n iterated similar matrix
    B = makesimilar(A)
    iters = 0
    leig = B[-1][-1]
    diff = 1
    max_iters = 2000
    while diff > 1e-10 and iters < max_iters:
        B = makesimilar(B)
        iters += 1
        diff = abs(B[-1][-1] - leig)
        leig = B[-1][-1]

    return [B[i][i] for i in range(len(B))], iters

# QR algorithms with Shift
# A_0 - s*I = Q_0 * R_0
# A_1 = R_0 * Q_0 + s*I
# ...
# A_k - s*I = Q_k * R_k
# A_{k+1} = R_k * Q_k + s*I
# s is chosen at every iteration to accelerate convergence, often close to the value of an eigenvalue, usually the bottom-right element of A_k, which is an approximation of an eigenvalue. This helps to speed up convergence, especially for eigenvalues that are close together. The algorithm continues until the off-diagonal elements of A_k are sufficiently small, indicating that A_k has converged to an upper triangular matrix with the eigenvalues on the diagonal.
def eig_qrshift(A, shindex=-1):
    # parameter shindex: shift index (choose which diagonal element to use as the shift, default is the bottom-right element) - just for visualization & speed demonstration
    # A: n x n matrix (ma trận hàng)
    # B: n x n iterated similar matrix
    # I: n x n identity matrix
    # shift: n x n matrix equal to leigh * I
    # C: n x n shifted matrix, C = B - shift
    B = makesimilar(A)
    iters = 0
    leigh = B[shindex][shindex]
    diff = 1
    max_iters = 2000
    while diff > 1e-10 and iters < max_iters:
        I = identity_matrix(len(B))
        shift = scalar_multiply_matrix(I, leigh)
        C = matrix_subtract(B, shift) # B_old - s*I = C = Q * R
        B = matrix_add(makesimilar(C), shift) # now make B_new = R * Q (makesimilar(C)) + shift
        iters += 1
        diff = abs(B[shindex][shindex] - leigh)
        leigh = B[shindex][shindex]

    return [B[i][i] for i in range(len(B))], iters