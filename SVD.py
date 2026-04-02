from utils import matrix_multiply, transpose, normalize
from QR import project
from diagonalization import diagonalize

EPSILON = 1e-10


def _mat_vec_mul(A, v):
	# A: m x n, v: n
	return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]


def _complete_orthonormal_basis(existing_cols, dim):
	# Build a full orthonormal basis in R^dim from existing vectors using Gram-Schmidt logic.
	basis = []

	for v in existing_cols:
		w = v[:]
		for u in basis:
			w = [w[i] - project(w, u)[i] for i in range(dim)]
		if sum(x * x for x in w) > EPSILON:
			basis.append(normalize(w))

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


def svd_decomposition(A, use_shift=False):
	# A: m x n matrix stored row-major (m row-vectors in R^n)
	# W: n x n matrix, W = A^T A
	# U: m x m orthogonal matrix (columns are left-singular vectors)
	# Sigma: m x n diagonal-rectangular matrix
	# V: n x n orthogonal matrix (columns are right-singular vectors)
	m = len(A)
	n = len(A[0])

	W = matrix_multiply(transpose(A), A)

	V, D = diagonalize(W, use_shift=use_shift)
	lambdas = [D[i][i] for i in range(n)]
	lambdas = [max(0.0, lam) for lam in lambdas]
	singular_values = [lam ** 0.5 for lam in lambdas]

	Sigma = [[0.0 for _ in range(n)] for _ in range(m)]
	for i in range(min(m, n)):
		Sigma[i][i] = singular_values[i]

	V_cols = transpose(V)

	u_cols = []
	for i in range(len(singular_values)):
		sigma = singular_values[i]
		if sigma > EPSILON:
			v_i = V_cols[i]
			Av = _mat_vec_mul(A, v_i)
			u_i = [x / sigma for x in Av]
			u_cols.append(normalize(u_i))

	full_u_cols = _complete_orthonormal_basis(u_cols, m)
	U = transpose(full_u_cols)

	full_v_cols = _complete_orthonormal_basis(V_cols, n)
	V = transpose(full_v_cols)

	return U, Sigma, V
