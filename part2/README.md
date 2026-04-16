# SVD and Diagonalization Toolkit

This folder contains matrix routines for eigenvalue computation, diagonalization, and SVD. The code is designed for correctness checks, method comparison, and benchmarking on small and medium matrices.

## 1. Folder Structure

- `utils.py`: basic matrix and vector routines such as transpose, addition, multiplication, normalization, identity, determinant helpers, and shape checks.
- `QR.py`: Gram-Schmidt orthogonalization, QR decomposition, similarity update, QR iteration, and shifted QR iteration.
- `diagonalization.py`: inverse iteration, power method, deflation, nullspace solve, and the main diagonalization pipeline.
- `SVD.py`: singular value decomposition built from the eigendecomposition of $A^T A$.
- `tests.ipynb`: notebook for small-matrix tests, NumPy comparisons, and benchmark plots.
- `requirements.txt`: Python packages needed to run the notebook.

## 2. Student Information

- **Võ Lân Tuấn**
  - MSSV: `24120240`
  - Đơn vị: Khoa Công nghệ Thông tin - HCMUS
- **Phí Hoàng Đức**
  - MSSV: `24120248`
  - Đơn vị: Khoa Công nghệ Thông tin - HCMUS
- **Môn học:** Toán Ứng Dụng và Thống Kê (MTH00051)
- **Học kỳ:** II - Năm học 2025-2026

## 3. Matrix Convention

The entire codebase uses one convention: each matrix row is a vector.

- If $A \in \mathbb{R}^{m \times n}$, then $A$ has $m$ row-vectors.
- Each row-vector lies in $\mathbb{R}^n$.
- Matrices are stored row-major as Python lists of rows.

Example:
$$
A =
\begin{bmatrix}
a_{00} & a_{01} \\
a_{10} & a_{11} \\
a_{20} & a_{21}
\end{bmatrix}
\in \mathbb{R}^{3 \times 2}
$$
means three row-vectors in $\mathbb{R}^2$.

When a routine needs column vectors, the implementation explicitly uses $A^T$ first.

## 4. Mathematical Overview

### 4.1 QR decomposition

For a square matrix $A \in \mathbb{R}^{n \times n}$, QR decomposition writes
$$
A = QR,
$$
where $Q$ is orthogonal and $R$ is upper triangular.

In this project, QR is built from Gram-Schmidt on the transposed row-major data. The decomposition is then used as the core step in QR iteration.

### 4.2 QR iteration

Starting from $A_0 = A$, the unshifted QR method repeats
$$
A_k = Q_k R_k, \qquad A_{k+1} = R_k Q_k.
$$
Because
$$
A_{k+1} = Q_k^T A_k Q_k,
$$
each iterate is similar to the previous one and therefore keeps the same eigenvalues.

As the iterations progress, $A_k$ becomes closer to upper triangular, and the diagonal entries approach the eigenvalues.

### 4.3 Shifted QR iteration

The shifted method improves convergence by subtracting a shift $\mu_k I$ before factorization:
$$
A_k - \mu_k I = Q_k R_k,
$$
then updating by
$$
A_{k+1} = R_k Q_k + \mu_k I.
$$

The shift typically reduces the number of iterations needed to isolate the eigenvalues.

### 4.4 Power method and deflation

For a dominant eigenpair, the power method iterates
$$
v_{k+1} = \frac{A v_k}{\|A v_k\|},
\qquad
\lambda_k = \frac{v_k^T A v_k}{v_k^T v_k}.
$$

If $|\lambda_1| > |\lambda_2|$, the method tends to converge to the eigenvector for $\lambda_1$.

To extract additional eigenpairs, the code deflates after each dominant pair is found:
$$
A \leftarrow A - \lambda v v^T.
$$

This approach is simple and useful for comparison, but it is usually less stable than QR-based methods on clustered spectra.

### 4.5 Inverse iteration

Given an approximate eigenvalue $\lambda$, inverse iteration repeatedly solves
$$
(A - \lambda I) y_{k+1} = x_k,
\qquad
x_{k+1} = \frac{y_{k+1}}{\|y_{k+1}\|}.
$$

The inverse step amplifies the eigenvector component associated with the shift $\lambda$. This is usually fast when the eigenvalue estimate is already close.

### 4.6 Nullspace solve from $(A - \lambda I)x = 0$

The fallback routine `eigenvector_from_lambda` builds the homogeneous system
$$
(A - \lambda I)x = 0,
$$
then row-reduces it, identifies free variables, and back-substitutes to obtain a nonzero vector in the nullspace.

This is used when inverse iteration is not sufficient or the matrix is numerically delicate.

### 4.7 Diagonalization

For a diagonalizable matrix $A$, the code seeks
$$
AP = PD,
$$
where the columns of $P$ are eigenvectors and $D$ is diagonal with the matching eigenvalues.

If $A$ is symmetric, the eigenvectors can be chosen orthonormal, giving the more stable form
$$
A = Q D Q^T.
$$

### 4.8 SVD via $A^T A$

For $A \in \mathbb{R}^{m \times n}$, the code forms
$$
W = A^T A \in \mathbb{R}^{n \times n}.
$$
Then it diagonalizes
$$
W = V D V^T,
$$
where the diagonal entries of $D$ are the eigenvalues of $W$.

The singular values are
$$
\sigma_i = \sqrt{\max(\lambda_i, 0)}.
$$
Right singular vectors are the columns of $V$, and for each $\sigma_i > 0$ the corresponding left singular vector is
$$
u_i = \frac{A v_i}{\sigma_i}.
$$

Finally, the code constructs
$$
A = U \Sigma V^T.
$$

This route is easy to implement and works well for comparison and benchmarking, especially when the internal eigenvalue method is configurable.

## 5. Numerical Error Metrics

The notebook checks correctness using small matrices and compares against NumPy where available.

- Eigen residual:
$$
\|A v - \lambda v\|_2
$$
- Diagonalization residual:
$$
\|AP - PD\|_F
$$
- SVD reconstruction residual:
$$
\|A - U\Sigma V^T\|_F
$$
- Orthogonality errors:
$$
\|U^T U - I\|_F, \qquad \|V^T V - I\|_F
$$

These appear directly in the notebook output so each test shows both the matrix used and the error against the NumPy reference where that reference exists.

## 6. Time Complexity

Let $n$ be the size of a square matrix and let $m \times n$ be a rectangular matrix for SVD.

### Per-step costs

- Matrix-vector multiply: $O(n^2)$.
- Dense matrix-matrix multiply: $O(n^3)$.
- Classical QR factorization: about $O(n^3)$.
- Dense linear solve: about $O(n^3)$.

### Method-level estimates

1. `qr`
   - One iteration costs about $O(n^3)$.
   - Total: about $O(k_{qr} n^3)$.

2. `qr_shift`
   - Same asymptotic cost per step as QR.
   - Usually converges in fewer iterations:
   $$
   O(k_{shift} n^3), \quad k_{shift} < k_{qr} \text{ typically}.
   $$

3. `power_method`
   - One dominant eigenpair: $O(k_{pow} n^2)$.
   - Multiple eigenpairs with deflation: roughly $O(k_{pow} n^3)$ in total.

4. `inverse_iteration`
   - Each refinement step solves a dense system, so $O(n^3)$ per step.
   - With a small fixed number of steps $s$, that is $O(s n^3)$ per vector.

5. SVD through $A^T A$
   - Build $A^T A$: $O(m n^2)$.
   - Diagonalize the $n \times n$ matrix: method-dependent, usually cubic.
   - Build $U$ from the right singular vectors: additional $O(m n^2)$ work.

The implementation is pure Python, so the constants matter a lot. That is why the notebook separates small-matrix correctness tests from larger-matrix benchmarks.

## 7. Recommended Method Choices

- Default eigenvalue path: `qr_shift`
- Default eigenvector path: `inverse_iteration`
- If using `power_method` for eigenvectors, use `power_method` for eigenvalues as well to keep eigenpair ordering consistent in the pipeline.
- Use `eigenvector_from_lambda` when you want the nullspace-based route for eigenvector construction.
- Use the benchmark cells for larger matrices only after the small-matrix correctness tests pass.

## 8. Notebook Workflow

Use `tests.ipynb` in this order:

1. Basic utility tests.
2. Determinant tests.
3. QR helper tests.
4. QR and shifted QR tests, including NumPy comparison.
5. Power method, deflation, and eigenvector helper tests.
6. Full diagonalization tests.
7. SVD tests, including NumPy comparison.
8. Benchmarks at the end.

That structure keeps the correctness checks on small matrices and reserves larger matrices for timing comparisons.
