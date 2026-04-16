# Project 1 - Applied Math and Statistics

README này tập trung mô tả hai thư mục đã hoàn thiện là `part1/` và `part2/`.

`part3/` và `report/` hiện chưa được tài liệu hóa theo yêu cầu và sẽ bổ sung sau.

## 1) Cấu trúc thư mục tổng quan (phần hiện có)

```text
project1/
├── README.md
├── requirements.txt
├── part1/
│   ├── determinant.py
│   ├── gaussian.py
│   ├── inverse.py
│   ├── rank_and_basic.py
│   ├── verify.py
│   ├── test.py
│   └── part1_demo.ipynb
└── part2/
    ├── utils.py
    ├── QR.py
    ├── diagonalization.py
    ├── SVD.py
    ├── tests.ipynb
    └── README.md
```

## 2) Mô tả chi tiết `part1/`

`part1/` triển khai các phép toán đại số tuyến tính cơ bản bằng Python thuần (list of lists), gồm định thức, khử Gauss, nghịch đảo, hạng và các không gian cơ sở.

### `part1/determinant.py`
- Chức năng: tính định thức bằng khử Gauss với partial pivoting.
- Hàm quan trọng:
  - `determinant(A)`: trả về `det(A)`; theo dõi số lần đổi dòng để sửa dấu định thức.
- Ghi chú: dùng ngưỡng `EPS` để xử lý số gần 0.

### `part1/gaussian.py`
- Chức năng: giải hệ phương trình tuyến tính `Ax=b` bằng khử Gauss.
- Hàm quan trọng:
  - `gaussian_eliminate(A, b)`: khử tiến tạo ma trận tam giác trên `U`, rồi gọi thế lùi để tìm nghiệm.
  - `back_substitution(U, c)`: giải hệ tam giác trên.
- Kết quả chính: trả về `U`, nghiệm `x`, và số lần đổi dòng.

### `part1/inverse.py`
- Chức năng: tính ma trận nghịch đảo bằng Gauss-Jordan trên ma trận ghép `[A | I]`.
- Hàm quan trọng:
  - `inverse(A)`: trả về `A^{-1}` hoặc `None` nếu ma trận suy biến.

### `part1/rank_and_basic.py`
- Chức năng: tìm hạng và cơ sở của các không gian cơ bản (row/column/null space).
- Hàm quan trọng:
  - `rref_with_pivots(A)`: đưa ma trận về RREF và lấy danh sách cột pivot.
  - `rank_and_basis(A)`: trả về `(rank, row_basis, col_basis, null_basis)`.
  - `clean_row(row)`, `is_zero_row(row)`: hàm phụ để làm sạch số học và nhận diện dòng 0.

### `part1/verify.py`
- Chức năng: kiểm chứng kết quả bằng NumPy.
- Hàm quan trọng:
  - `verify_solution(A, x, b)`
  - `verify_inverse(A, A_inv)`
  - `verify_determinant(A, det_custom)`
  - `verify_rank(A, rank_custom)`

### `part1/test.py`
- Chức năng: script chạy test tổng hợp cho nhiều bộ dữ liệu khó (Hilbert, near-singular, scale lớn/nhỏ, random, phụ thuộc tuyến tính).
- Hàm quan trọng:
  - `run_test(name, A, b)`: gọi toàn bộ pipeline và in kết quả.
  - `main()`: dựng bộ test và thực thi.

### `part1/part1_demo.ipynb`
- Chức năng: notebook demo/thực nghiệm cho các hàm của phần 1.
- Nội dung chính: import module, chạy minh họa, đối chiếu kết quả.

## 3) Mô tả chi tiết `part2/`

`part2/` tập trung vào phân tích trị riêng/véc-tơ riêng, chéo hóa và SVD, cũng triển khai thuần Python để dễ theo dõi thuật toán.

### `part2/utils.py`
- Chức năng: bộ tiện ích toán ma trận/vector dùng chung.
- Hàm quan trọng:
  - Kiểm tra dữ liệu: `_validate_non_empty_rectangular_matrix`, `_validate_square_matrix`, `shape`.
  - Phép toán cơ bản: `transpose`, `matrix_add`, `matrix_subtract`, `matrix_multiply`, `scalar_multiply_matrix`, `identity_matrix`, `normalize`.
  - Định thức tham khảo: `laplace_get_determinant` (chậm), `gauss_get_determinant` (O(n^3)).

### `part2/QR.py`
- Chức năng: QR decomposition và QR algorithm để tìm eigenvalues.
- Hàm quan trọng:
  - `gram_schmidt(A)`: trực chuẩn hóa cột.
  - `qr_decomposition(A)`: phân rã `A = QR`.
  - `makesimilar(A)`: một bước lặp tương đương `A -> RQ`.
  - `eig_qr(A, max_iters=...)`: QR không shift.
  - `eig_qrshift(A, shindex=-1, max_iters=...)`: QR có Rayleigh shift, hội tụ nhanh hơn.
  - `project(u, v)`: phép chiếu vector.

### `part2/diagonalization.py`
- Chức năng: pipeline chéo hóa ma trận đối xứng với nhiều chiến lược trị riêng/véc-tơ riêng.
- Hàm quan trọng:
  - `power_iteration(A, ...)` + `deflation(A, lam, v)`.
  - `_inverse_iteration(A, lam, ...)` và `_solve_linear_system(A, b)`.
  - `solve_rref(A)`, `_back_substitute_null_vector(U, pivot_cols)`, `eigenvector_from_lambda(A, lam)`.
  - `diagonalize(A, eigenvalue_method='qr_shift', eigenvector_method='inverse_iteration')`: hàm trung tâm, trả về `(P, D)`.

### `part2/SVD.py`
- Chức năng: SVD theo hướng chéo hóa `A^T A`.
- Hàm quan trọng:
  - `svd_decomposition(A, eigenvalue_method='qr_shift', eigenvector_method='inverse_iteration')`: trả về `(U, Sigma, V)` sao cho `A ≈ U * Sigma * V^T`.
  - `_mat_vec_mul`, `_complete_orthonormal_basis`, `_extend_basis_preserving`: hàm phụ để dựng cơ sở trực chuẩn ổn định.

### `part2/tests.ipynb`
- Chức năng: notebook test, đối chiếu NumPy và benchmark thời gian cho các thuật toán part2.

### `part2/README.md`
- Chức năng: tài liệu học thuật chi tiết cho phần 2 (ý tưởng toán học, độ phức tạp, workflow notebook, khuyến nghị phương pháp).

## 4) requirements

File `requirements.txt` đang chứa:
- `numpy>=1.21.0`
- `matplotlib`