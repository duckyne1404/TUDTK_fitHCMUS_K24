# ĐỒ ÁN 1: MA TRẬN VÀ CƠ SỞ CỦA TÍNH TOÁN KHOA HỌC
**Môn học:** Toán Ứng Dụng và Thống Kê (MTH00051)
**Học kỳ:** II - Năm học 2025-2026

---

## 👤 Thông tin sinh viên
* **Họ và tên:** Võ Lân Tuấn
* **MSSV:** 24120240
* **Đơn vị:** Khoa Công nghệ Thông tin - HCMUS

---

## 1. Cấu trúc project

- `utils.py`: các phép toán ma trận cơ bản.
- `QR.py`: Gram-Schmidt, QR decomposition, QR iteration, QR iteration có shift.
- `diagonalization.py`: tìm vector riêng và chéo hóa ma trận.
- `SVD.py`: phân rã giá trị kỳ dị.
- `tests.ipynb`: notebook kiểm thử, benchmark và ví dụ output.
- `requirements.txt`: thư viện cần cài để chạy notebook.

---

## 2. Quy ước dữ liệu ma trận

Quy ước thống nhất trong toàn bộ project là **mỗi hàng là một vector**.

- Với ma trận $A \in \mathbb{R}^{m \times n}$:
  - $A$ có $m$ vector hàng.
  - Mỗi vector hàng có $n$ phần tử.
- Khi một thuật toán cần làm việc theo cột, code sẽ ghi rõ bước `transpose(A)` trước khi xử lý.
- Quy ước này được giữ nhất quán trong README, code và notebook.

---

## 3. Cài đặt môi trường và chạy notebook

### 3.1 Tạo môi trường ảo

Trong thư mục project, chạy:

```bash
python3 -m venv .venv
```

Kích hoạt môi trường:

```bash
source .venv/bin/activate
```

Nếu muốn thoát môi trường ảo:

```bash
deactivate
```

### 3.2 Cài dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

File `requirements.txt` hiện chỉ cần:

- `numpy`
- `matplotlib`

### 3.3 Chạy `tests.ipynb`

1. Mở `tests.ipynb` trong VS Code.
2. Chọn kernel Python từ `.venv` vừa tạo.
3. Chạy các cell từ trên xuống dưới.
4. Không nên chạy lẻ các cell benchmark nếu chưa chạy cell import ở đầu notebook, vì notebook có logic reload module để luôn lấy code mới nhất từ file `.py`.

Thứ tự hợp lý là:

1. Cell import và helper.
2. Section 1: test hàm nhỏ.
3. Section 2: test QR.
4. Section 3: test chéo hóa.
5. Section 4: test SVD.
6. Section 5: benchmark QR shift / no-shift.
7. Section 5.5: benchmark SVD shift / no-shift.
8. Section 6: ví dụ output so sánh với NumPy.

---

## 4. QR decomposition và bài toán trị riêng

### 4.1 Mục tiêu

QR decomposition phân rã một ma trận vuông $A$ thành:

$$
A = QR
$$

trong đó:

- $Q$ có các cột trực chuẩn.
- $R$ là ma trận tam giác trên.

Từ QR iteration, ta lặp nhiều lần để đưa $A$ về gần ma trận chéo, từ đó đọc ra trị riêng xấp xỉ trên đường chéo.

### 4.2 Shape và ý nghĩa

- Đầu vào của các hàm trị riêng là $A \in \mathbb{R}^{n \times n}$.
- `qr_decomposition(A)` trả về:
  - `Q`: $n \times n$
  - `R`: $n \times n$
- `eig_qr(A)` và `eig_qrshift(A)` trả về:
  - danh sách trị riêng xấp xỉ
  - số vòng lặp đã dùng

### 4.3 Công thức Gram-Schmidt

Project dùng quy ước row-major, nên Gram-Schmidt được viết bằng cách transpose trước để xem các cột như các vector đầu vào.

Giả sử $A^T = [v_1, v_2, \dots, v_n]$ theo từng hàng của ma trận đã transpose, ta có:

$$
\nu_1 = v_1
$$

$$
\nu_j = v_j - \sum_{i=1}^{j-1} \mathrm{proj}_{q_i}(v_j)
$$

$$
q_j = \frac{\nu_j}{\|\nu_j\|}
$$

với

$$
\mathrm{proj}_{q_i}(v) = \frac{v \cdot q_i}{q_i \cdot q_i} q_i
$$

Nếu $\|\nu_j\|$ quá nhỏ, vector đó được bỏ qua để tránh chia cho số gần 0.

### 4.4 QR decomposition

Khi có các vector trực chuẩn $q_i$, ta xây dựng:

$$
R_{ij} = q_i \cdot a_j \quad (i \le j)
$$

và do đó:

$$
A = QR
$$

### 4.5 QR iteration không shift

Ý tưởng lặp:

1. Phân rã $A_k = Q_k R_k$.
2. Ghép lại $A_{k+1} = R_k Q_k$.
3. Lặp cho đến khi các phần tử đường chéo ổn định.

Về mặt toán học, $A_{k+1}$ đồng dạng với $A_k$, nên trị riêng được bảo toàn.

### 4.6 QR iteration có shift

Shift dùng để tăng tốc hội tụ:

1. Chọn shift $s_k$ từ phần tử đường chéo cuối.
2. Phân rã:

$$
A_k - s_k I = Q_k R_k
$$

3. Ghép lại:

$$
A_{k+1} = R_k Q_k + s_k I
$$

Shift thường giúp các trị riêng tách nhau nhanh hơn, nhưng với một số ma trận xấu điều kiện, phép trực chuẩn hóa có thể gặp vector gần 0.

### 4.7 Snippet code chính

```python
def gram_schmidt(A):
    A_T = transpose(A)
    orthonormal_rows = []

    for v in A_T:
        w = v[:]
        for u in orthonormal_rows:
            w = [w[i] - project(w, u)[i] for i in range(len(w))]
        if sum(x * x for x in w) > EPSILON:
            orthonormal_rows.append(normalize(w))

    return transpose(orthonormal_rows)
```

```python
def eig_qr(A):
    current = [row[:] for row in A]
    iterations = 0

    while not converged(current):
        Q, R = qr_decomposition(current)
        current = matrix_multiply(R, Q)
        iterations += 1

    return [current[i][i] for i in range(len(current))], iterations
```

### 4.8 Hàm tương ứng trong code

- `gram_schmidt(A)`
- `qr_decomposition(A)`
- `makesimilar(A)`
- `eig_qr(A)`
- `eig_qrshift(A)`

---

## 5. Chéo hóa ma trận

### 5.1 Mục tiêu

Tìm $P$ và $D$ sao cho:

$$
AP = PD
$$

Trong đó:

- $D$ là ma trận chéo chứa trị riêng.
- Cột thứ $i$ của $P$ là vector riêng tương ứng với $D_{ii}$.

Nếu $A$ chéo hóa được và $P$ khả nghịch, ta cũng có:

$$
A = P D P^{-1}
$$

### 5.2 Shape

- $A \in \mathbb{R}^{n \times n}$
- $P \in \mathbb{R}^{n \times n}$
- $D \in \mathbb{R}^{n \times n}$

### 5.3 Quy trình thuật toán

1. Dùng QR eigen để lấy trị riêng xấp xỉ.
2. Sắp xếp trị riêng giảm dần để kết quả ổn định hơn khi so sánh.
3. Với từng trị riêng $\lambda$:
   - Lập hệ $(A - \lambda I)x = 0$.
   - Ưu tiên inverse iteration để tìm vector riêng ổn định số hơn.
   - Nếu gặp hệ gần suy biến hoặc không rút được vector riêng hợp lệ, fallback sang RREF và back-substitution.
4. Chuẩn hóa vector riêng.
5. Ghép vector riêng thành $P$, đồng thời đặt các trị riêng lên đường chéo của $D$.

### 5.4 Vì sao cần inverse iteration

QR chỉ trả về trị riêng xấp xỉ. Nếu thay trực tiếp vào $(A - \lambda I)x = 0$, hệ có thể không còn đúng suy biến do sai số số học. Inverse iteration giúp “kéo” vector riêng về gần không gian riêng thật hơn trước khi fallback sang RREF.

### 5.5 Snippet code chính

```python
def diagonalize(A, use_shift=False):
    if use_shift:
        eigenvalues, _ = eig_qrshift(A)
    else:
        eigenvalues, _ = eig_qr(A)

    eigenvalues = sorted(eigenvalues, reverse=True)
    eigenvectors = [eigenvector_from_lambda(A, lam) for lam in eigenvalues]

    P = [list(col) for col in transpose(eigenvectors)]
    D = [[0.0 for _ in range(len(A))] for _ in range(len(A))]
    for i, lam in enumerate(eigenvalues):
        D[i][i] = lam

    return P, D
```

```python
def eigenvector_from_lambda(A, lam):
    try:
        return _inverse_iteration(A, lam)
    except Exception:
        return _back_substitute_null_vector(*solve_rref(A_minus_lambda_I))
```

### 5.6 Hàm tương ứng trong code

- `solve_rref(A)`
- `_inverse_iteration(A, lam, ...)`
- `eigenvector_from_lambda(A, lam)`
- `diagonalize(A, use_shift=False)`

---

## 6. Singular Value Decomposition (SVD)

### 6.1 Mục tiêu

Phân rã một ma trận $A$ thành:

$$
A = U \Sigma V^T
$$

Trong đó:

- $U$ là ma trận trực giao $m \times m$.
- $\Sigma$ là ma trận chéo chữ nhật $m \times n$.
- $V$ là ma trận trực giao $n \times n$.

### 6.2 Shape

- $A \in \mathbb{R}^{m \times n}$
- $W = A^T A \in \mathbb{R}^{n \times n}$
- $U \in \mathbb{R}^{m \times m}$
- $\Sigma \in \mathbb{R}^{m \times n}$
- $V \in \mathbb{R}^{n \times n}$

### 6.3 Công thức nền tảng

Từ $W = A^T A$:

$$
W v_i = \lambda_i v_i
$$

Ta suy ra singular value:

$$
\sigma_i = \sqrt{\max(\lambda_i, 0)}
$$

Với $\sigma_i > 0$:

$$
 u_i = \frac{1}{\sigma_i} A v_i
$$

Khi số cột chưa đủ để tạo ma trận vuông trực giao, project sẽ bổ sung thêm vector trực chuẩn để hoàn tất cơ sở.

### 6.4 Quy trình thuật toán

1. Tính $W = A^T A$.
2. Diagonalize $W$ để lấy trị riêng và vector riêng.
3. Sắp xếp trị riêng giảm dần và lấy $\sigma_i = \sqrt{\lambda_i}$.
4. Xây dựng $\Sigma$ với các $\sigma_i$ nằm trên đường chéo.
5. Lấy các vector riêng của $W$ làm cột của $V$.
6. Tính các vector trái $u_i = Av_i / \sigma_i$ với $\sigma_i > 0$.
7. Hoàn thiện $U$ và $V$ thành ma trận trực giao đầy đủ bằng Gram-Schmidt.

### 6.5 Snippet code chính

```python
def svd_decomposition(A, use_shift=False):
    m = len(A)
    n = len(A[0])

    W = matrix_multiply(transpose(A), A)
    V, D = diagonalize(W, use_shift=use_shift)

    lambdas = [max(0.0, D[i][i]) for i in range(n)]
    singular_values = [lam ** 0.5 for lam in lambdas]

    Sigma = [[0.0 for _ in range(n)] for _ in range(m)]
    for i in range(min(m, n)):
        Sigma[i][i] = singular_values[i]

    V_cols = transpose(V)
    u_cols = []
    for i, sigma in enumerate(singular_values):
        if sigma > EPSILON:
            Av = _mat_vec_mul(A, V_cols[i])
            u_cols.append(normalize([x / sigma for x in Av]))

    U = transpose(_complete_orthonormal_basis(u_cols, m))
    V = transpose(_complete_orthonormal_basis(V_cols, n))
    return U, Sigma, V
```

### 6.6 Hàm tương ứng trong code

- `svd_decomposition(A, use_shift=False)`
- `_complete_orthonormal_basis(...)`
- `_mat_vec_mul(A, v)`
- `project(...)` từ `QR.py`

---

## 7. Notebook kiểm thử và benchmark

Notebook `tests.ipynb` được chia theo tầng để dễ kiểm tra:

1. **Hàm nhỏ**: `transpose`, `matrix_multiply`, `matrix_add`, `matrix_subtract`, `normalize`, `_dot`, `project`.
2. **QR**: `gram_schmidt`, `qr_decomposition`, `makesimilar`, `eig_qr`, `eig_qrshift`.
3. **Chéo hóa**: `solve_rref`, `eigenvector_from_lambda`, `diagonalize`.
4. **SVD**: shape, trực giao, tái tạo ma trận, singular values, quan hệ $Av_i = \sigma_i u_i$.
5. **Benchmark QR**: so sánh `eig_qr` và `eig_qrshift` trên cùng tập ma trận đối xứng ngẫu nhiên.
6. **Benchmark SVD**: so sánh `svd_decomposition(A, use_shift=False)` và `svd_decomposition(A, use_shift=True)`.
7. **Ví dụ output**: in kết quả với các ma trận nhỏ và lớn để đối chiếu trực tiếp với NumPy.

Lưu ý: NumPy được dùng để **đối chiếu và kiểm thử**, không phải là benchmark chính trong phần benchmark SVD hiện tại.

---

## 8. Cách đọc kết quả benchmark

- Benchmark QR đo số vòng lặp và thời gian trung bình giữa bản không shift và bản có shift.
- Benchmark SVD đo cùng thứ đó trên hai biến thể của chính thuật toán SVD:
  - `use_shift=False`
  - `use_shift=True`
- Độ lệch tái tạo $\|A - U\Sigma V^T\|$ giúp nhìn mức ổn định số học khi kích thước tăng lên.

---

## 9. Ghi chú giới hạn hiện tại

- Gram-Schmidt cổ điển vẫn nhạy cảm số học hơn Householder hoặc modified Gram-Schmidt.
- Shifted QR có thể gặp thất bại với một số đầu vào xấu điều kiện do vector trung gian gần 0.
- Sai số tái tạo của SVD tăng khi kích thước ma trận lớn hơn, do tích lũy sai số từ các bước trực chuẩn hóa và phân rã lặp.
- Đây là giới hạn tự nhiên của phiên bản cài đặt thuần Python hiện tại, không phải lỗi của notebook kiểm thử.
