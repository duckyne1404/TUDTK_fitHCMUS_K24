import random
import math
from matrix_utils import (
    matrix_multiply_vector, dot_product, vector_norm, normalize, outer_product, matrix_subtract,
    print_matrix, matrix_multiply, transpose
)
def power_iteration(A, iterations=1000, epsilon=1e-9):
    if not A or len(A) != len(A[0]):
        raise ValueError("Ma trận phải là ma trận vuông và không rỗng!")
    
    n = len(A)
    v = [random.uniform(-1, 1) for _ in range(n)]
    v = normalize(v)
    v_history = [v]

    for i in range(iterations):
        Av = matrix_multiply_vector(A, v)
        eigenvalue = dot_product(v, Av) 
        
        norm_Av = vector_norm(Av)
        if norm_Av < 1e-15:
            return 0.0, v
            
        v_next = [x / norm_Av for x in Av]

        # Kiểm tra hội tụ chuẩn
        if 1 - abs(dot_product(v, v_next)) < epsilon:
            # print(f"Hội tụ tại vòng lặp {i}")
            return eigenvalue, v_next

        # Kiểm tra Ping-pong
        if len(v_history) >= 2:
            if 1 - abs(dot_product(v_next, v_history[-2])) < epsilon:
                # print(f"Phát hiện lỗi Ping-pong tại vòng lặp {i}")
                return norm_Av, v_next

        v = v_next
        v_history.append(v)
        if len(v_history) > 2: v_history.pop(0)

    # print(f"Cảnh báo: Không hội tụ hoàn toàn. Đã chạy hết {iterations} vòng lặp")
    return eigenvalue, v


def deflation(A, lam, v):
    vvT = outer_product(v, v)
    n = len(A) 
    scaled = [[lam * vvT[i][j] for j in range(n)] for i in range(n)]

    return matrix_subtract(A, scaled)

def eigen_decomposition(A):
    n = len(A)

    A_work = [row[:] for row in A]

    eigenvalues = []
    eigenvectors = []

    for _ in range(n):
        lam, v = power_iteration(A_work)

        eigenvalues.append(lam)
        eigenvectors.append(v)

        A_work = deflation(A_work, lam, v)
    
    return eigenvalues, eigenvectors


def build_PD(eigenvalues, eigenvectors):
    n = len(eigenvalues)

    # P matrix (eigenvectors làm cột)
    P = [[eigenvectors[j][i] for j in range(n)] for i in range(n)]

    # D matrix
    D = [[0]*n for _ in range(n)]
    for i in range(n):
        D[i][i] = eigenvalues[i]

    return P, D

def matrix_inverse(A):
    n = len(A)

    # tạo ma trận ghép [A | I]
    M = [row[:] + [1 if i == j else 0 for j in range(n)] 
         for i, row in enumerate(A)]

    for i in range(n):

        # ===== Partial Pivoting =====
        max_row = i
        max_val = abs(M[i][i])

        for r in range(i+1, n):
            if abs(M[r][i]) > max_val:
                max_val = abs(M[r][i])
                max_row = r

        if max_val == 0:
            raise ValueError("Matrix is singular")

        # swap rows
        if max_row != i:
            M[i], M[max_row] = M[max_row], M[i]

        # ===== Normalize pivot row =====
        pivot = M[i][i]
        for j in range(2*n):
            M[i][j] /= pivot

        # ===== Eliminate other rows =====
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(2*n):
                    M[k][j] -= factor * M[i][j]

    # lấy phần nghịch đảo
    inv = [row[n:] for row in M]

    return inv

def matrix_error(A, B):
    rows = len(A)
    cols = len(A[0])

    err = 0
    for i in range(rows):
        for j in range(cols):
            err += abs(A[i][j] - B[i][j])

    return err

def demo_diagonalization(A, P, D):
    P_inv = matrix_inverse(P)

    PD = matrix_multiply(P, D)
    A_reconstructed = matrix_multiply(PD, P_inv)

    print("\n=== Diagonalization Demo ===\n")

    print_matrix(A, "A =")
    print_matrix(P, "P =")
    print_matrix(D, "D =")
    print_matrix(P_inv, "P⁻¹ =")

    print("Check: A ≈ P D P⁻¹\n")

    print_matrix(A_reconstructed, "P D P⁻¹ =")

    err = matrix_error(A, A_reconstructed)
    print(f"Sai số tái cấu trúc (Reconstruction error): {err:.2e}")


# Hàm xây SVD

# Hàm gán sigma tạo ma trận sigma
def build_sigma(singular_values, m, n):
    Sigma = [[0]*n for _ in range(m)]

    for i in range(min(len(singular_values), m, n)):
        Sigma[i][i] = singular_values[i]

    return Sigma

def svd(A):
    AT = transpose(A)
    B = matrix_multiply(AT, A)

    evals, evecs = eigen_decomposition(B)

    # 1. Sắp xếp cặp trị riêng, vector riêng theo trị riêng giảm dần
    pairs = list(zip(evals, evecs))
    pairs.sort(key = lambda x: x[0], reverse = True)

    sorted_evals = [p[0] for p in pairs]
    sorted_evecs = [p[1] for p in pairs]

    # 2. Tính Singular Values (sigma = sqrt(|lambda|))
    singular_values = [math.sqrt(max(0, l)) for l in sorted_evals]
    # max(0,l) để tránh lỗi số âm rất nhỏ do sai số

    # 3. Xây dựng V (Các vector riêng sorted_evecs là cột của V)
    V, _ = build_PD(sorted_evals, sorted_evecs)

    m, n = len(A), len(A[0])
    Sigma = build_sigma(singular_values, m, n)

    # 4. Tính U bằng công thức u_i = (1/sigma_i) * A * v_i 
    # Ta tính AV trước, sau đó chia mỗi cột cho sigma tương ứng

    AV = matrix_multiply(A, V)
    U = [[0.0 for _ in range(n)] for _ in range(m)]
    
    for j in range(min (m, n)):
        if singular_values[j] > 1e-10:
            for i in range(m):
                U[i][j] = AV[i][j] / singular_values[j]
        else: # Nếu sigma = 0, cột đó của U có thể để là 0 
            pass

    return U, Sigma, V
def demo_svd(A):
    print("\n" + "="*40)
    print("SVD DEMO: A = U * Σ * Vᵀ")
    print("="*40)

    # 1. Lấy kết quả từ hàm svd của Đức
    U_raw, Sigma_raw, V = svd(A)
    m = len(A)
    n = len(A[0])
    k = min(m, n) # Số lượng giá trị suy biến thực tế

    # 2. Chuẩn bị Sigma dạng VUÔNG (k x k) để nhân cho dễ
    # Vì Sigma_raw của Đức đang là (m x n), ta tỉa lấy phần (k x k)
    Sigma_diag = [[0.0]*k for _ in range(k)]
    singular_vals = []
    for i in range(k):
        val = Sigma_raw[i][i]
        Sigma_diag[i][i] = val
        singular_vals.append(round(val, 4))

    # 3. Chuẩn bị U và VT để khớp kích thước
    # U lấy (m x k), VT lấy (k x n)
    U_fix = [row[:k] for row in U_raw]
    VT = transpose(V)
    if len(VT) > k:
        VT = VT[:k]

    # 4. Tái cấu trúc: A_rec = (U * Sigma) * VT
    # Phép nhân: (m x k) * (k x k) * (k x n) = (m x n)
    try:
        US = matrix_multiply(U_fix, Sigma_diag)
        A_reconstructed = matrix_multiply(US, VT)
    except Exception as e:
        print(f"Lỗi kích thước: {e}")
        return

    # 5. IN KẾT QUẢ
    if m <= 5 and n <= 5:
        print_matrix(A, "1. Ma trận gốc A:")
        print(f"2. Các giá trị suy biến (Singular Values): {singular_vals}")
        print_matrix(U_fix, "3. Ma trận U:")
        print_matrix(Sigma_diag, "4. Ma trận Sigma (Square):")
        print_matrix(VT, "5. Ma trận Vᵀ:")
        print_matrix(A_reconstructed, "6. Kết quả tái tạo UΣVᵀ:")
    
    # 6. Kiểm tra sai số
    err = matrix_error(A, A_reconstructed)
    print("-" * 40)
    print(f"Sai số sau khi tái cấu trúc: {err:.2e}")
    if err < 1e-10:
        print("KẾT QUẢ HOÀN HẢO!")
    else:
        print("Sai số hơi lớn, hãy kiểm tra lại bước Deflation.")