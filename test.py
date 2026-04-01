import time
import random
from matrix_utils import generate_symmetric_matrix
from main_function import (
    eigen_decomposition, build_PD, demo_diagonalization,
    svd, demo_svd
)

def generate_random_matrix(m, n):
    return [[random.uniform(-10, 10) for _ in range(n)] for _ in range(m)]

def run_multi_test():
    # --- PHẦN 1: TEST CHÉO HÓA ---
    print(f"\n[PHẦN 1] CHÉO HÓA MA TRẬN VUÔNG ĐỐI XỨNG")
    A_sym = generate_symmetric_matrix(3)
    vals, vecs = eigen_decomposition(A_sym)
    P, D = build_PD(vals, vecs)
    demo_diagonalization(A_sym, P, D)

    # --- PHẦN 2: TEST SVD ---
    print(f"\n[PHẦN 2] PHÂN RÃ GIÁ TRỊ SUY BIẾN (SVD)")
    A_rect = generate_random_matrix(4, 3)
    demo_svd(A_rect)
# --- PHẦN 3: BENCHMARK ---
    print(f"\n[PHẦN 3] KIỂM TRA HIỆU NĂNG (n=40)")
    n_size = 40
    
    # Tạo ma trận cho SVD (Ma trận bất kỳ)
    A_rect = generate_random_matrix(n_size, n_size)
    
    # Tạo ma trận cho Chéo hóa (Nên dùng ma trận đối xứng để hội tụ tốt)
    A_sym = generate_symmetric_matrix(n_size) 

    # 1. Đo thời gian SVD
    start_svd = time.time()
    svd(A_rect)
    time_svd = time.time() - start_svd
    print(f"⏱️  Thời gian xử lý SVD ({n_size}x{n_size}): {time_svd:.4f} giây")

    # 2. Đo thời gian Chéo hóa (Eigen-decomposition)
    start_eigen = time.time()
    eigen_decomposition(A_sym)
    time_eigen = time.time() - start_eigen
    print(f"⏱️  Thời gian xử lý Chéo hóa ({n_size}x{n_size}): {time_eigen:.4f} giây")

    # So sánh nhanh
    print(f"\n=> Chéo hóa chạy nhanh gấp {time_svd/time_eigen:.2f} lần so với SVD" if time_eigen < time_svd 
          else f"\n=> SVD chạy nhanh gấp {time_eigen/time_svd:.2f} lần so với Chéo hóa")