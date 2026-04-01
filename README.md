# ĐỒ ÁN 1: MA TRẬN VÀ CƠ SỞ CỦA TÍNH TOÁN KHOA HỌC
[cite_start]**Môn học:** Toán Ứng Dụng và Thống Kê (MTH00051)
[cite_start]**Học kỳ:** II - Năm học 2025-2026

---

## 👤 Thông tin sinh viên
* **Họ và tên:** Phí Hoàng Đức
* **MSSV:** 24120248
* [cite_start]**Đơn vị:** Khoa Công nghệ Thông tin - HCMUS

---

## 📂 Cấu trúc mã nguồn
[cite_start]Dựa trên yêu cầu của đồ án, các tệp tin được tổ chức như sau[cite: 334, 343]:

1.  **`main.py`**: Điểm khởi đầu của chương trình, hiển thị thông tin sinh viên và điều phối các kịch bản kiểm thử.
2.  **`main_function.py`**: Chứa các thuật toán lõi được cài đặt từ đầu (from scratch):
    * **Power Iteration**: Tìm trị riêng và vector riêng.
    * **Eigen-decomposition**: Chéo hóa ma trận vuông.
    * **SVD (Singular Value Decomposition)**: Phân rã giá trị suy biến.
    * **Matrix Inverse**: Nghịch đảo ma trận bằng Gauss-Jordan với **Partial Pivoting**.
3.  **`matrix_utils.py`**: Các hàm bổ trợ xử lý ma trận và vector (nhân, chuyển vị, chuẩn hóa,...).
4.  **`test.py`**: Kịch bản kiểm thử tự động, đánh giá sai số tái cấu trúc và đo lường hiệu năng (Benchmark).

---

## 🚀 Chức năng chính (Phần 2: Phân rã ma trận)
[cite_start]Đồ án này tập trung cài đặt và thực nghiệm các phương pháp phân rã ma trận mà không sử dụng thư viện hỗ trợ tính toán sẵn (`numpy.linalg`)[cite: 99, 181]:

* [cite_start]**Chéo hóa ma trận ($A = PDP^{-1}$)**: Áp dụng cho ma trận vuông đối xứng, tìm ma trận đường chéo $D$ và ma trận vector riêng $P$[cite: 183, 184].
* [cite_start]**Phân rã SVD ($A = U\Sigma V^T$)**: Sử dụng ma trận hiệp phương sai $A^T A$ để tìm các giá trị suy biến và các vector trực giao[cite: 218, 222].
* [cite_start]**Tính toán ổn định**: Sử dụng kỹ thuật **Partial Pivoting** để giảm thiểu sai số làm tròn trong quá trình khử Gauss-Jordan[cite: 129, 131].
* [cite_start]**Hiệu năng**: Tích hợp bộ đo thời gian xử lý cho ma trận kích thước lớn (Benchmark) để phân tích chi phí tính toán $O(n^3)$[cite: 259, 288].

---

## 🛠️ Hướng dẫn khởi chạy

### 1. Yêu cầu hệ thống
* [cite_start]Python 3.10 hoặc cao hơn[cite: 92].
* [cite_start]Các thư viện hỗ trợ (chỉ dùng để kiểm chứng kết quả hoặc vẽ đồ thị): `numpy`, `matplotlib`[cite: 93, 96].

### 2. Cài đặt môi trường
Tạo file `requirements.txt` và cài đặt:
```bash
pip install numpy matplotlib
