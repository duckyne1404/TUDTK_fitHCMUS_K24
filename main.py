from test import run_multi_test

def main():
    print("="*60)
    print("CHƯƠNG TRÌNH DEMO CHÉO HÓA MA TRẬN VÀ SVD - HCMUS")
    print("="*60)
    
    try:
        # Gọi toàn bộ các kịch bản test (Diagonalization, SVD, Benchmark)
        run_multi_test()
    except Exception as e:
        print(f"\nLỗi hệ thống: {e}")
    
    print("\n" + "="*60)
    print("Đã hoàn thành tất cả nội dung kiểm tra.")
    print("="*60)

if __name__ == "__main__":
    main()