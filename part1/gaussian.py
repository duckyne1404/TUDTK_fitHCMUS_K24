EPS = 1e-14

def back_substitution(U, c):
    n = len(U)
    #Tao list nghiem pt
    x = [0.0] * n                                                
    
    #Duet quay lui tu cuoi dong
    for i in range(n - 1, -1, -1):                                  
        if abs(U[i][i]) < EPS:
            raise ValueError("He tam giac tren suy bien, khong co nghiem duy nhat.")
        #Giai nghiem
        sum_ax = 0.0
        for j in range(i + 1, n):
            sum_ax += U[i][j] * x[j]

        x[i] = (c[i] - sum_ax) / U[i][i]
        
    return x

def gaussian_eliminate(A, b):
    n = len(A)
    Ab = [A[i] + [b[i]] for i in range(n)]
    swaps = 0

    for i in range(n):
        #Tuong tu phan chon pivot o det
        max_row = i
        for k in range(i + 1, n):
            if abs(Ab[k][i]) > abs(Ab[max_row][i]):
                max_row = k
        
        if max_row != i:
            Ab[i], Ab[max_row] = Ab[max_row], Ab[i]
            swaps += 1
            
        pivot = Ab[i][i]

        #Bo qua viec khu cac phan tu bang 0
        if abs(pivot) < EPS:
            raise ValueError("He tam giac tren suy bien, khong co nghiem duy nhat.")

        for k in range(i + 1, n):
            factor = Ab[k][i] / pivot
            for j in range(i, n + 1):
                Ab[k][j] -= factor * Ab[i][j]

    U = [row[:n] for row in Ab]  #Cat tu idx 0 den n - 1 trong dong
    c = [row[n] for row in Ab]   #Cat dung phan tu thu n
    
    x = back_substitution(U, c)
    
    return U, x, swaps