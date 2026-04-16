EPS = 1e-18

def determinant(A):
    n = len(A)
    matrix = [row[:] for row in A]
    
    det = 1.0
    swaps = 0

    for i in range(n):
        #TIm phan tu lon nhat trong cot
        max_row = i
        for k in range(i + 1, n):
            if abs(matrix[k][i]) > abs(matrix[max_row][i]):
                max_row = k
        
        #Neu phan tu lon nhat cua cot ma van xap xi EPS hay xap xi 0 
        # => Ma tran khong co dinh thuc
        if abs(matrix[max_row][i]) < EPS:
            return 0.0

        #Swap dong co phan tu chon lem pivot len tren
        if max_row != i:
            matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
            swaps += 1
            
        pivot = matrix[i][i]

        for k in range(i + 1, n):
            factor = matrix[k][i] / pivot
            for j in range(i, n):
                matrix[k][j] -= factor * matrix[i][j]

    for i in range(n):
        det *= matrix[i][i]

    #swap lam doi dau dinh thuc
    if swaps % 2 != 0:
        det = -det
        
    return det