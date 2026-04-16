EPS = 1e-14

def inverse(A):
    n = len(A)

    #Ghep A voi ma tran don vi I
    aug = []
    for i in range(n):
        row = A[i][:] + [0.0]*n
        row[n + i] = 1.0
        aug.append(row)

    for i in range(n):
        max_row = i
        for k in range(i + 1, n):
            if abs(aug[k][i]) > abs(aug[max_row][i]):
                max_row = k

        #Ma tran co dinh thuc bang 0 => khong kha nghich
        if abs(aug[max_row][i]) < EPS:
            return None

        aug[i], aug[max_row] = aug[max_row], aug[i]

        pivot = aug[i][i]

        #Gaus Jordan buoc cac phan tu duong cheo phai bang 1
        for j in range(2*n):
            aug[i][j] /= pivot

        #Khu tao ma tran don vi
        for k in range(n):
            if k == i:
                continue
            factor = aug[k][i]
            for j in range(2*n):
                aug[k][j] -= factor * aug[i][j]

    #Lay phan ma tran khac ma tran don vi I
    return [row[n:] for row in aug]