EPS = 1e-12
'''
Vi tỏng file co yeu cau rank co the lay tu ma tran REF
Nhung row space phai lay tu ma tran RREF
'''
#Chuyen tat ca phan tu xap xi 0 thanh 0
#tranh ket qua ra so rat le khi in ra
def clean_row(row):
    cleaned = []
    for x in row:
        if abs(x) < EPS:
            cleaned.append(0.0)
        else:
            cleaned.append(x)
    return cleaned


#Kiem tra 1 dong co phai dong 0 hay khong
def is_zero_row(row):
    return all(abs(x) < EPS for x in row)


#Dua ma tran ve RREF va lay cac cot pivot
def rref_with_pivots(A):
    n = len(A)
    m = len(A[0])
    matrix = [row[:] for row in A]

    row = 0
    pivot_cols = []

    for col in range(m):
        #Tim dong co pivot lon nhat trong cot hien tai
        pivot = row
        max_val = 0.0
        for r in range(row, n):
            if abs(matrix[r][col]) > max_val:
                max_val = abs(matrix[r][col])
                pivot = r

        #Neu ca cot deu xap xi 0 thi bo qua cot nay
        if max_val < EPS:
            continue

        #Dua dong chua pivot len tren
        matrix[row], matrix[pivot] = matrix[pivot], matrix[row]

        #Chuan hoa pivot ve 1
        pivot_val = matrix[row][col]
        for j in range(col, m):
            matrix[row][j] /= pivot_val

        #Khu tat ca phan tu tren va duoi pivot
        #de dua ma tran ve RREF thay vi REF
        for r in range(n):
            if r == row:
                continue

            factor = matrix[r][col]
            for j in range(col, m):
                matrix[r][j] -= factor * matrix[row][j]

        matrix[row] = clean_row(matrix[row])
        pivot_cols.append(col)
        row += 1

        if row == n:
            break

    #Lam sach lai ca ma tran
    for i in range(n):
        matrix[i] = clean_row(matrix[i])

    return matrix, pivot_cols


def rank_and_basis(A):
    n = len(A)
    m = len(A[0])

    #Lay RREF va cac cot pivot
    R, pivot_cols = rref_with_pivots(A)

    #Rank bang so cot pivot
    rank = len(pivot_cols)

    #Lay khong gian cot tu cac cot pivot cua ma tran goc
    cols_space = []
    for col in pivot_cols:
        column = []
        for i in range(n):
            column.append(A[i][col])
        cols_space.append(column)

    #Lay khong gian dong tu cac dong khac 0 cua RREF
    rows_space = []
    for r in R:
        if not is_zero_row(r):
            rows_space.append(clean_row(r))

    #Tim cac bien tu do
    free_vars = []
    for j in range(m):
        found = False
        for pc in pivot_cols:
            if j == pc:
                found = True
                break
        if not found:
            free_vars.append(j)

    #Tim co so cua null space
    null_space = []
    for free in free_vars:
        #Cho 1 bien tu do bang 1, con lai bang 0
        x = [0.0] * m
        x[free] = 1.0

        #Voi RREF:
        #x[pivot] + tong(R[i][j] * x[j]) = 0
        #=> x[pivot] = -tong(...)
        for i in range(rank - 1, -1, -1):
            pivot_col = pivot_cols[i]

            sum_ax = 0.0
            for j in range(pivot_col + 1, m):
                sum_ax += R[i][j] * x[j]

            x[pivot_col] = -sum_ax

        null_space.append(clean_row(x))

    return rank, rows_space, cols_space, null_space