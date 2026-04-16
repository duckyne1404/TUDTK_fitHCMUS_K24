from determinant import determinant
from gaussian import gaussian_eliminate
from inverse import inverse
from rank_and_basic import rank_and_basis
from verify import verify_solution, verify_inverse, verify_determinant, verify_rank

def print_matrix(M):
    for row in M:
        print([round(x,6) for x in row])

def run_test(name, A, b):
    print("=====", name, "=====")

    det_val = determinant(A)
    print("det:", det_val, "->", verify_determinant(A, det_val))

    U, x, _ = gaussian_eliminate(A, b)
    print("U:")
    print_matrix(U)

    print("solution:", verify_solution(A, x, b))

    inv = inverse(A)
    if inv is None:
        print("inverse: SKIP")
    else:
        print("inverse:", verify_inverse(A, inv))

    rank, row_basis, col_basis, null_basis = rank_and_basis(A)

    print("rank:", rank, "->", verify_rank(A, rank))

    print("row_basis:")
    for v in row_basis:
        print([round(x,6) for x in v])

    print("col_basis:")
    for v in col_basis:
        print([round(x,6) for x in v])

    print("null_basis:")
    for v in null_basis:
        print([round(x,6) for x in v])

    print()


def main():

    A1 = [
        [1/(i+j+1) for j in range(5)]
        for i in range(5)
    ]
    b1 = [1,1,1,1,1]

    A2 = [
        [1,1,1],
        [1,1+1e-6,1],
        [1,1,1+1e-6]
    ]
    b2 = [3,3,3]

    A3 = [
        [1e10,2e10],
        [3e10,4e10]
    ]
    b3 = [3e10,7e10]

    A4 = [
        [1e-10,2e-10],
        [3e-10,4e-10]
    ]
    b4 = [3e-10,7e-10]

    import random
    random.seed(0)
    A5 = [[random.random() for _ in range(6)] for _ in range(6)]
    b5 = [random.random() for _ in range(6)]

    A6 = [
        [1,2,3,4],
        [2,4,6,8],
        [3,5,7,9],
        [4,8,12,16]
    ]
    b6 = [10,20,30,40]

    A7 = [
        [1e-10,1,1,1],
        [1,1e-10,1,1],
        [1,1,1e-10,1],
        [1,1,1,1e-10]
    ]
    b7 = [3,3,3,3]

    A8 = [
        [1,1,1,1,1],
        [1,1,1,1,1.000001],
        [1,1,1,1.000001,1],
        [1,1,1.000001,1,1],
        [1,1.000001,1,1,1]
    ]
    b8 = [5,5.000001,5.000001,5.000001,5.000001]

    run_test("Hilbert 5x5", A1, b1)
    run_test("Near Singular", A2, b2)
    run_test("Large Values", A3, b3)
    run_test("Small Values", A4, b4)
    run_test("Random 6x6", A5, b5)
    run_test("Hidden dependent", A6, b6)
    run_test("Mixed Hard", A7, b7)
    run_test("Ultra Hard", A8, b8)


if __name__ == "__main__":
    main()