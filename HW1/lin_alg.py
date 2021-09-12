import numpy as np

def check_LDU(P, A, L, D, U):
    lhs = P @ A
    rhs = L @ D @ U
    return np.array_equal(lhs, rhs)

def LDU_decomp(A):
    m = A.shape[0]
    n = A.shape[1]
    P = np.identity(m)
    L = np.identity(m)
    A_prime = np.copy(A)

    for i in range(min(m, n)):
        print("P:")
        print(P)
        print("L:")
        print(L)
        print("A':")
        print(A_prime)
        # If the i, i element is 0, swap with some later row where the j, i (j > i)
        # element is not 0, this exists because we are given A is invertible and therefore
        # full rank
        if A_prime[i, i] == 0:
            j = i + 1
            while j == 0:
                j += 1

            # Swap rows in A_prime, P and the proper portions of rows in L
            A_prime[[i, j]] = A_prime[[j, i]]
            P[[i, j]] = P[[j, i]]

            cols = [x for x in range(i)]
            L[[i, j], cols] = L[[j, i], cols]

        # Subtract scaled row i so that the j, i positions are 0 for all j > i
        # Also store operations in L
        x = A_prime[i, i]
        for j in range(i + 1, m):
            coeff = A_prime[j, i] / x
            A_prime[j] -= coeff * A_prime[i]
            L[j, i] = coeff

    """
    print("P:")
    print(P)
    print("L:")
    print(L)
    print("A':")
    print(A_prime)
    """

    # Decompose A' into D and U
    D = np.identity(m)
    U = np.zeros((m, n))
    for i in range(n):
        D[i, i] = A_prime[i, i]
        U[i] = A_prime[i] / D[i, i]

    return P, L, D, U


def test_LDU(A):

    P, L, D, U = LDU_decomp(A)

    print("A:")
    print(A)
    print("P:")
    print(P)
    print("L:")
    print(L)
    print("D:")
    print(D)
    print("U:")
    print(U)
    print("PA:")
    print(P @ A)
    print("LDU:")
    print(L @ D @ U)

test_LDU(np.array([[1, 1, 0], [1, 1, 2], [4, 2, 3]], dtype=float))
test_LDU(np.array([[7, 6, 1], [4, 5, 1], [7, 7, 7]], dtype=float))
test_LDU(np.array([[12, 12, 0, 0], [3, 0, -2, 0], [0, 1, -1, 0], [0, 0, 0, 1], [0, 0, 1, 1]], dtype=float))
test_LDU(np.array([[7, 6, 4], [0, 3, 3], [7, 3, 1]], dtype=float))
