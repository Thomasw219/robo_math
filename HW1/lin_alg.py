import numpy as np

def check_LDU(P, A, L, D, U):
    lhs = P @ A
    rhs = L @ D @ U
    return np.array_equal(lhs, rhs)

def LDU_decomp(A):
    n = A.shape[0]
    P = np.identity(n)
    L = np.identity(n)
    A_prime = np.copy(A)

    for i in range(n):
        """
        print("P:")
        print(P)
        print("L:")
        print(L)
        print("A':")
        print(A_prime)
        """
        # If the i, i element is 0, swap with some later row where the m, i (m > i)
        # element is not 0, this exists because we are given A is invertible and therefore
        # full rank
        if A_prime[i, i] == 0:
            m = i + 1
            while m == 0:
                m += 1

            # Swap rows in A_prime, P and the proper portions of rows in L
            A_prime[[i, m]] = A_prime[[m, i]]
            P[[i, m]] = P[[m, i]]

            cols = [x for x in range(i)]
            L[[i, m], cols] = L[[m, i], cols]

        # Subtract scaled row i so that the j, i positions are 0 for all j > i
        # Also store operations in L
        x = A_prime[i, i]
        for j in range(i + 1, n):
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
    D = np.identity(n)
    U = np.zeros((n, n))
    for i in range(n):
        D[i, i] = A_prime[i, i]
        U[i] = A_prime[i] / D[i, i]

    return P, L, D, U


A = np.array([[1, 1, 0], [1, 1, 2], [4, 2, 3]], dtype=float)
P, L, D, U = LDU_decomp(A)

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
