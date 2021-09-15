import numpy as np
from scipy.linalg import lu

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
        """
        print("P:")
        print(P)
        print("L:")
        print(L)
        print("A':")
        print(A_prime)
        """
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

def print_LDU(A, P, L, D, U):
    with np.printoptions(precision=5, suppress=True, threshold=5):
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

def print_SVD(A, U, S, V_T):
    with np.printoptions(precision=5, suppress=True, threshold=5):
        print("A:")
        print(A)
        print("U:")
        print(U)
        print("S:")
        print(S)
        print("V^T:")
        print(V_T)
        print("USV^T:")
        print(U @ S @ V_T)

def test_LDU(A):
    P, L, D, U = LDU_decomp(A)
    print_LDU(A, P, L, D, U)

def test_sp_LDU(A):
    k = min(A.shape[0], A.shape[1])
    P, L, U = lu(A)
    D = np.identity(k)
    for i in range(k):
        D[i, i] = U[i, i]
        if U[i, i] == 0:
            break
        else:
            U[i] = U[i] / U[i, i]

    print_LDU(A, P, L, D, U)

def test_np_SVD(A):
    U, s, V_T = np.linalg.svd(A)
    S = np.zeros_like(A)
    for i, sig in enumerate(s):
        S[i, i] = sig
    print_SVD(A, U, S, V_T)

def SVD_solution(A, b):
    U, s, V_T = np.linalg.svd(A)
    s = np.array([sig for sig in s if np.abs(sig) > 0.00001])
    S_inv = np.transpose(np.zeros_like(A))
    for i, sig in enumerate(s):
        S_inv[i, i] = 1 / sig

    return np.transpose(V_T) @ S_inv @ np.transpose(U) @ b


"""
#test_LDU(np.array([[1, 1, 0], [1, 1, 2], [4, 2, 3]], dtype=float))
A_1 = np.array([[7, 6, 1], [4, 5, 1], [7, 7, 7]], dtype=float)
A_2 = np.array([[12, 12, 0, 0], [3, 0, -2, 0], [0, 1, -1, 0],
        [0, 0, 0, 1], [0, 0, 1, 1]], dtype=float)
A_3 = np.array([[7, 6, 4], [0, 3, 3], [7, 3, 1]], dtype=float)

#test_LDU(A_1)
test_np_SVD(A_1)
#test_sp_LDU(A_2)
test_np_SVD(A_2)
#test_sp_LDU(A_3)
test_np_SVD(A_3)
"""

A = np.array([[7, 6, 4], [0, 3, 3], [7, 3, 1]], dtype=float)
b_1 = np.array([[5], [-3], [8]], dtype=float)
b_2 = np.array([[2], [0], [11]], dtype=float)

x_bar_1 = SVD_solution(A, b_1)
x_bar_2 = SVD_solution(A, b_2)

v = np.array([[-0.1980], [0.6931], [-0.6931]], dtype=float)

print(x_bar_1)
print(A @ x_bar_1)
print(x_bar_2)
print(A @ x_bar_2)

print(A @ (x_bar_1 + v))
