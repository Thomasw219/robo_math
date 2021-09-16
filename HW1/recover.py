import numpy as np
from scipy.spatial.transform import Rotation

def generate_points(n, sd=1):
    return np.random.normal(scale=sd, size=(3, n))

def translate_and_rotate(points):
    n = points.shape[1]
    centroid = np.sum(points, axis=1).reshape((3, 1)) / n

    origin_centroid_points = points - centroid

    rotvec = np.random.normal(size=3)
    rotvec /= np.linalg.norm(rotvec)

    r = Rotation.from_rotvec(np.random.uniform(low=0, high=np.pi) * rotvec)
    R = r.as_matrix()
    rotated_points = R @ origin_centroid_points

    translation = np.random.normal(size=(3,1))
    rotated_and_translated_points = rotated_points + translation

    return rotated_and_translated_points, R, translation - centroid

def recover(P, Q):
    n = P.shape[1]
    P_centroid = np.sum(P, axis=1).reshape((3, 1)) / n
    Q_centroid = np.sum(Q, axis=1).reshape((3, 1)) / n

    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid

    U, s, V_T = np.linalg.svd(P_centered)
    s = np.array([sig for sig in s if np.abs(sig) > 0.00001])
    S_inv = np.transpose(np.zeros_like(P_centered))
    for i, sig in enumerate(s):
        S_inv[i, i] = 1 / sig

    R = Q_centered @ np.transpose(V_T) @ S_inv @ np.transpose(U)

    return R, Q_centroid - P_centroid, P_centroid, Q_centroid

def test_recover(n):
    P = generate_points(n)
    print("P:")
    print(P)
    Q, R, t = translate_and_rotate(P)
    print("Q:")
    print(Q)
    print("Original Rotation:")
    print(R)
    print("Original Translation:")
    print(t)

    R_prime, t_prime, P_centroid, Q_centroid = recover(P, Q)
    print("Recovered Rotation:")
    print(R_prime)
    print("Recovered Translation:")
    print(t_prime)

    P_centered = P - P_centroid
    Q_centered = R_prime @ P_centered
    Q_prime = Q_centered + Q_centroid
    print("Reconstructed Q:")
    print(Q_prime)

test_recover(3)
test_recover(4)
test_recover(5)
test_recover(6)
