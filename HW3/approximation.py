import numpy as np
import matplotlib.pyplot as plt

def q1():
    x = np.linspace(0, np.pi, num=200)
    y = np.cos(x)
    plt.plot(x, y)
    plt.savefig("figures/q1a.png")

def q2():
    filename = "data/problem2.txt"
    with open(filename, 'r') as f:
        samples = np.array([float(token) for token in f.readline().split()])
    x = np.array([i / 100 for i in range(101)])
    f_of_x = samples
    p_of_x = np.log(f_of_x)
    max_degree = 10
    basis_function_matrix = np.zeros((x.shape[0], max_degree + 1))
    for i in range(max_degree + 1):
        basis_function_matrix[:, i] = np.power(x, i)
    print(basis_function_matrix)

    X = basis_function_matrix
    y = p_of_x.reshape((-1, 1))
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    print(beta)
    integer_beta = np.round(beta)
    errors = np.exp(X @ integer_beta).flatten() - f_of_x
    print(errors)

def read_point_cloud_file(f_name):
    with open(f_name, 'r') as f:
        lines = f.readlines()
        points = []
        for l in lines:
            points.append([float(token) for token in l.split()])
        return np.array(points)

def q4():
    clear_table_points = read_point_cloud_file("data/clear_table.txt")
    print(clear_table_points.shape)

#q1()
#q2()
q4()
