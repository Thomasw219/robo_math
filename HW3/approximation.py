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

def linear_regression(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y.reshape((-1, 1))

def get_distances(X, plane_params):
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    denom = np.sqrt(plane_params[0]**2 + plane_params[1]**2 + plane_params[2]**2)
    distances = np.abs(X @ plane_params.reshape((-1, 1))) / denom
    return distances

def plot_planes_and_points(name, planes, plane_points, other_points, x_range, y_range, z_range):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for params in planes:
        a, b, c, d = params
        x = np.linspace(x_range[0], x_range[1], 10)
        z = np.linspace(z_range[0], z_range[1], 10)
        X, Z = np.meshgrid(x, z)
        Y = (-1 * a * X - c * Z - d) / b
        ax.plot_surface(X, Y, Z, color='g', alpha=0.3)
    if plane_points is not None:
        ax.scatter(plane_points[:, 0], plane_points[:, 1], plane_points[:, 2], color='g', s=2)

    if other_points is not None:
        ax.scatter(other_points[:, 0], other_points[:, 1], other_points[:, 2], color='r', s=2)
    plt.savefig(name)

def get_ranges(X):
    x_range = (np.min(X[:, 0]), np.max(X[:, 0]))
    y_range = (np.min(X[:, 1]), np.max(X[:, 1]))
    z_range = (np.min(X[:, 2]), np.max(X[:, 2]))
    return x_range, y_range, z_range

def partition_points(X, distances, threshold):
    in_points = []
    out_points = []
    for i, x in enumerate(X):
        if distances[i] < threshold:
            in_points.append(x)
        else:
            out_points.append(x)
    return np.array(in_points), np.array(out_points)

THRESH = 0.01
def regress_process(X, points):
    beta = linear_regression(points, -1 * np.ones((points.shape[0], 1)))
    plane_params = np.concatenate([beta.flatten(), [1]])
    distances = get_distances(points, plane_params)
    plane_points, other_points = partition_points(points, distances, 0.01)
    return plane_params, plane_points, other_points, distances

def RANSAC(points, num_planes, num_samples, sample_size):
    for i in range(num_planes):
        for j in range(num_samples):
            points_subset = np.random.choice(points, size=sample_size)

def q4():
    clear_table_points = read_point_cloud_file("data/clear_table.txt")
    plane_params, plane_points, other_points, distances = regress_process(clear_table_points, clear_table_points)
    ranges = get_ranges(clear_table_points)
    plot_planes_and_points("figures/clear_table.png", [plane_params], plane_points, other_points, *ranges)
    print("Clear table fit average distance: {}".format(np.mean(distances)))
    print("Points within threshold of {} from plane(s): {}/{}".format(THRESH, plane_points.shape[0], clear_table_points.shape[0]))

    cluttered_table_points = read_point_cloud_file("data/cluttered_table.txt")
    plane_params, plane_points, other_points, distances = regress_process(cluttered_table_points, cluttered_table_points)
    ranges = get_ranges(cluttered_table_points)
    plot_planes_and_points("figures/cluttered_table.png", [plane_params], plane_points, other_points, *ranges)
    print("Cluttered table fit average distance: {}".format(np.mean(distances)))
    print("Points within threshold of {} from plane(s): {}/{}".format(THRESH, plane_points.shape[0], clear_table_points.shape[0]))

#q1()
#q2()
q4()
