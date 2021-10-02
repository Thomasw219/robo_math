import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from functools import partial

# 1a functions
def compute_table(x, y, dtype='float'):
    n = len(x)
    table = np.zeros((n, n), dtype=dtype)
    for i, y_i in enumerate(y):
        table[i, 0] = y_i

    for i in range(1, n):
        for j in range(i, n):
            table[j, i] = (table[j - 1][i - 1] - table[j][i - 1]) / (x[j - i] - x[j])

    return table

def compute_coefficients(x, y):
    table = compute_table(x, y)
    n = len(x)
    coefficients = np.array([table[i, i] for i in range(n)])
    return coefficients

def divided_differences_interpolation(x, y, x_eval):
    coefficients = compute_coefficients(x, y)
    y_eval = np.zeros_like(x_eval)
    for i in range(len(x)):
        term = np.ones_like(y_eval) * coefficients[i]
        for j in range(i):
            term *= x_eval - x[j]
        y_eval += term
    return y_eval

def q1():
    # 1b computation and graph
    print("1b")
    x = np.array([0, 1/8, 1/4, 1/2, 3/4, 1])
    y = np.exp(-1 / 2 * np.square(x))

    x_eval = np.array([1 / 3])
    y_eval = divided_differences_interpolation(x, y, x_eval)

    print("x: {}\ny: {}".format(x_eval, y_eval))

    plt.scatter(x, y, c='b', label='Given')
    plt.scatter(x_eval, y_eval, c='g', label='Interpolated Value(s)')
    plt.legend()
    plt.savefig("./1b.png")

    # 1c computation and graphs
    print("1c")
    x_eval = np.array([0.06])
    for n in [2, 4, 40]:
        x = np.array([i * 2 / n - 1 for i in range(n + 1)])
        y = np.array(np.divide(1, np.add(1, 36 * np.square(x))))
        y_eval = divided_differences_interpolation(x, y, x_eval)

        print("x: {}\ny: {}".format(x_eval, y_eval))

        plt.clf()
        plt.scatter(x, y, c='b', label='Given')
        plt.scatter(x_eval, y_eval, c='g', label='Interpolated Value(s)')
        plt.legend()
        plt.savefig("./1c_n={}.png".format(n))

    print("Actual y value: {}".format(1 / (1 + 36 * 0.06**2)))

    # 1d computation and graphs
    print("1d")
    x_eval = np.linspace(-1, 1, num=1000)

    max_errors = []
    n_values = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 40]
    for n in n_values:
        x = np.array([i * 2 / n - 1 for i in range(n + 1)])
        y = np.array(np.divide(1, np.add(1, 36 * np.square(x))))

        y_real = np.array(np.divide(1, np.add(1, 36 * np.square(x_eval))))
        y_eval = divided_differences_interpolation(x, y, x_eval)
        pointwise_errors = np.abs(y_real - y_eval)
        max_error = np.max(pointwise_errors)
        max_errors.append(max_error)

        plt.clf()
        plt.ylim(-1, 1)
        plt.xlim(-1, 1)
        plt.plot(x_eval, [0 for i in range(len(x_eval))], c='k')
        plt.scatter(x, y, c='b', label='Given')
        plt.plot(x_eval, y_eval, c='g', label='Interpolated Value(s)')
        plt.plot(x_eval, y_real, c='b', label='Real Values')
        plt.plot(x_eval, pointwise_errors, c='r', label='Error')
        plt.legend()
        plt.savefig("./1d_n={}.png".format(n))

    print("n values: {}\nErrors: {}".format(n_values, max_errors))
    plt.clf()
    plt.scatter(n_values, np.log(max_errors))
    plt.savefig("./1d_errors_by_n.png")

# 3 functions and computations
def newtons_method(x_0, function, derivative, conv_error=1e-10, conv_iter=5, max_iter=1000):
    past_values = deque()
    x_n = x_0
    iteration = 0
    while (len(past_values) < conv_iter or np.max(np.abs(np.array(past_values) - x_n)) > conv_error) and iteration < max_iter:
        x_np1 = x_n - function(x_n) / derivative(x_n)
        past_values.appendleft(x_np1)
        if len(past_values) > conv_iter:
            past_values.pop()
        x_n = x_np1
        iteration += 1

    return past_values[0], iteration

def problem_function(x):
    return np.tan(x) - x

def problem_function_derivative(x):
    return 1 / np.cos(x)**2 - 1

def find_initialization_point(low, high, function, num=100):
    x = np.random.uniform(low, high, size=num)
    y = function(x)
    index = np.nanargmin(np.abs(y))
    x_0 = x[index]
    return x_0

def q3():
    print("[19 - pi, 19]")
    x_0 = find_initialization_point(19 - np.pi, 19, problem_function)
    print("x_0: {}".format(x_0))
    x_root, iteration = newtons_method(x_0, problem_function, problem_function_derivative)
    print("x_root: {}\nNewton's method iterations: {}".format(x_root, iteration))

    print("\n[19, 19 + pi]")
    x_0 = find_initialization_point(19, 19 + np.pi, problem_function)
    print("x_0: {}".format(x_0))
    x_root, iteration = newtons_method(x_0, problem_function, problem_function_derivative)
    print("x_root: {}\nNewton's method iterations: {}".format(x_root, iteration))

# 5a functions
def quadratic_interpolation(x, function, conv_error):
    y = function(x)
    print("x: {}, y (norm): {}".format(x[2], np.abs(y[2])))
    if np.abs(y[2]) < conv_error:
        return 0

    differences_table = compute_table(x, y, dtype='complex')
    a = differences_table[2, 2]
    b = differences_table[2, 1] + a * (x[2] - x[1])
    c = y[2]

    rooted_determinant = np.sqrt((b**2 - 4 * a * c).astype('complex'))
    denom1 = b + rooted_determinant
    denom2 = b - rooted_determinant
    if np.abs(denom1) > np.abs(denom2):
        root = 2 * c / denom1
    else:
        root = 2 * c / denom2
    return root

def deflated_function_template(roots, polynomial, x):
    output = polynomial(x).astype('complex')
    for r in roots:
        output /= (x - r).astype('complex')
    return output

def mullers_method(x_m2, x_m1, x_0, num_roots, function, conv_error=1e-10, conv_iter=5, max_iter=1000):
    roots = []
    iterations = []
    deflated_function = function
    for i in range(num_roots):
        print("Root: {}".format(i + 1))
        past_values = deque()
        iteration = 0
        interpolate_points = deque([x_m2, x_m1, x_0])
        while (len(past_values) < conv_iter or np.max(np.abs(np.array(past_values) - interpolate_points[-1])) > conv_error) and iteration < max_iter:
            x = np.array(interpolate_points)
            x_np1 = x[2] - quadratic_interpolation(x, deflated_function, conv_error)
            past_values.appendleft(x_np1)
            if len(past_values) > conv_iter:
                past_values.pop()
            interpolate_points.popleft()
            interpolate_points.append(x_np1)
            iteration += 1

        root = past_values[0]
        roots.append(root)
        deflated_function = partial(deflated_function_template, roots, function)
        iterations.append(iteration)

    return roots, iterations

def polynomial_function(x):
    return np.power(x, 4) + x + 1

def q5():
    roots, iterations = mullers_method(1, 2, 3, 4, polynomial_function)
    print("Roots: {}\nIterations: {}".format(roots, iterations))

# 6 computations
def q6():
    Q = np.array([[1, -2, 1, -2, 0], [0, 1, -2, 1, -2], [1, 1, -6, 0, 0], [0, 1, 1, -6, 0], [0, 0, 1, 1, -6]])
    print("Det A: {}".format(np.linalg.det(Q)))
    A = np.delete(Q, 0, axis=0)
    A_1 = np.delete(A, 0, axis=1)
    A_2 = np.delete(A, 1, axis=1)
    x = (-1)**3 * np.linalg.det(A_1) / np.linalg.det(A_2)
    print("x: {}".format(x))

# 7 computations
def resultant_determinant(x):
    return 16 * np.power(x, 4) + 144 * np.power(x, 3) + 480 * np.power(x, 2) + 692 * x + 359

def q7():
    roots, iterations = mullers_method(1, 2, 3, 4, resultant_determinant)
    print("Roots: {}\nIterations: {}".format(roots, iterations))

# 8 functions and computations
def goes_above(x):
    if np.mean(x[1]) > np.mean(x[0]):
        return True
    else:
        return False

def in_triangle(x, x_0, x_1, x_2):
    points = [x_0, x_1, x_2]
    for i in range(3):
        side_vec = points[(i + 1) % 3] - points[i]
        point_vec = x - points[i]
        direction_vec = points[(i + 2) % 3] - points[i]
        orthogonal_point_vec = point_vec - (np.dot(point_vec, side_vec) / np.dot(side_vec, side_vec)) * side_vec
        orthogonal_direction_vec = direction_vec - (np.dot(direction_vec, side_vec) / np.dot(side_vec, side_vec)) * side_vec
        if np.linalg.norm(orthogonal_direction_vec) < 1e-5:
            return False
        index = np.argmax(np.abs(orthogonal_direction_vec))
        scalar = orthogonal_point_vec[index] / orthogonal_direction_vec[index]
        if scalar <= 0:
            return False
    return True

FILE = "./paths.txt"
def parse_file():
    with open(FILE, 'r') as f:
        lines = f.readlines()
        paths = np.zeros((len(lines) // 2, 50, 2))
        for i in range(len(lines) // 2):
            x_values = [float(s) for s in lines[2 * i].split()]
            y_values = [float(s) for s in lines[2 * i + 1].split()]
            for j in range(50):
                paths[i, j, 0] = x_values[j]
                paths[i, j, 1] = y_values[j]

    return paths

def sort_paths(paths):
    above_paths = []
    below_paths = []
    for path in paths:
        mean = np.mean(path, axis=0)
        if goes_above(mean):
            above_paths.append(path)
        else:
            below_paths.append(path)
    return np.stack(above_paths), np.stack(below_paths)

def angle_between_vectors(u, v):
    return np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))

def find_reference_paths(p, above_paths, below_paths):
    if goes_above(p):
        paths = above_paths
    else:
        paths = below_paths
    n = len(paths)
    for i in range(n):
        x_1 = paths[i][0]
        for j in range(n):
            x_2 = paths[j][0]
            theta = angle_between_vectors(x_1 - p, x_2 - p)
            if j == i or theta >= 5 * np.pi / 6 or theta <= np.pi / 2:
                continue
            for k in range(n):
                x_3 = paths[k][0]
                if k == i or k == j or not in_triangle(p, x_1, x_2, x_3):
                    continue
                return np.stack([paths[i], paths[j], paths[k]])

def find_coefficients(p, ref_paths):
    x_1 = ref_paths[0, 0]
    x_2 = ref_paths[1, 0]
    x_3 = ref_paths[2, 0]
    A = np.concatenate([np.stack([x_1, x_2, x_3]).transpose(), np.array([1, 1, 1]).reshape((1, 3))])
    b = np.concatenate([p, np.array([1])]).reshape((3, 1))
    return (np.linalg.inv(A) @ b).ravel()

def synthesize_path(ref_paths, coefficients):
    return ref_paths[0] * coefficients[0] + ref_paths[1] * coefficients[1] + ref_paths[2] * coefficients[2]

def draw_ring():
    circle = plt.Circle((5, 5), 1.5, color='r')
    fig, ax = plt.subplots()
    ax.add_patch(circle)
    plt.ylim(0, 12)
    plt.xlim(0, 12)

def interpolate_path(path, t, interval=1):
    lower_idx = np.floor(t / interval).astype(int)
    upper_idx = min(lower_idx + 1, len(path) - 1)
    alpha = t % interval
    return path[upper_idx] * alpha + path[lower_idx] * (1 - alpha)

def q8():
    points = [np.array([0.8, 1.8]), np.array([2.2, 1.0]), np.array([2.7, 1.4])]
    paths = parse_file()
    plt.clf()
    draw_ring()
    for path in paths:
        plt.plot(path[:, 0], path[:, 1], c='g')
    above_paths, below_paths = sort_paths(paths)

    for p in points:
        ref_paths = find_reference_paths(p, above_paths, below_paths)
        coefficients = find_coefficients(p, ref_paths)
        new_path = synthesize_path(ref_paths, coefficients)
        plt.plot(new_path[:, 0], new_path[:, 1], c='b')

    plt.savefig('synthesized_paths.png')

    for p in points:
        plt.clf()
        draw_ring()
        ref_paths = find_reference_paths(p, above_paths, below_paths)
        for path in ref_paths:
            plt.scatter(path[:, 0], path[:, 1], c='g', s=1)
        coefficients = find_coefficients(p, ref_paths)
        new_path = synthesize_path(ref_paths, coefficients)
        plt.scatter(new_path[:, 0], new_path[:, 1], c='b', s=1)
        plt.savefig("synthesized_from_point_{}_{}.png".format(p[0], p[1]))

    for p in points:
        plt.clf()
        draw_ring()
        ref_paths = find_reference_paths(p, above_paths, below_paths)
        for path in ref_paths:
            high_res_path = np.stack([interpolate_path(path, t) for t in np.arange(0, 50, 0.02)])
            plt.scatter(high_res_path[:, 0], high_res_path[:, 1], c='g', s=1)
        coefficients = find_coefficients(p, ref_paths)
        new_path = synthesize_path(ref_paths, coefficients)
        high_res_path = np.stack([interpolate_path(new_path, t) for t in np.arange(0, 50, 0.02)])
        plt.scatter(high_res_path[:, 0], high_res_path[:, 1], c='b', s=1)
        plt.savefig("high_res_synthesized_from_point_{}_{}.png".format(p[0], p[1]))


#q1()
#q3()
#q5()
#q6()
#q7()
#q8()
