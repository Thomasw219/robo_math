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

    print("Actual y value: {}".format(1 / (1 + 36 * 0.006**2)))

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
    if np.abs(y[2]) < conv_error:
        return x[2]
    print(y)
    differences_table = compute_table(x, y, dtype='complex')
    a = differences_table[2, 2]
    b = differences_table[2, 1] + a * (x[2] - x[1])
    c = y[2]
    print(a, b, c)

    rooted_determinant = np.sqrt((b**2 - 4 * a * c).astype('complex'))
    print(rooted_determinant)
    denom1 = b + rooted_determinant
    denom2 = b - rooted_determinant
    if np.abs(denom1) > np.abs(denom2):
        root = 2 * c / denom1
    else:
        root = 2 * c / denom2
    print(root)
    return root

def deflated_function(roots, polynomial, x):
    output = polynomial(x)
    for r in roots:
        output /= (x - r)
    return output

def mullers_method(x_m2, x_m1, x_0, num_roots, function, conv_error=1e-10, conv_iter=5, max_iter=10):
    roots = []
    iterations = []
    deflated_function = function
    for i in range(num_roots):
        print("Root: {}".format(i + 1))
        past_values = deque()
        iteration = 0
        interpolate_points = deque([x_m2, x_m1, x_0])
        while (len(past_values) < conv_iter or np.max(np.abs(np.array(past_values) - interpolate_points[-1])) > conv_error) and iteration < max_iter:
            print(interpolate_points)
            x = np.array(interpolate_points)
            x_np1 = x[2] - quadratic_interpolation(x, function, conv_error)
            past_values.appendleft(x_np1)
            if len(past_values) > conv_iter:
                past_values.pop()
            interpolate_points.popleft()
            interpolate_points.append(x_np1)
            iteration += 1

        root = past_values[0]
        deflated_function = partial(deflated_function, roots, function)
        roots.append(root)
        iterations.append(iteration)

    return roots, iterations

def polynomial_function(x):
    return np.power(x, 2)
#    return np.power(x, 4) + x + 1

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

#q1()
#q3()
q5()
#q6()
