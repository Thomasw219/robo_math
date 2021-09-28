import numpy as np
import matplotlib.pyplot as plt

from collections import deque

def compute_coefficients(x, y):
    n = len(x)
    table = np.zeros((n, n))
    for i, y_i in enumerate(y):
        table[i, 0] = y_i

    for i in range(1, n):
        for j in range(i, n):
            table[j, i] = (table[j - 1][i - 1] - table[j][i - 1]) / (x[j - i] - x[j])

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


def newtons_method(x_0, function, derivative, conv_error=1e-7, conv_iter=5):
    past_values = deque()
    x_n = x_0
    while len(past_values) < conv_iter or np.max(np.abs(np.array(past_values) - x_n)) > conv_error:
        x_np1 = x_n - function(x_n) / derivative(x_n)
        past_values.appendleft(x_np1)
        if len(past_values) > conv_iter:
            past_values.pop()
        x_n = x_np1

    return past_values[0]
