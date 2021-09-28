import numpy as np

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


