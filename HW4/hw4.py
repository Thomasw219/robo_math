import numpy as np
import matplotlib.pyplot as plt

def euler_update(x_n, y_n, h, params):
    derivative = params["derivative"]
    y_np1 = y_n + h * derivative(x_n, y_n)
    return y_np1

def rungekutta4_update(x_n, y_n, h, params):
    derivative = params["derivative"]
    k_1 = h * derivative(x_n, y_n)
    k_2 = h * derivative(x_n + h / 2, y_n + k_1 / 2)
    k_3 = h * derivative(x_n + h / 2, y_n + k_2 / 2)
    k_4 = h * derivative(x_n + h, y_n + k_3)
    y_np1 = y_n + (k_1 + k_2 + k_3 + k_4) / 6
    return y_np1

def adamsbashforth_update(x_n, y_n, h, params):
    f_n = params["f_n"]
    f_nm1 = params["f_nm1"]
    f_nm2 = params["f_nm2"]
    f_nm3 = params["f_nm3"]
#    print(y_n)
#    print(f_n, f_nm1, f_nm2, f_nm3)
#    print((h / 24) * (55 * f_n - 59 * f_nm1 + 37 * f_nm2 - 9 * f_nm3))
    y_np1 = y_n + (h / 24) * (55 * f_n - 59 * f_nm1 + 37 * f_nm2 - 9 * f_nm3)
    return y_np1

def interpolate(x_0, y_0, h, steps, derivative=None, initialization_points=None, update=None):
    x = [x_0]
    y = [y_0]
    for n in range(steps):
        if initialization_points is None:
            params = {"derivative" : derivative}
        else:
            params = {"derivative" : derivative,
                "f_n" : initialization_points[-1],
                "f_nm1" : initialization_points[-2],
                "f_nm2" : initialization_points[-3],
                "f_nm3" : initialization_points[-4]}
        x_np1 = x[-1] + h
        y_np1 = update(x[-1], y[-1], h, params)
        x.append(x_np1)
        y.append(y_np1)
#        print(x_np1, y_np1)
        if initialization_points is not None:
            initialization_points.append(derivative(x_np1, y_np1))
    return x, y

def q1_derivative(x, y):
    return 1 / (2 * y)

def q1():
    x_0 = 2
    y_0 = 1
    h = -0.05
    plt.figure()
    euler_x, euler_y = interpolate(x_0, y_0, h, 20, derivative=q1_derivative, update=euler_update)
    rk4_x, rk4_y = interpolate(x_0, y_0, h, 20, derivative=q1_derivative, update=rungekutta4_update)
    # Given function values at these points
    initialization_points = [q1_derivative(2.15, 1.07238052947636), q1_derivative(2.10, 1.04880884817015), q1_derivative(2.05, 1.02469507659596), q1_derivative(2.0, 1)]
    # Actual function values at initialization points
    #initialization_points = [q1_derivative(2.15, 1.05207427), q1_derivative(2.10, 1.03492411), q1_derivative(2.05, 1.01756854), q1_derivative(2.0, 1)]
    ab_x, ab_y = interpolate(x_0, y_0, h, 20, derivative=q1_derivative, initialization_points=initialization_points, update=adamsbashforth_update)
    true_x = euler_x
    true_y = np.sqrt(true_x) + 1 - np.sqrt(2)
    plt.plot(euler_x, euler_y, label="Euler solution")
    plt.plot(rk4_x, rk4_y, label="Runge Kutta 4 solution")
    plt.plot(ab_x, ab_y, label="Adams Bashforth solution")
    plt.plot(true_x, true_y, label="True function value")
    plt.legend()
    plt.savefig("./figures/q1.png")
    table = np.stack([true_x, true_y, euler_y, np.abs(euler_y - true_y), rk4_y, np.abs(rk4_y - true_y), ab_y, np.abs(ab_y - true_y)], axis=-1)
    for k, x in enumerate(table):
        print("{} & ".format(k), end='')
        for i, y in enumerate(x):
            if i == 0:
                print("{:.2f}".format(y), end='')
            elif i > 1 and i % 2 == 1:
                print("{:.4f}".format(y), end='')
            else:
                print("{:.2f}".format(y), end='')

            if i < x.shape[0] - 1:
                print(" & ", end='')
            else:
                print(" \\\\")

        print("\\hline")

q1()
