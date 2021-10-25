import numpy as np

def eulers_update(x_n, y_n, h, derivative):
    y_np1 = y_n + h * derivative(x_n, y_n)
    return y_np1

def rungekutta4_update(x_n, y_n, h, derivative):
    k_1 = h * derivative(x_n, y_n)
    k_2 = h * derivative(x_n + h / 2, y_n + k_1 / 2)
    k_3 = h * derivative(x_n + h / 2, y_n + k_2 / 2)
    k_4 = h * derivative(x_n + h, y_n + k_3)
    y_np1 = y_n + (k_1 + k_2 + k_3 + k_4) / 6
    return y_np1

def q1():

