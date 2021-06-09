import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import time


h = 0.01
d = 0.00001
t0 = 1.0
l_cond_type = 2
r_cond_type = 1


a = 4.0


def u(t, x):
    return ((t ** 2.0) + 1.0) * x - 1.0 + (2.0 / (25.0 * (np.pi ** 2.0))) * (1.0 - np.exp(-25.0 * (np.pi ** 2.0) * t)) * np.cos(5.0 * np.pi * x / 2.0)
    # return 0.0


def u0(x):
    return x - 1.0
    # return 35


def fi_l(t):
    return (t ** 2.0) + 1.0
    # return 100


def fi_r(t):
    return (t ** 2.0)
    # return 70


def f(t, x):
    return 2.0 * t * x + 2.0 * np.cos(5.0 * np.pi * x / 2.0)
    # return 0.0


def explicit_difference_scheme(h, d, t0, l_cond_type, r_cond_type, visual_mode):
    x_part = int(1.0 / h) + 1
    t_part = int(1.0 / d) + 1
    x_points = np.linspace(0.0, 1.0, x_part)
    t_points = np.linspace(0.0, 1.0, t_part)
    v_1 = np.zeros(x_part)
    v_2 = np.vectorize(u0)(x_points)
    if (l_cond_type == 1):
        v_2[0] = fi_l(0.0)
    else:
        v_2[0] = (2 * h * fi_l(0.0) - 4 * v_2[1] + v_2[2]) / (-3)
    if (r_cond_type == 1):
        v_2[-1] = fi_r(0.0)
    else:
        v_2[-1] = 2 * h * fi_r(0.0) + 4 * v_2[-2] - v_2[-3]

    fig, ax, line1 = None, None, None
    if visual_mode:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line1, = ax.plot(x_points, v_2, 'r-')

    n = 0
    while t_points[n] < t0:
        # print(v_2)

        if visual_mode:
            line1.set_ydata(v_2)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)

        n += 1
        v_1 = np.copy(v_2)
        for m in range(1, x_part - 1):
            v_2[m] = d * ((a * (v_1[m + 1] - 2 * v_1[m] + v_1[m - 1]) / (h ** 2)) + f(n * d, m * h)) + v_1[m]
        if (l_cond_type == 1):
            v_2[0] = fi_l(n * d)
        else:
            v_2[0] = (2 * h * fi_l(n * d) - 4 * v_2[1] + v_2[2]) / (-3)
        if (r_cond_type == 1):
            v_2[-1] = fi_r(n * d)
        else:
            v_2[-1] = 2 * h * fi_r(n * d) + 4 * v_2[-2] - v_2[-3]

    return v_2


def implicit_difference_scheme(h, d, t0, l_cond_type, r_cond_type, visual_mode):
    def trid_matr_alg(v_1, d, n, h):
        q = d / (h ** 2)
        M = len(v_1)
        # print(M)
        v_2 = np.zeros(M)
        alpha = np.zeros(M)
        beta = np.zeros(M)
        hi_1 = 0.0
        hi_2 = 0.0
        nu_1 = 0.0
        nu_2 = 0.0
        if (l_cond_type == 1):
            hi_1 = 0.0
            nu_1 = fi_l(n * d)
        else:
            hi_1 = 1.0
            nu_1 = -fi_l(n * d) * h
        if (r_cond_type == 1):
            hi_2 = 0.0
            nu_2 = fi_r(n * d)
        else:
            hi_2 = 1.0
            nu_2 = fi_r(n * d) * h
        alpha[1] = hi_1
        beta[1] = nu_1
        for i in range(1, M - 1):
            alpha[i + 1] = q / ((2.0 * q + 1.0) - alpha[i] * q)
            beta[i + 1] = (q * beta[i] + v_1[i] + d * f((n + 1) * d, i * h)) / ((2.0 * q + 1.0) - alpha[i] * q)
        v_2[-1] = (nu_2 + hi_2 * beta[-1]) / (1.0 - hi_2 * alpha[-1])
        for i in range(M - 2, -1, -1):
            v_2[i] = alpha[i + 1] * v_2[i + 1] + beta[i + 1]

        return v_2

    x_part = int(1.0 / h) + 1
    t_part = int(1.0 / d) + 1
    x_points = np.linspace(0.0, 1.0, x_part)
    t_points = np.linspace(0.0, 1.0, t_part)
    v_1 = np.zeros(x_part)
    v_2 = np.vectorize(u0)(x_points)
    if (l_cond_type == 1):
        v_2[0] = fi_l(0.0)
    else:
        v_2[0] = v_2[1] - h * fi_l(0.0)
    if (r_cond_type == 1):
        v_2[-1] = fi_r(0.0)
    else:
        v_2[-1] = v_2[-2] + h * fi_r(0.0)
        
    fig, ax, line1 = None, None, None
    if visual_mode:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        line1, = ax.plot(x_points, v_2, 'r-')

    n = 0
    while t_points[n] < t0:
        # print(v_2)

        if visual_mode:
            line1.set_ydata(v_2)
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.01)

        n += 1
        v_1 = np.copy(v_2)
        v_2 = trid_matr_alg(v_1=v_1, d=d, n=n, h=h)
    plt.close()

    return v_2


def errors(h, v_values, t0):
    x_points = np.linspace(0.0, 1.0, len(v_values))
    # print(x_points)
    u_values = np.vectorize(u)(t0, x_points)
    print(u_values)
    abs_C_norm = np.max(np.abs(u_values - v_values))
    rel_C_norm = abs_C_norm / np.max(np.abs(u_values))
    abs_I_norm = (h ** 0.5) * la.norm(u_values - v_values)
    rel_I_norm = abs_I_norm / ((h ** 0.5) * la.norm(u_values))

    return abs_C_norm, rel_C_norm, abs_I_norm, rel_I_norm


mode = int(input('enter mode 0 -- test 1 -- visual: '))
scheme = int(input('enter scheme 0 -- explicit 1 -- implicit: '))
if scheme:
    scheme = implicit_difference_scheme
else:
    scheme = explicit_difference_scheme

v = scheme(h=h, d=d, t0=t0, l_cond_type=l_cond_type, r_cond_type=r_cond_type, visual_mode=mode)
print(v)
if not mode:
    err = errors(h=h, v_values=v, t0=t0)
    print('abs_C_err =', err[0])
    print('rel_C_err =', err[1])
    print('abs_I_err =', err[2])
    print('rel_I_err =', err[3])
