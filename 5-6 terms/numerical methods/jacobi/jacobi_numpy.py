import timeit
import numpy as np


def sign(x):
    if x == 0:
        return 0.0
    elif x > 0:
        return 1.0
    else:
        return -1.0


def gen_formula():
    dim = int(input('enter dim: '))
    matr = np.eye(dim, dtype=np.float128)
    for i in range(dim):
        matr[dim - 1][i] = i + 1
        matr[i][dim - 1] = i + 1

    return dim, matr


def file_input(input_file):
    f = open(input_file)
    dim = int(f.readline())
    matr = np.loadtxt("input_matrix.txt", skiprows=1, dtype=np.float128)

    return dim, matr


def search_max_elem(dim, matr, rows_sums):
    m = np.float128(0.0)
    mi, mj = 0, 0

    for i in range(dim - 1):
        for j in range(i + 1, dim):
            if abs(matr[i][j]) > m:
                m, mi, mj = abs(matr[i][j]), i, j

    return mi, mj


def search_opt_elem(dim, matr, rows_sums):
    m = np.float128(0.0)
    mi, mj = 0, 0

    for i in range(dim - 1):
        if rows_sums[i] > m:
            m, mi = rows_sums[i], i
    m = np.float128(0.0)
    for j in range(mi + 1, dim):
        if abs(matr[mi][j]) > m:
            m, mj = abs(matr[mi][j]), j

    return mi, mj


def cycle_search():
    mi, mj = 0, 1

    def incrementer(dim, matr, rows_sums):
        nonlocal mi, mj
        if mj == dim - 1:
            if mi == dim - 2:
                mi, mj = 0, 1
            else:
                mi += 1
                mj = mi + 1
        else:
            mj += 1

        return mi, mj

    return incrementer


def jacobi(dim, matr, eigenval, eigenvect, eps, mode):
    rows_sums = np.zeros(dim - 1, dtype=np.float128)
    get_ij = None
    if mode == 0:
        get_ij = search_max_elem
    elif mode == 1:
        get_ij = search_opt_elem
    else:
        get_ij = cycle_search()

    summa = np.float128(0.0)
    for i in range(dim - 1):
        for j in range(i + 1, dim):
            rows_sums[i] += matr[i][j] ** 2
            summa += matr[i][j] ** 2
    summa *= 2
    iter_num = 0
    time = timeit.default_timer()

    while summa > eps:
        iter_num += 1
        print(iter_num, summa)
        '''if iter_num % 10000 == 0:
            print(matr)'''
        mi, mj = get_ij(dim, matr, rows_sums)
        matr_ij = matr[mi][mj]
        x = -2 * matr[mi][mj]
        y = matr[mi][mi] - matr[mj][mj]
        cos_fi = None
        sin_fi = None
        if abs(y) < 1e-20:
            cos_fi = np.float128(1 / 2 ** 0.5)
            sin_fi = np.float128(1 / 2 ** 0.5)
        else:
            cos_fi = np.float128((1 / 2 + abs(y) / (2 * (x ** 2 + y ** 2) ** 0.5)) ** 0.5)
            sin_fi = np.float128(sign(x * y) * abs(x) / (2 * cos_fi * (x ** 2 + y ** 2) ** 0.5))

        for k in range(dim):
            save_elem = matr[mi][k]
            save_elem_2 = matr[mj][k]
            matr[mi][k] = cos_fi * matr[mi][k] - sin_fi * matr[mj][k]
            matr[mj][k] = sin_fi * save_elem + cos_fi * matr[mj][k]
            if k > mi:
                rows_sums[mi] -= (save_elem ** 2 - matr[mi][k] ** 2)
            if k > mj:
                rows_sums[mj] -= (save_elem_2 ** 2 - matr[mj][k] ** 2)

        for k in range(dim):
            save_elem = matr[k][mi]
            save_elem_2 = matr[k][mj]
            matr[k][mi] = cos_fi * matr[k][mi] - sin_fi * matr[k][mj]
            matr[k][mj] = sin_fi * save_elem + cos_fi * matr[k][mj]
            if k < mi:
                rows_sums[k] -= (save_elem ** 2 - matr[k][mi] ** 2)
            if k < mj:
                rows_sums[k] -= (save_elem_2 ** 2 - matr[k][mj] ** 2)

        for k in range(dim):
            save_elem = eigenvect[k][mi]
            eigenvect[k][mi] = cos_fi * eigenvect[k][mi] - sin_fi * eigenvect[k][mj]
            eigenvect[k][mj] = sin_fi * save_elem + cos_fi * eigenvect[k][mj]

        summa += 2 * (matr[mi][mj] ** 2 - matr_ij ** 2)

    for i in range(dim):
        eigenval[i] = matr[i][i]

    for i in range(dim):
        for j in range(dim):
            eigenvect[i][j] /= eigenvect[dim - 1][j]

    time = timeit.default_timer() - time

    return iter_num, time


def print_result(dim, eigenval, eigenvect, output_file, iter_num, time):
    print("iter_num = " + str(iter_num) + "\ttime = "
          + ("%.8f" % time) + " sec\n")
    for x in eigenval:
        print("{:.4e}".format(x), end='\t')
    print('\n')
    if dim < 10:
        for i in range(dim):
            print('#' + str(i + 1), end='\t\t')
        print()
        for x in eigenvect:
            for y in x:
                print("{:.4e}".format(y), end='\t')
            print()
    else:
        f = open(output_file, 'w')
        for i in range(dim):
            f.write('#' + str(i + 1) + '\t\t')
        f.write('\n')
        for x in eigenvect:
            for y in x:
                f.write(("{:.4e}".format(y)) + '\t')
            f.write('\n')


INPUT_FILE = '/home/svasilyev/projects/prak/jacobi/input_matrix.txt'
OUTPUT_FILE = '/home/svasilyev/projects/prak/jacobi/output_matrix.txt'
EPS = 1e-8
mode = int(input("1 - file / 2 - formula: "))
if mode == 1:
    dim, matr = file_input(INPUT_FILE)
else:
    dim, matr = gen_formula()
eigenval, eigenvect = np.empty(dim, dtype=np.float128), np.eye(dim, dtype=np.float128)
mode = int(input("enter mode -- 0 - max, 1 - opt, 2 - cycle: "))
print()
iter_num, time = jacobi(dim, matr, eigenval, eigenvect, EPS, mode)
print_result(dim, eigenval, eigenvect, OUTPUT_FILE, iter_num, time)
