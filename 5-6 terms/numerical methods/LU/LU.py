def gen_formula():
    dim = int(input('enter dim: '))
    matr = [[0.0 for i in range(dim)] for j in range(dim)]

    for i in range(dim):
        for j in range(dim):
            matr[i][j] = 1.0 / (2 * dim - i - j - 1)

    b = [0.0] * dim
    for i in range(dim):
        for j in range(dim):
            b[i] += matr[i][j]

    return dim, matr, b


def file_input(filename):
    txt = ''
    n = 0

    f = open(filename)
    dim = int(f.read(1))
    matr = [[] for i in range(dim)]
    b = []
    f.read(1)
    c = f.read(1)
    for k in range(dim):
        while c != '\n':
            while(c != '\t') and (c != '\n'):
                txt += c
                c = f.read(1)
            num = float(txt)
            if n != dim:
                matr[k].append(num)
            else:
                b.append(num)
            n += 1
            txt = ''
            if c == '\t':
                c = f.read(1)
        n = 0
        txt = ''
        c = f.read(1)

    return dim, matr, b


def lu_decomposition(dim, matr, b, x):

    eps = 1e-19
    p = [i for i in range(dim)]
    for i in range(len(x)):
        x[i] = 0.0

    for i in range(dim):
        print("-------------")
        # print matr
        for g in range(dim):
            for h in range(dim):
                print("%.5f" % matr[p[g]][h], end='\t')
            print()
        print()

        for j in range(i, dim):
            summa = 0.0
            for k in range(i):
                summa += matr[p[j]][k] * matr[p[k]][i]
            matr[p[j]][i] = matr[p[j]][i] - summa
            if abs(matr[p[i]][i]) < eps:
                return None

        # print matr
        for g in range(dim):
            for h in range(dim):
                print("%.5f" % matr[p[g]][h], end='\t')
            print()
        print()

        m = 0.0
        maxi = i
        for j in range(i, dim):
            if abs(matr[p[j]][i]) > m:
                m = abs(matr[p[j]][i])
                maxi = j
        p[i], p[maxi] = p[maxi], p[i]

        # print matr
        for g in range(dim):
            for h in range(dim):
                print("%.5f" % matr[p[g]][h], end='\t')
            print()
        print()

        for j in range(i + 1, dim):
            summa = 0.0
            for k in range(i):
                summa += matr[p[i]][k] * matr[p[k]][j]
            matr[p[i]][j] = (matr[p[i]][j] - summa) / matr[p[i]][i]

    y = [0.0] * dim
    for j in range(dim):
        summa = 0.0
        for k in range(j):
            summa += matr[p[j]][k] * y[p[k]]
        y[p[j]] = (b[p[j]] - summa) / matr[p[j]][j]

    for j in range(dim - 1, -1, -1):
        summa = 0.0
        for k in range(dim - 1, j, -1):
            summa += matr[p[j]][k] * x[k]
        x[j] = (y[p[j]] - summa)

    return p


def residual(dim, matr, p, solution, b):
    x = [0.0] * dim
    for i in range(dim):
        x[p[i]] = solution[i]
        for j in range(i + 1, dim):
            x[p[i]] += solution[j] * matr[p[i]][j]

    y = [0.0] * dim
    for i in range(dim):
        for j in range(i + 1):
            y[p[i]] += x[p[j]] * matr[p[i]][j]

    z = [0.0] * dim
    for i in range(dim):
        z[i] = (b[i] - y[i]) ** 2

    res = 0.0
    for i in range(dim):
        res += z[i]

    return res ** 0.5


def lu_print(dim, matr, p, solution):
    if dim < 10:
        print('L:')
        for i in range(dim):
            for j in range(i + 1):
                print("%.5e" % matr[p[i]][j], end='\t')
            for j in range(i + 1, dim):
                print("%.5e" % 0.0, end='\t')
            print()

        print()

        print('U:')
        for i in range(dim):
            for j in range(i):
                print("%.5e" % 0.0, end='\t')
            print("%.5e" % U_DIAG, end='\t')
            for j in range(i + 1, dim):
                print("%.5e" % matr[p[i]][j], end='\t')
            print()

        print()

        print('Solution:')
        for i in range(dim):
            print("%.5e" % solution[i])
    else:
        output_file_name = '/home/svasilyev/projects/prak/LU/output_matrix'
        f = open(output_file_name, 'w')

        f.write('L:\n')
        for i in range(dim):
            for j in range(i + 1):
                f.write("%.3e" % matr[p[i]][j])
                f.write('\t')
            for j in range(i + 1, dim):
                f.write("%.3e" % 0.0)
                f.write('\t')
            f.write('\n')

        f.write('\n')

        f.write('U:\n')
        for i in range(dim):
            for j in range(i):
                f.write("%.3e" % 0.0)
                f.write('\t')
            f.write("%.3e" % U_DIAG)
            f.write('\t')
            for j in range(i + 1, dim):
                f.write("%.3e" % matr[p[i]][j])
                f.write('\t')
            f.write('\n')

        f.write('\n')

        f.write('Solution:\n')
        for i in range(dim):
            f.write("%.3f" % solution[p[i]])
            f.write('\n')


input_file_name = '/home/svasilyev/projects/prak/LU/input_matrix1'
U_DIAG = 1.000
q = int(input('0 - file, 1 - formula: '))
print()
if q == 0:
    system = file_input(input_file_name)
else:
    system = gen_formula()

for i in range(system[0]):
    for j in range(system[0]):
        print("%.5f" % system[1][i][j], end='\t')
    print()
print()

solution = [0.0] * system[0]
p = lu_decomposition(system[0], system[1], system[2], solution)
# print("p = ", p, end='\n\n')
if p:
    lu_print(system[0], system[1], p, solution)
    print()
    print("Residual: %.3e" % residual(system[0], system[1], p, solution, system[2]))
else:
    print("det A = 0")
