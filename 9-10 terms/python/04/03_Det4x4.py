matr = eval(input())

det = 0
sign_1 = 1
for i in range(4):
    sign_2 = 1
    for j in range(4):
        if i == j:
            continue
        t = tuple(k for k in range(4) if k not in (i, j))
        det += sign_1 * matr[0][i] * sign_2 * matr[1][j] \
            * (matr[2][t[0]] * matr[3][t[1]] - matr[2][t[1]] * matr[3][t[0]])
        sign_2 *= -1
    sign_1 *= -1

print(det)
