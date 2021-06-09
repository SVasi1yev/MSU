import numpy as np


def gen_pow_matrix(pripoly):
    pr = pripoly
    q = -1
    while not (pr == 0):
        pr //= 2
        q += 1

    table = np.zeros((2 ** q - 1, 2), dtype=int)
    mask = 2 ** q
    r = pripoly ^ mask

    table[0][1] = 2
    table[1][0] = 1
    for i in range(1, 2 ** q - 1):
        table[i][1] = table[i - 1][1] * 2
        if not ((table[i][1] & mask) == 0):
            table[i][1] ^= mask
            table[i][1] ^= r
        table[table[i][1] - 1][0] = i + 1

    return table


def add(X, Y):
    return X ^ Y


def sum(X, axis=0):
    if axis == 0:
        res = np.zeros(X.shape[1], dtype=int)
        for i in range(X.shape[0]):
            res ^= X[i]
        res = res.reshape((1, res.size))
    else:
        res = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                res[j] ^= X[j][i]
        res = res.reshape((res.size, 1))

    return res


def prod(X, Y, pm):
    res = np.zeros(X.shape, dtype=int)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if not ((X[i][j] == 0) or (Y[i][j] == 0)):
                res[i][j] = pm[((pm[X[i][j] - 1][0] + pm[Y[i][j] - 1][0]) % pm.shape[0]) - 1][1]

    return res


def divide(X, Y, pm):
    res = np.zeros(X.shape, dtype=int)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if Y[i][j] == 0:
                return np.nan
            elif not (X[i][j] == 0):
                res[i][j] = pm[((pm[X[i][j] - 1][0] + pm.shape[0] - pm[Y[i][j] - 1][0]) % pm.shape[0]) - 1][1]

    return res


def linsolve(A, b, pm):
    A1 = np.copy(A)
    b1 = np.copy(b)
    res = np.zeros(A1.shape[1], dtype=int)

    for i in range(A1.shape[0]):
        fl = True
        for j in range(i, A1.shape[0]):
            if not (A1[j][i] == 0):
                fl = False
                temp = np.copy(A1[i])
                A1[i] = np.copy(A1[j])
                A1[j] = np.copy(temp)
                temp = b1[i]
                b1[i] = b1[j]
                b1[j] = temp
                break
        if fl:
            return np.nan

        b1[i] = divide(np.array([[b1[i]]]), np.array([[A1[i][i]]]), pm)[0][0]
        temp = divide(np.array([A1[i]]), np.array([np.full(A1.shape[1], A1[i][i], dtype=int)]), pm)
        A1[i] = np.copy(temp[0])

        for j in range(i + 1, A1.shape[0]):
            b1[j] = add(prod(np.array([[b1[i]]]), np.array([[A1[j][i]]]), pm)[0][0], b1[j])
            A1[j] = np.copy(add(prod(np.array([A1[i]]), np.array([np.full(A1.shape[1], A1[j][i], dtype=int)]), pm)[0], A1[j]))  

    b1[-1] = divide(np.array([[b1[-1]]]), np.array([[A1[-1][-1]]]), pm)[0][0]
    temp = divide(np.array([A1[-1]]), np.array([np.full(A1.shape[1], A1[-1][-1], dtype=int)]), pm)
    A1[-1] = np.copy(temp[0])

    for i in range(A1.shape[0] - 1, -1, -1):
        res[i] = b1[i]
        for j in range(i + 1, A1.shape[0]):
            res[i] = add(res[i], prod(np.array([[A1[i][j]]]), np.array([[res[j]]]), pm))

    return res


def minpoly(x, pm):
    roots = []
    for a in x:
        temp = a
        while not (temp in roots):
            roots.append(temp)
            temp = prod(np.array([[temp]]), np.array([[temp]]), pm)[0][0]

    roots = np.sort(np.array(roots, dtype=int))
    minpol = np.array([1], dtype=int)
    for a in roots:
        pol = np.array([1, a], dtype=int)
        minpol = polyprod(minpol, pol, pm)

    return (minpol, roots)


def polyval(p, x, pm):
    res = np.zeros(x.size, dtype=int)

    for i in range(x.size):
        if x[i]:
            for j in range(p.size):
                temp = pm[((pm[x[i] - 1][0] * (p.size - 1 - j)) % pm.shape[0]) - 1][1]
                temp = prod(np.array([[temp]]), np.array([[p[j]]]), pm)[0][0]
                res[i] = add(res[i], temp)
        else:
            res[i] = 0

    return res


def polyadd(p1, p2):
    res = np.empty(max(p1.size, p2.size), dtype=int)
    for i in range(1, min(p1.size, p2.size) + 1):
        res[-i] = add(p1[-i], p2[-i])
    for i in range(max(p1.size, p2.size) - min(p1.size, p2.size)):
        if p1.size > p2.size:
            res[i] = p1[i]
        else:
            res[i] = p2[i]
    i = 0
    while res[i] == 0:
        i += 1
        if i == res.size:
            i -= 1
            break
    res = np.array(res[i: res.size], dtype=int)

    return res


def polyprod(p1, p2, pm):
    res = np.zeros(p1.size + p2.size - 1, dtype=int)

    for i in range(p1.size):
        for j in range(p2.size):
            temp = prod(np.array([[p1[i]]]), np.array([[p2[j]]]), pm)[0][0]
            res[i + j] = add(res[i + j], temp)

    i = 0
    while res[i] == 0:
        i += 1
        if i == res.size:
            i -= 1
            break
    res = np.array(res[i: res.size], dtype=int)

    return res


def polydiv(p1, p2, pm):
    t = polyadd(p2, np.array([0], dtype=int))
    r = polyadd(p1, np.array([0], dtype=int))
    if t.size > r.size:
        return (np.array([0]), r)
    q = np.zeros(r.size - t.size + 1, dtype=int)
    while (r.size >= t.size) and not (r.size == 1 and r[0] == 0):
        temp = np.zeros(r.size - t.size + 1, dtype=int)
        temp[0] = divide(np.array([[r[0]]]), np.array([[t[0]]]), pm)[0][0]
        q = polyadd(q, temp)
        temp = polyprod(temp, t, pm)
        r = polyadd(r, temp)

    return (q, r)


def euclid(p1, p2, pm, max_deg=0):
    r1 = None
    r2 = None
    y1 = np.array([0], dtype=int)
    y2 = np.array([1], dtype=int)
    x1 = np.array([1], dtype=int)
    x2 = np.array([0], dtype=int)
    if p1.size >= p2.size:
        r1 = np.copy(p1)
        r2 = np.copy(p2)
    else:
        r1 = np.copy(p2)
        r2 = np.copy(p1)

    q, r1 = polydiv(r1, r2, pm)
    r1, r2 = r2, r1

    temp = polyprod(y2, q, pm)
    y1 = polyadd(temp, y1)
    y1, y2 = y2, y1

    temp = polyprod(x2, q, pm)
    x1 = polyadd(temp, x1)
    x1, x2 = x2, x1

    while (r2.size - 1) > max_deg:
        q, r1 = polydiv(r1, r2, pm)
        r1, r2 = r2, r1

        temp = polyprod(y2, q, pm)
        y1 = polyadd(temp, y1)
        y1, y2 = y2, y1

        temp = polyprod(x2, q, pm)
        x1 = polyadd(temp, x1)
        x1, x2 = x2, x1

    if p1.size >= p2.size:
        return (r2, x2, y2)
    else:
        return (r2, y2, x2)
