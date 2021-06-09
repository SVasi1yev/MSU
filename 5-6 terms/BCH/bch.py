import random
import numpy as np
import gf


class BCH:
    def __init__(self, n, t):
        pripoly = None
        with open("primpoly.txt") as f:
            for line in f:
                for x in line.split(", "):
                    if (int(x) > n) and (int(x) < ((n + 1) * 2)):
                        pripoly = int(x)
                        break

        self.pm = gf.gen_pow_matrix(pripoly)
        temp = [self.pm[0][1]]
        for i in range(2 * t - 1):
            temp.append(gf.prod(np.array([[temp[i]]], dtype=int), np.array([[temp[0]]], dtype=int), self.pm)[0][0])
        self.R = np.array(temp, dtype=int)
        self.g = gf.minpoly(self.R, self.pm)[0]

    def encode(self, U):
        k = U.shape[1]
        n = self.pm.shape[0]
        m = n - k
        res = np.zeros((U.shape[0], n), dtype=int)
        xm = np.zeros(m + 1, dtype=int)
        xm[0] = 1
        for i in range(U.shape[0]):
            _, r = gf.polydiv(gf.polyprod(xm, U[i], self.pm), self.g, self.pm)
            temp = np.copy(gf.polyadd(gf.polyprod(xm, U[i], self.pm), r))
            for j in range(res[i].size - temp.size):
                res[i][j] = 0
            for j in range(res[i].size - temp.size, res[i].size):
                res[i][j] = temp[j - res[i].size + temp.size]

        return res

    def decode(self, W, method='euclid'):
        res = np.empty(W.shape)
        for i in range(W.shape[0]):
            s = gf.polyval(W[i].astype(int), self.R, self.pm)
            if s.sum() == 0:
                res[i] = np.copy(W[i])
                continue
            v = None
            L = None
            fl = True
            if method == 'pgz':
                for j in range(round(self.R.size / 2), 0, -1):
                    A = np.empty((j, j), dtype=int)
                    b = np.empty(j, dtype=int)
                    for k in range(j):
                        for t in range(j):
                            A[k][t] = s[k + t]
                        b[k] = s[j + k]
                    L = gf.linsolve(A, b, self.pm)
                    if not (L is np.nan):
                        L = np.append(L, 1)
                        v = j
                        fl = False
                        break
                if fl:
                    for j in range(res.shape[1]):
                        res[i][j] = np.nan
                        continue
            else:
                s_pol = np.empty(self.R.size + 1, dtype=int)
                for j in range(self.R.size):
                    s_pol[j] = s[-j - 1]
                s_pol[-1] = 1
                z_pol = np.zeros(self.R.size + 2, dtype=int)
                z_pol[0] = 1
                _, _1, L = gf.euclid(z_pol, s_pol, self.pm, max_deg=round(self.R.size / 2))
                v = L.size - 1
                fl = False

            errors = []
            for j in range(self.pm.shape[0]):
                if gf.polyval(L, np.array([self.pm[j][1]], dtype=int), self.pm)[0] == 0:
                    errors.append(j + 1)
            errors = np.array(errors, dtype=int)
            if not (errors.size == v):
                fl = True

            temp = np.copy(W[i].astype(int))
            for j in range(errors.size):
                temp[errors[j] - 1] = (temp[errors[j] - 1] + 1) % 2

            s = gf.polyval(temp, self.R, self.pm)
            if not (s.sum() == 0):
                fl = True

            if fl:
                for j in range(res.shape[1]):
                    res[i][j] = np.nan
            else:
                for j in range(res.shape[1]):
                    res[i][j] = temp[j]

        return res

    def dist(self):
        n = self.pm.shape[0]
        m = self.g.size - 1
        k = n - m
        min_ = n + 1
        U = np.empty(k, dtype=int)
        for i in range(2 ** k):
            temp = i
            for j in range(k):
                U[j] = temp % 2
                temp //= 2
            a = self.encode(np.array([U]))[0]
            if (a.sum() < min_) and (a.sum() > 0):
                min_ = a.sum()

        return min_

    def info(self):
        n = self.pm.shape[0]
        m = self.g.size - 1
        k = n - m
        r = k / n
        # print("n =", n)
        # print("m =", m)
        # print("k =", k)
        # print("r =", r)

        return n, m, k, r

    def words_gen(self):
        n = self.pm.shape[0]
        m = self.g.size - 1
        k = n - m
        # print(min(2 ** k, 100))
        U = np.empty((min(2 ** k, 100), k), dtype=int)
        for i in range(min(2 ** k, 100)):
            temp = i
            for j in range(k):
                U[i][j] = temp % 2
                temp //= 2
        a = self.encode(np.array(U))
        # print(U)
        # print(a)

        return a

    def stat(self, error_num, try_num):
        n, m, k, r = self.info()
        correct = 0
        wrong = 0
        deny = 0
        a = np.empty(k, dtype=int)
        for i in range(try_num):
            for j in range(a.size):
                a[j] = random.randint(0, 1)
            b = self.encode(np.array([a]))[0]
            c = np.copy(b)
            for j in range(error_num):
                c[j] = 1 - c[j]
            d = self.decode(np.array([c]))[0]
            if np.isnan(d[0]):
                deny += 1
            elif (b == d).all():
                correct += 1
            else:
                wrong += 1

        return correct / try_num, wrong / try_num, deny / try_num
