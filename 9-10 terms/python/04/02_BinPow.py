def BinPow(a, n, f):
    res = None
    bin_n = str(bin(n))[2:]
    for e in bin_n:
        if e == '1':
            if res is None:
                res = a
            else:
                res = f(f(res, res), a)
        else:
            res = f(res, res)
    return res
