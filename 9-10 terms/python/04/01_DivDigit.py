def divdigit(num):
    t = num
    res = 0
    while t > 0:
        t, r = divmod(t, 10)
        if r != 0:
            res += int(num % r == 0)
    return res
