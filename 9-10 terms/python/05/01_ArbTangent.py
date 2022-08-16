from math import factorial
from decimal import Decimal, getcontext


deg = Decimal(input())
prec = int(input())
getcontext().prec = prec + 10
EPS = Decimal(f'1e-{prec + 10}')


def decimal_fact(k):
    res = Decimal(1)
    for i in range(1, int(k+1)):
        res *= Decimal(i)

    return res


def count_pi():
    prev = Decimal(-1)
    cur = Decimal(0)
    sign = Decimal(1)
    k = Decimal(0)
    k_0 = 0

    fact_1 = Decimal(1)
    fact_2 = Decimal(1)
    fact_3 = Decimal(1)
    fact = Decimal(1)

    t = Decimal(640320)**3
    a = Decimal('0.5')
    c = Decimal(545140134)
    c_0 = Decimal(0)
    b = Decimal(13591409)
    d = Decimal(12)

    while abs((Decimal(1) / (cur if cur != Decimal(0) else Decimal('0.1'))) - (Decimal(1) / (prev if prev != Decimal(0) else Decimal('0.1')))) > EPS:
        prev = cur
        # cur += d * sign * fact_1 * (b+c*k) / \
        #     (fact_2 * fact_3 * t**(k+a))
        # cur += d * sign * factorial(6*k_0) * b / \
        #        (factorial(3*k_0) * factorial(k_0) ** 3 * t ** (k + a))
        cur += d * sign * fact * (b + c_0) / (t ** (k + a))

        for i in range(6 * k_0 + 1, 6 * k_0 + 7):
            fact *= i
        for i in range(3 * k_0 + 1, 3 * k_0 + 4):
            fact /= i
        for i in range(3):
            fact /= k_0 + 1

        k += 1
        c_0 += c

        # b += c
        # for i in range(6 * k_0 + 1, 6 * (k_0 + 1) + 1):
        #     fact_1 *= i
        # for i in range(3 * k_0 + 1, 3 * (k_0 + 1) + 1):
        #     fact_2 *= i
        # fact_3 *= Decimal(k_0 + 1) ** 3
        k_0 += 1

        sign *= -1
    print(k_0)
    return Decimal(1) / cur


def count_sin(rad, pi):
    prev = Decimal(-1)
    cur = Decimal(0)
    sign = Decimal(1)
    k = Decimal(0)
    k_0 = 0
    fact_1 = Decimal(1)
    while abs(cur - prev) > EPS:
        prev = cur
        cur += sign * rad**(2*k+1) / fact_1
        sign *= -1
        k += 1

        for i in range(2 * k_0 + 2, 2 * (k_0 + 1) + 2):
            fact_1 *= i
        k_0 += 1

    return cur


def count_cos(rad, pi):
    prev = Decimal(-1)
    cur = Decimal(0)
    sign = Decimal(1)
    k = Decimal(0)
    k_0 = 0
    fact_1 = Decimal(1)
    while abs(cur - prev) > EPS:
        prev = cur
        cur += sign * rad ** (2 * k) / fact_1
        sign *= -1
        k += 1

        for i in range(2 * k_0 + 1, 2 * (k_0 + 1) + 1):
            fact_1 *= Decimal(i)
        k_0 += 1

    return cur

from time import time
s = time()
pi = count_pi()
rad = deg * pi / Decimal(200)
sin = count_sin(rad, pi)
cos = count_cos(rad, pi)
getcontext().prec = prec
print(sin / cos)
print(time() - s)
