from collections import Counter


num = int(input())
factors = Counter()

while num > 1:
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            factors[i] += 1
            num //= i
            break
    else:
        if num > 1:
            factors[num] += 1
            num = 1

s = None
for k, v in factors.items():
    if s is None:
        if len(factors) == 1 and v == 1:
            print('NO')
            break
        s = set()
        s.add(v)
        for i in range(2, int(k**0.5) + 1):
            if v % i == 0:
                s.add(i)
    else:
        t = set()
        t.add(v)
        for i in range(2, int(k ** 0.5) + 1):
            if v % i == 0:
                t.add(i)
        s = s.intersection(t)
        if len(s) == 0:
            print('NO')
            break
else:
    print('YES')
