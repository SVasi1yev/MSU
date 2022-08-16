k = int(input())

num = k
r, a = divmod(k * k, 10)
i = 1
while a != k or r != 0:
    num += 10**i * a
    i += 1
    r, a = divmod(k * a + r, 10)
print(num)
