num = int(input())
t = num
l = 0
while t > 0:
    t //= 10
    l += 1

for i in range(l // 2):
    a = num // (10 ** (l - 2 * i - 1))
    b = num % 10
    num = num % (10 ** (l - 2 * i - 1)) // 10
    if a != b:
        print('NO')
        break
else:
    print('YES')
