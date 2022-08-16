import math


num = int(input())

for i in range(2, int(num**0.5) + 1):
    p = math.log(num, i)
    if math.fabs(int(p) - p) < 1e-6:
        print('YES')
        break
else:
    print('NO')
