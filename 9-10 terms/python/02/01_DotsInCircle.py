x0, y0, r = tuple(map(int, input().split(",")))
res = 'YES'
x, y = tuple(map(int, input().split(",")))
while (x, y) != (0, 0):
    if res != 'NO' and ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 > r:
        res = 'NO'
    x, y = tuple(map(int, input().split(",")))
print(res)
