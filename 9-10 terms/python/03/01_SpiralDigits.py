m, n = tuple(map(int, input().split(',')))

BOARD_LEN = 2 * m + 2 * n - 4

for i in range(n):
    for j in range(m):
        sides = ('u', 'r', 'd', 'l')
        offsets = (i, m-j-1, n-i-1, j)

        p = ('u', i, 0)
        for k in range(4):
            if offsets[k] < p[1]:
                p = (sides[k], offsets[k], k)

        sizes = (m - 2 * p[1], n - 1 - 2 * p[1], m - 1 - 2 * p[1], n - 2 - 2 * p[1])
        adds = (j - p[1], i - p[1] - 1, m - p[1] - j - 2, n - p[1] - i - 2)

        num = p[1] * (2 * BOARD_LEN - 8 * (p[1] - 1)) / 2
        num += sum(sizes[:p[2]]) + adds[p[2]]

        print(round(num % 10), end=' ')
    print()
