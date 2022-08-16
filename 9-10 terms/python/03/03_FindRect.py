prev, cur = input(), input()
n = len(prev)
res = 0
while cur[0] != '-':
    for i in range(n):
        if cur[i] == '#' and (i == 0 or cur[i-1] == '.') and prev[i] in ('.', '-'):
            res += 1
    prev, cur = cur, input()
print(res)





