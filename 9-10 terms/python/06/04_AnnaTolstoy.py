import sys
import random
from collections import defaultdict
N = int(input())
txt = sys.stdin.read().replace("\n    ", " @ ").split()

d = defaultdict(list)
for i in range(len(txt) - 2):
    d[tuple(txt[i:i+2])].append(txt[i+2])

res = []
keys = list(d.keys())
t = random.choice(keys)
# print(t)
res += list(t) + [random.choice(d[t])]
N -= 3
for i in range(N):
    # print(res)
    res += [random.choice(d[tuple(res[-2:])])]

res = ' '.join(res)
res.replace(' @ ', '\n    ')
print(res)
