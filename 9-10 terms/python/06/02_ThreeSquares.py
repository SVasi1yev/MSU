import math

data = set(eval(input()))

s = set()
M = max(data)
for i in range(1, int(math.sqrt(M) + 1)):
    sqrt_i = i*i
    for j in range(i, int(math.sqrt(M - sqrt_i) + 1)):
        sqrt_i_j = sqrt_i + j*j
        for k in range(j, int(math.sqrt(M - sqrt_i_j) + 1)):
            s.add(sqrt_i_j + k*k)

print(len(data.intersection(s)))
