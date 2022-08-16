data = []
c = input()
while ' ' in c:
    t = c.split()
    data.append((float(t[0]), float(t[1]), float(t[2]), t[3]))
    c = input()

max_dist = -1
pair = None
for i in range(len(data) - 1):
    for j in range(i + 1, len(data)):
        dist = sum((data[i][k] - data[j][k])*(data[i][k] - data[j][k]) for k in range(3))
        if max_dist < dist:
            max_dist = dist
            pair = (i, j)

print(' '.join(sorted([data[pair[0]][3], data[pair[1]][3]])))
