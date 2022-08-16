intervals = sorted(list(eval(input())), key=lambda x: x[0])

i = 0
while i < len(intervals) - 1:
    if intervals[i][1] > intervals[i+1][0]:
        intervals[i] = (intervals[i][0], max(intervals[i][1], intervals[i+1][1]))
        del intervals[i+1]
    else:
        i += 1

res = 0
for e in intervals:
    res += e[1] - e[0]

print(res)



