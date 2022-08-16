from collections import defaultdict, deque


def check_path():
    d_map = defaultdict(set)

    c = input().split()
    while len(c) == 2:
        d_map[c[0]].add(c[1])
        d_map[c[1]].add(c[0])
        c = input().split()
    d_enter = c[0]
    d_exit = input().strip()

    if d_enter == d_exit:
        return 'YES'

    proc = {d_enter}
    q = deque()
    for e in d_map[d_enter]:
        if e not in proc:
            if e == d_exit:
                return 'YES'
            q.append(e)

    while len(q) > 0:
        cur = q.popleft()
        proc.add(cur)
        for e in d_map[cur]:
            if e not in proc:
                if e == d_exit:
                    return 'YES'
                q.append(e)

    return 'NO'


print(check_path())
