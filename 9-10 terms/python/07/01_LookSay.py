def LookSay():
    yield 1
    prev = '1'
    cur = ''
    while True:
        cur_c = None
        count = 0
        for c in prev:
            if cur_c is None:
                cur_c = c
                count = 1
            elif cur_c == c:
                count += 1
            else:
                yield count
                cur += str(count)
                yield int(cur_c)
                cur += cur_c
                cur_c = c
                count = 1
        yield count
        cur += str(count)
        yield int(cur_c)
        cur += cur_c
        prev, cur = cur, ''
