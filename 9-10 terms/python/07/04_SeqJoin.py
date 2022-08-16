def joinseq(*s):
    seqs = []
    for e in s:
        seqs.append(iter(e))
    a = []
    for seq in seqs:
        a.append(next(seq, None))
    min_ = 1
    while min_ is not None:
        argmin, min_ = -1, None
        for i in range(len(a)):
            if a[i] is None:
                continue
            if a[i] is not None:
                if (argmin == -1) or (min_ > a[i]):
                    argmin, min_ = i, a[i]
        if min_ is not None:
            yield min_
        a[argmin] = next(seqs[argmin], None)
