def superposition(funmod, funseq):
    res = []
    for f in funseq:
        def tmp(x, f1=funmod, f2=f):
            return f1(f2(x))
        res.append(tmp)
    return res
