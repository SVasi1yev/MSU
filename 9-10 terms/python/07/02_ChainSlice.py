import itertools


def chainslice(begin, end, *seqs):
    chain = itertools.chain(*seqs)
    return itertools.islice(chain, begin, end)
