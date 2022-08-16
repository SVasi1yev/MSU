import random
import itertools


def cycle(iterable):
    saved = []
    for element in iterable:
        t = tuple(element)
        saved.append(t)
        yield t
    while saved:
        for element in saved:
              yield element


def randomes(p):
    for i, e in enumerate(cycle(p)):
        yield random.randint(e[0], e[1])
