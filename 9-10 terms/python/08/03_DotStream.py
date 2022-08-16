def g(a, n, step):
    for i in range(n):
        yield a + i * step


class Dots:
    def __init__(self, a, b):
        self.a = float(a)
        self.b = float(b)
        self.length = self.b - self.a

    def __getitem__(self, idx):
        if isinstance(idx, int):
            step = self.length / (idx - 1)
            res = g(self.a, idx, step)
            return res
        if isinstance(idx, slice):
            if idx.step is None:
                step = self.length / (idx.stop - 1)
                return self.a + idx.start * step
            else:
                step = self.length / (idx.step - 1)

                start = idx.start if idx.start is not None else 0
                stop = idx.stop if idx.stop is not None else idx.step
                res = g(self.a + start * step, stop - start, step)
                return res
