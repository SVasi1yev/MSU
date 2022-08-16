class DivStr(str):
    for k, f in str.__dict__.items():
        if callable(f):
            if k in {'__getitem__', '__add__', '__mul__', '__rmul__'} \
                    or not k.startswith('__'):
                def t(self, *args, f=f, **kwargs, ):
                    a = f(self.string, *args, **kwargs)
                    return a if type(a) != str else DivStr(a)
                t.__name__ = f.__name__
                locals()[k] = t

    def __init__(self, string):
        self.string = string

    def __len__(self):
        return len(self.string)

    def __floordiv__(self, num):
        c = len(self) // num
        res = []
        for i in range(num):
            res.append(DivStr(self.string[i * c: (i+1) * c]))
        return res

    def __mod__(self, num):
        r = (len(self) % num)
        return DivStr(self.string[-r:]) if r > 0 else ''

    def __str__(self):
        return self.string
