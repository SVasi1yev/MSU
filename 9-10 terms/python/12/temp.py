from inspect import getfullargspec
class A:
    def f(self, a: int, b, c, d, *e, f=1, **kwargs: int) -> int:
        print(a, b)
        print(c, d)
        print(e)
        print(kwargs)

a = A()
print(getfullargspec(a.f))
print(callable(a.f))

a.f(1, 2, 3, 4, e="qwe")