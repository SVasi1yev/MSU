from functools import wraps


def counter(f):
    # @wraps(f)
    class Wrapper:
        def __init__(self):
            self.count = 0

        def counter(self):
            return self.count

        def __call__(self, *args, **kwargs):
            self.count += 1
            return f(*args, **kwargs)

    wrapper = Wrapper()
    wrapper.__name__ = f.__name__
    return wrapper


# @counter
# def fun(a, b):
#     return a * 1 + b
#
# print(fun.counter())
# res = sum(fun(i, i + 1) for i in range(5))
# print(fun.counter(), res)
# print(fun.__name__)
