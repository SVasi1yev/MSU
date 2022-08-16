from functools import wraps


def cast(type_):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            res = f(*args, **kwargs)
            return type_(res)
        return wrapper
    return decorator


# @cast(int)
# def fun(a, b):
#     return a * 2 + b
# print(fun(12, 34) * 2)
# print(fun("12", "34") * 2)
# print(fun(12.765, 34.654) * 2)