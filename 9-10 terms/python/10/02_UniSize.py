class Dsc:
    def __get__(self, obj, cls):
        if '__len__' in dir(obj):
            return len(obj)
        elif '__abs__' in dir(obj):
            return abs(obj)
        else:
            return 0


def sizer(cls):
    cls.size = Dsc()
    def wrapper(*args, **kwargs):
        obj = cls(*args, **kwargs)
        # obj.size = Dsc()
        return obj
    return wrapper


# @sizer
# class S(str):
#     pass
#
#
# @sizer
# class N(complex):
#     pass
#
#
# @sizer
# class E(Exception):
#     pass
#
#
# for obj in S("QWER"), N(3+4j), E("Exceptions know no lengths!"):
#     print(obj, obj.size)
