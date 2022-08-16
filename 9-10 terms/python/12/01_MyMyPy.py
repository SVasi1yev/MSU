from inspect import getfullargspec


class checked(type):
    def __new__(cls, name, parents, ns):
        new_ns = {}
        for attr in ns:
            # print(attr)
            if callable(ns[attr]):

                def wrapper(*args, __f=ns[attr], **kwargs):
                    fargs = getfullargspec(__f)
                    # print(args, kwargs, __f)
                    # print(fargs)
                    proc = set()
                    defaults = {}
                    e = [[], [], [], []]

                    if fargs.defaults is not None:
                        for i in range(len(fargs.defaults)):
                            defaults[fargs.args[-(i+1)]] = fargs.defaults[-(i+1)]
                    if fargs.kwonlydefaults is not None:
                        for k in fargs.kwonlydefaults:
                            defaults[k] = fargs.kwonlydefaults[k]

                    for kwarg in kwargs:
                        if kwarg == fargs.varargs and kwarg in fargs.annotations:
                            for el in kwargs[kwarg]:
                                if not isinstance(el, fargs.annotations[kwarg]):
                                    e[2].append(TypeError(f'Type mismatch: {kwarg}'))
                        if kwarg in fargs.args:
                            proc.add(kwarg)
                            if kwarg in fargs.annotations and not isinstance(kwargs[kwarg], fargs.annotations[kwarg]):
                                e[1].append(TypeError(f'Type mismatch: {kwarg}'))
                        elif kwarg in fargs.kwonlyargs:
                            proc.add(kwarg)
                            if kwarg in fargs.annotations and not isinstance(kwargs[kwarg], fargs.annotations[kwarg]):
                                e[1].append(TypeError(f'Type mismatch: {kwarg}'))
                        else:
                            # print(kwarg, kwargs[kwarg], fargs.varkw)
                            proc.add(kwarg)
                            if fargs.varkw is not None and fargs.varkw in fargs.annotations and not isinstance(kwargs[kwarg], fargs.annotations[fargs.varkw]):
                                e[3].append(TypeError(f'Type mismatch: {fargs.varkw}'))

                    i = 0
                    for arg in args:
                        if i < len(fargs.args):
                            if fargs.args[i] not in proc:
                                proc.add(fargs.args[i])
                                if fargs.args[i] in fargs.annotations and not isinstance(arg,
                                                                                 fargs.annotations[fargs.args[i]]):
                                    e[0].append(TypeError(f'Type mismatch: {fargs.args[i]}'))
                        else:
                            if fargs.varargs is not None and fargs.varargs in fargs.annotations \
                                    and not isinstance(arg, fargs.annotations[fargs.varargs]):
                                e[2].append(TypeError(f'Type mismatch: {fargs.varargs}'))
                        i += 1

                    for arg in set(fargs.args).difference(proc):
                        if arg in defaults and arg in fargs.annotations \
                                and not isinstance(defaults[arg], fargs.annotations[arg]):
                            e[0].append(TypeError(f'Type mismatch: {arg}'))

                    # print(e)
                    for i in range(len(e)):
                        if len(e[i]) > 0:
                            raise e[i][0]
                    res = __f(*args, **kwargs)
                    if 'return' in fargs.annotations and not isinstance(res, fargs.annotations['return']):
                        raise TypeError(f'Type mismatch: {"return"}')

                    return res

                wrapper.name = attr
                new_ns[attr] = wrapper
            else:
                new_ns[attr] = ns[attr]

        # for attr in new_ns:
        #     if callable(new_ns[attr]):
        #         print(new_ns[attr].__closure__)

        return super().__new__(cls, name, parents, new_ns)



# class E(metaclass=checked):
#     def __init__(self, var: int):
#         # print('__init__')
#         self.var = var if var % 2 else str(var)
#
#     def mix(self, val: int, opt) -> int:
#         return self.var*val + opt
#
#     def al(self, c: int, d: int=1, *e:int, f:int=1, **g:int):
#         return self.var*d
#
# e1 = E(1)
# e2 = E(2)
# code = """
# e1.mix("q", "q")
# e1.mix(2, 3)
# e2.mix(2, "3")
# e1.al("q")
# e1.al(1, 2, 3, 4, 5, 6, foo=7, bar=8)
# e2.al(1, 2, 3, 4, 5, 6, foo=7, bar=8)
# e1.al("E", 2, 3, 4, 5, 6, foo=7, bar=8)
# e1.al(1, "E", 3, 4, 5, 6, foo=7, bar=8)
# e1.al(1, 2, "E", 4, 5, 6, foo=7, bar=8)
# e1.al(1, 2, 3, "E", 5, 6, foo="7", bar=8)
# e1.al(1, f="E", d=1)
# e1.al(1, f=1, d="E")
# e1.al(1, f="E", d="1")
# e1.al(1, d="E", f="1")
# e1.al(1, e="E")
# e1.al(1, g="E")
# """
#
# for c in code.strip().split("\n"):
#     try:
#         res = eval(c)
#     except TypeError as E:
#         res = E
#     print(f"Run: {c}\nGot: {res}")
