import importlib
import inspect
import textwrap
import ast
import difflib
import sys
from collections import deque


def proc_module(module_name):
    module = importlib.import_module(module_name)
    funcs = []
    d = deque()
    for mem in inspect.getmembers(module):
        d.append((f'{module_name}.{mem[0]}', mem[1]))
    while len(d) > 0:
        mem = d.popleft()
        if inspect.isfunction(mem[1]):
            funcs.append(mem)
        if inspect.isclass(mem[1]) and not mem[0].split('.')[-1].startswith('__'):
            for in_mem in inspect.getmembers(mem[1]):
                d.append((f'{mem[0]}.{in_mem[0]}', in_mem[1]))

    return list(funcs)


if __name__ == '__main__':
    funcs = []
    for module in sys.argv[1:]:
        funcs += proc_module(module)

    preps = {}
    for i in range(len(funcs)):
        funcs[i] = list(funcs[i])
        funcs[i].append(textwrap.dedent(inspect.getsource(funcs[i][1])))
        funcs[i].append(ast.parse(funcs[i][2]))
        for node in ast.walk(funcs[i][3]):
            for attr in ['name', 'id', 'arg', 'attr']:
                if hasattr(node, attr):
                    setattr(node, attr, '_')
        funcs[i].append(ast.unparse(funcs[i][3]))
        preps[funcs[i][0]] = funcs[i][4]

    names = list(preps.keys())
    out = []
    for i in range(len(names) - 1):
        for j in range(i + 1, len(names)):
            ratio = difflib.SequenceMatcher(None, preps[names[i]], preps[names[j]]).ratio()
            if ratio > 0.95:
                out.append(f'{names[i]} : {names[j]}')

    for e in sorted(out):
        print(e)
