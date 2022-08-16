class Tester:
    def __init__(self, fun):
        self.fun = fun
        
    def __call__(self, suite, allowed=()):
        res = 0
        for s in suite:
            try:
                self.fun(*s)
            except tuple(allowed) as e:
                res = -1
            except Exception as e:
                return 1

        return res
