from fractions import Fraction
import itertools


class Sausage:
    def __init__(self, word='pork!', length='1'):
        length = Fraction(length)
        if length <= 0:
            self.num = 0
            self.length = Fraction(0, 12)
            self.str_length = Fraction(0)
        else:
            self.num = int((length._numerator / length._denominator) * 12)
            self.length = Fraction(self.num, 12)
            self.str_length = length
        self.word = word

    def __str__(self):
        if self.num == 0:
            res = '/|\n||\n||\n||\n\\|'
            return res

        res = ''
        d, r = divmod(self.num, 12)
        for i in range(d):
             res += '/' + 12*'-' + '\\'
        if r > 0:
            res += '/' + r * '-' + '|'
        res += '\n'
        for i in range(3):
            for j in range(d):
                res += '|'
                it = itertools.cycle(self.word)
                for k in range(12):
                    res += next(it)
                res += '|'
            if r > 0:
                res += '|'
                it = itertools.cycle(self.word)
                for j in range(r):
                    res += next(it)
                res += '|'
            res += '\n'
        for i in range(d):
            res += '\\' + 12*'-' + '/'
        if r > 0:
            res += '\\' + r * '-' + '|'

        return res

    def __add__(self, other):
        return Sausage(self.word, self.str_length + other.str_length)

    def __sub__(self, other):
        return Sausage(self.word, self.str_length - other.str_length)

    def __mul__(self, other):
        return Sausage(self.word, self.str_length * other)

    def __rmul__(self, other):
        return Sausage(self.word, self.str_length * other)

    def __truediv__(self, other):
        return Sausage(self.word, self.length / other)

    def __abs__(self):
        return self.str_length

    def __bool__(self):
        return abs(self) > 0

#
# a, b, c = Sausage(), Sausage("HAM", "5/6"), Sausage("SPAM.", 1.25)
# print(a, b, c, sep='\n')
# print(a + b + c, abs(a + b + c))
# print(b * 2, 4 * c / 5, sep="\n")
# d, e = 2 * b + a / 3 - 25 * c / 16, a - c
# print(d, not d, abs(d))
# print(e, not e, abs(e))
