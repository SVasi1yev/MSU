class Triangle:
    def __init__(self, a, b, c):
        self.str_sides = [float(a), float(b), float(c)]
        self.str = ':'.join(map(str, self.str_sides))
        self.sides = sorted(self.str_sides)
        self.not_empty = True
        for i in range(3):
            if self.sides[i] <= 0:
                self.not_empty = False
                break
            if sum(self.sides[:i]) + sum(self.sides[i+1:]) < self.sides[i]:
                self.not_empty = False
                break
        if self.not_empty:
            p = sum(self.sides) / 2
            self.s = (p*(p-self.sides[0])*(p-self.sides[1])*(p-self.sides[2])) ** 0.5
        else:
            self.s = 0

    def __bool__(self):
        return self.not_empty

    def __abs__(self):
        return self.s

    def __eq__(self, other):
        for i in range(3):
            if self.sides[i] != other.sides[i]:
                return False
        return True

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return abs(self) < abs(other)

    def __le__(self, other):
        return abs(self) <= abs(other)

    def __gt__(self, other):
        return abs(self) > abs(other)

    def __ge__(self, other):
        return abs(self) >= abs(other)

    def __str__(self):
        return self.str
