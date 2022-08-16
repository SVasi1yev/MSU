from time import strftime
from pyfiglet import figlet_format
from locale import setlocale, LC_ALL
import sys


def date(format='%Y %d %b, %A', font='graceful'):
    return figlet_format(strftime(format), font=font)


if __name__ == '__main__':
    setlocale(LC_ALL, ('ru_RU', 'UTF-8'))
    args = sys.argv
    if len(args) == 1:
        print(date())
    elif len(args) == 2:
        print(date(args[1]))
    else:
        print(date(args[1], args[2]))
