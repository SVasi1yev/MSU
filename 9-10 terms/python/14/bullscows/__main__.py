import sys
import os
import urllib.request

from .bullscows import gameplay


def ask(prompt: str, valid: list = None) -> str:
    print('111')
    guess = input(prompt)
    print('222')
    if valid is None:
        return guess
    while guess not in valid:
        guess = input(prompt)
    return guess


def inform(format_string: str, bulls: int, cows: int) -> None:
    print(format_string.format(bulls, cows))


if __name__ == '__main__':
    args = sys.argv
    length = 5
    if len(args) > 2:
        length = int(args[2])

    path = args[1]
    words = []
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == length:
                    words.append(line)
    else:
        t = urllib.request.urlopen(path).read().decode().split()
        for word in t:
            if len(word) == length:
                words.append(word)

    tries = gameplay(ask, inform, words)
    print('Количество попыток: ', tries)
