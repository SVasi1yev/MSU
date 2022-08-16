import textdistance
import random


def bullscows(guess: str, secter: str) -> (int, int):
    return textdistance.hamming.similarity(guess, secter), \
           textdistance.bag.similarity(guess, secter)


def gameplay(ask: callable, inform: callable, words: list) -> int:
    secter = random.choice(words)
    tries = 0
    bulls = -1
    while bulls != len(secter):
        tries += 1
        guess = ask("Введите слово: ")
        print(guess)
        bulls, cows = bullscows(guess, secter)
        inform("Быки: {}, Коровы: {}", bulls, cows)
    return tries
