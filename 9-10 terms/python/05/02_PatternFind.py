string = input()
pattern = input()

for i in range(len(string) - len(pattern) + 1):
    for j in range(len(pattern)):

        if pattern[j] != '@':
            if string[i + j] != pattern[j]:
                break
    else:
        print(i)
        break
else:
    print(-1)