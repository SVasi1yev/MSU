from tqdm import tqdm

s = set()
lines = []
with open('Homework_BnopnyAnna.txt') as f:
    for i in range(75):
        lines.append(f.readline().strip())

    full = ' '.join(lines).split()
    for i in range(len(full)):
        for j in range(i + 1, len(full) + 1):
            s.add(tuple(full[i: j]))

    # (max(s, key=lambda x: len(x)))

    for line in tqdm(f):
        t = set()
        m = len(max(s, key=lambda x: len(x)))
        del lines[0]
        lines.append(line.strip())
        full = ' '.join(lines).split()
        for i in range(len(full)):
            for j in range(i + 1, i + m + 1):
                t.add(tuple(full[i: j]))

        s = s.intersection(t)

    print(s)

# {('не',), ('в',), ('что',), ('и',)}

