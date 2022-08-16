import zipfile
import io
import sys

d = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9, 'a': 10, 'b': 11, 'c': 12, 'd': 13, 'e': 14, 'f': 15
}

arch = []
i = 0
cur = 0
in_ = ''.join(sys.stdin.read().split())
for c in in_:
    if i % 2 == 0:
        cur = d[c]
    else:
        cur = cur * 16 + d[c]
        arch.append(cur)
    i += 1

arch = bytes(arch)
zf = zipfile.ZipFile(io.BytesIO(arch), "r")
count = 0
size = 0
for f in zf.infolist():
    if not f.is_dir():
        count += 1
        size += f.file_size

print(count, size)
