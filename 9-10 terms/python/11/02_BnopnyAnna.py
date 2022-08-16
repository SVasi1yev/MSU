import sys


class MyExp1(Exception):
    pass


s = {'не', 'что', 'и', 'в'}
codings = 'cp1026 cp1140 cp1256 cp273 cp437 cp500 cp775 cp850 cp852 cp855 cp857 cp860 cp861 cp862 cp863 cp865 cp866 gb18030 hp_roman8 iso8859_10 iso8859_11 iso8859_13 iso8859_14 iso8859_15 iso8859_16 iso8859_2 iso8859_4 iso8859_5 iso8859_9 koi8_r mac_cyrillic mac_greek mac_latin2 mac_roman utf_8'
codings = codings.split()


codings_pairs = {}
for cd in codings:
    codings_pairs[cd] = []
    for ce in codings:
        if ce == cd:
            continue
        try:
            for e in s:
                e_new = e.encode('utf-8')
                e_new = e_new.decode(ce)
                e_new = e_new.encode(cd)
                e_new = e_new.decode(cd)
                e_new = e_new.encode(ce)
        except UnicodeError as e:
            continue
        else:
            codings_pairs[cd].append(ce)

full = sys.stdin.buffer.read()

try:
    for cd1 in codings_pairs:
        for ce1 in codings_pairs[cd1]:
            try:
                full1 = full.decode(cd1)
                full1 = full1.encode(ce1)
            except UnicodeError:
                continue
            try:
                t = full1.decode()
                if len(set(t.split()).intersection(s)) == 4:
                    print(t.split('\n')[0])
                    raise MyExp1
            except UnicodeError:
                pass

            for cd2 in codings_pairs:
                for ce2 in codings_pairs[cd2]:
                    try:
                        full2 = full1.decode(cd2)
                        full2 = full2.encode(ce2)
                    except UnicodeError:
                        continue
                    try:
                        t = full2.decode()
                        if len(set(t.split()).intersection(s)) == 4:
                            print(t.split('\n')[0])
                            raise MyExp1
                    except UnicodeError:
                        pass

                    for cd3 in codings_pairs:
                        for ce3 in codings_pairs[cd3]:
                            try:
                                full3 = full2.decode(cd3)
                                full3 = full3.encode(ce3)
                            except UnicodeError:
                                continue
                            try:
                                t = full3.decode()
                                if len(set(t.split()).intersection(s)) == 4:
                                    print(t.split('\n')[0])
                                    raise MyExp1
                            except UnicodeError:
                                pass

except MyExp1:
    pass
else:
    print('FAIL')
