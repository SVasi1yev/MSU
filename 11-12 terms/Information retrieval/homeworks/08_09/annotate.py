import numpy as np
import pandas as pd
import pymystem3
import razdel
from sklearn.metrics.pairwise import cosine_similarity

from collections import defaultdict
import json

import re

SPLIT_RGX = re.compile(u'[A-Za-zА-Яа-я0-9]+', re.UNICODE)

def split(string):
    words = re.findall(SPLIT_RGX, string)
    return words

stem = pymystem3.Mystem()

def get_docs(file_name):
#     init_docs = []
    docs = []
    with open(file_name, 'r') as f:
        for i, line in enumerate(f):
            t = razdel.sentenize(line)
            for e in t:
                d = {}
                d['q_idx'] = i
                d['init'] = e.text.strip()
#                 init_docs.append((i, e.text.strip()))
                lemmas = stem.lemmatize(e.text)
                doc = ''.join(lemmas)
                words = split(doc)
                d['lemmas'] = words
#                 docs.append((i, words))
                docs.append(d)
                
    return docs

docs = get_docs('docs.txt')

for i, d in enumerate(docs):
    if 'rating' in d:
        continue
    print(d['init'])
    r = input(f'{i}: Rating: ')
    docs[i]['rating'] = r
    with open('rated_docs.json', 'w') as f:
        json.dump(docs, f)
