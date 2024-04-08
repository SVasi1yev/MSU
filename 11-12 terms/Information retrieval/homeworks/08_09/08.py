import numpy as np
import pandas as pd
import pymystem3
import razdel
import json
from sklearn.metrics.pairwise import cosine_similarity

from collections import defaultdict

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

def get_vocab(docs):
    lemmas = set()
    for d in docs:
        for lemma in d['lemmas']:
            lemmas.add(lemma)
            
    lemma2id = {lemma: i for i, lemma in enumerate(lemmas)}
    
    return lemma2id

def get_tf_idf(lemma2id, docs):
    vocab_len = len(lemma2id)
    
    df = np.zeros(vocab_len)
    tfs = []
    for d in docs:
        tfs.append(np.zeros(vocab_len))
        for lemma in set(d['lemmas']):
            if lemma not in lemma2id:
                continue
            df[lemma2id[lemma]] += 1
        for lemma in d['lemmas']:
            if lemma not in lemma2id:
                continue
            tfs[-1][lemma2id[lemma]] += 1
    
    idf = np.log(len(docs)/df)
    tfs = np.vstack(tfs)
    
    return tfs, idf

def model_01(tfs, idf):
    t = tfs * idf
    t = t / np.linalg.norm(t, axis=1).reshape(-1, 1)
    return t

def model_02(tfs, idf):
    t = ((0.5 + 0.5 * (tfs / tfs.max(axis=1).reshape(-1, 1))) * (tfs > 0) )* idf
    t = t / np.linalg.norm(t, axis=1).reshape(-1, 1)
    return t

def prop_model(docs_tfs, qs_tfs, alpha):
    glob_props = docs_tfs.sum(axis=0)
    glob_props /= glob_props.sum()
    scores = []
    loc_props = docs_tfs / docs_tfs.sum(axis=1).reshape(-1, 1)
    for i in range(qs_tfs.shape[0]):
        scores.append((
            np.log(0.000001 +((1-alpha)*glob_props + alpha*loc_props)) *
            (qs_tfs[i] > 0).astype(int)
         ).sum(axis=1))
    scores = np.array(scores)
    return scores.T

with open('rated_docs.json') as f:
    docs = json.load(f)
for i in range(len(docs)):
    r = int(docs[i]['rating'])
    rating = [0, 0, 0]
    rating[docs[i]['q_idx']] = r
    docs[i]['rating'] = rating

scores = []
for d in docs:
    scores.append(d['rating'])
scores = np.array(scores)

lemma2id = get_vocab(docs)
docs_tfs, idf = get_tf_idf(lemma2id, docs)
doc_m_1 = model_01(docs_tfs, idf)
doc_m_2 = model_02(docs_tfs, idf)

qs = get_docs('queries.txt')
qs_tfs, _ = get_tf_idf(lemma2id, qs)
qs_m_1 = model_01(qs_tfs, idf)
qs_m_2 = model_02(qs_tfs, idf)

scores_m_1 = cosine_similarity(doc_m_1, qs_m_1)
scores_m_2 = cosine_similarity(doc_m_2, qs_m_2)
scores_prop_05 = prop_model(docs_tfs, qs_tfs, 0.5)
scores_prop_09 = prop_model(docs_tfs, qs_tfs, 0.9)

def dcg(scores, model_scores, k=10):
    return (scores[model_scores.argsort()[::-1]][:k] / np.log2(np.arange(k) + 2)).sum()

def ndcg(scores, model_scores, k=10):
    dcg_ = dcg(scores, model_scores, k)
    idcg_ = dcg(scores, scores, k)
    return dcg_ / idcg_

def make_report(docs, qs, scores, model_scores, top_k=10, dcg_k=10):
    for i in range(len(qs)):
        print(f'Query {i+1}: {qs[i]["init"]}\n')
        print(f'NDCG: {ndcg(scores[:, i], model_scores[:, i], dcg_k)}')
        print(f'Top {top_k}:\n')
        for k, idx in enumerate(model_scores[:, i].argsort()[::-1][:top_k]):
            print(f'{k+1} (true: {round(scores[:, i][idx], 5)}) (model: {round(model_scores[:, i][idx], 5)}): {docs[idx]["init"]}\n')
        print(20*'#' + '\n')

make_report(docs, qs, scores, scores_m_1)
make_report(docs, qs, scores, scores_m_2)
make_report(docs, qs, scores, scores_prop_05)
make_report(docs, qs, scores, scores_prop_09)
