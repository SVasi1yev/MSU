import numpy as np
import pymystem3
import razdel
from sklearn.metrics.pairwise import cosine_similarity

import sys
import re


def split(string):
    SPLIT_RGX = re.compile(u'[A-Za-zА-Яа-я0-9]+', re.UNICODE)
    words = re.findall(SPLIT_RGX, string)
    return words


def get_docs(file_name):
    stem = pymystem3.Mystem()
    init_docs = []
    docs = []
    with open(file_name, 'r') as f:
        for line in f:
            t = razdel.sentenize(line)
            for e in t:
                init_docs.append(e.text.strip())
                lemmas = stem.lemmatize(e.text)
                doc = ''.join(lemmas)
                words = split(doc)
                docs.append(words)

    return init_docs, docs


def get_vocab(docs):
    lemmas = set()
    for d in docs:
        for lemma in d:
            lemmas.add(lemma)

    lemma2id = {lemma: i for i, lemma in enumerate(lemmas)}

    return lemma2id


def get_tf_idf(lemma2id, docs):
    vocab_len = len(lemma2id)

    df = np.zeros(vocab_len)
    tfs = []
    for d in docs:
        tfs.append(np.zeros(vocab_len))
        for lemma in set(d):
            if lemma not in lemma2id:
                continue
            df[lemma2id[lemma]] += 1
        for lemma in d:
            if lemma not in lemma2id:
                continue
            tfs[-1][lemma2id[lemma]] += 1

    idf = np.log(len(docs) / df)
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


def make_report(file_name, init_qs, init_docs, scores, top_k=10):
    init_qs = np.array(init_qs)
    init_docs = np.array(init_docs)
    with open(file_name, 'w') as f:
        for i in range(len(init_qs)):
            f.write(f'Query {i+1}: {init_qs[i]}\n')
            f.write(f'Top {top_k}:\n')
            for k, idx in enumerate(scores[:, i].argsort()[::-1][:top_k]):
                f.write(f'{k+1} ({round(scores[:, i][idx], 5)}): {init_docs[idx]}\n')
            f.write(20*'#' + '\n')


if __name__ == '__main__':
    docs_file = sys.argv[1]
    qs_file = sys.argv[2]
    top_k = int(sys.argv[3])

    init_docs, docs = get_docs(docs_file)
    lemma2id = get_vocab(docs)
    docs_tfs, idf = get_tf_idf(lemma2id, docs)
    doc_m_1 = model_01(docs_tfs, idf)
    doc_m_2 = model_02(docs_tfs, idf)

    init_qs, qs = get_docs(qs_file)
    qs_tfs, _ = get_tf_idf(lemma2id, qs)
    qs_m_1 = model_01(qs_tfs, idf)
    qs_m_2 = model_02(qs_tfs, idf)

    scores_m_1 = cosine_similarity(doc_m_1, qs_m_1)
    scores_m_2 = cosine_similarity(doc_m_2, qs_m_2)

    make_report('report_m_1.txt', init_qs, init_docs, scores_m_1, top_k)
    make_report('report_m_2.txt', init_qs, init_docs, scores_m_2, top_k)
