""" Generate lemma embeddings

Input arguments: Two filenames
1. Word vectors, in word2vec format.
2. Lemmas file: tab-separated where column 1 = lemma and column 2 = word
(Unimorph files work as input unaltered)

Output to stdout:
Lemma vectors for each word from input file 1, in word2vec format
For words with no known lemma, output has a zeros vector.

"""
import io
import os
import sys
import gzip
import random
from collections import OrderedDict

import numpy as np

def myopen(filename, **kwds):
    if filename.endswith(".gz"):
        return gzip.open(filename, **kwds)
    else:
        return open(filename, **kwds)

def first(xs):
    return next(iter(xs))

def load_vectors(lines):
    lines = iter(lines)
    n, d = map(int, next(lines).rstrip().split(' '))
    data = OrderedDict()
    for line in lines:
        word, *numbers = line.rstrip().split(' ')
        assert len(numbers) == d
        if len(numbers) == d:
            data[word] = np.array(list(map(float, numbers)))
    print("vec length:", len(data),  file=sys.stderr)
    #assert len(data) == n
    return data

def load_lemmas(lines):
    lemma2word = {}
    word2lemma = {}
    for line in lines:
        if not line.strip():
            continue
        lemma, word, features = line.strip().split("\t")
        lemma = lemma.casefold()
        word = word.casefold()
        if word in word2lemma:
            word2lemma[word].append(lemma)
        else:
            word2lemma[word] = [lemma]
        if lemma in lemma2word:
            lemma2word[lemma].append(word)
        else:
            lemma2word[lemma] = [word]
    print("lemma length:", len(lemma2word), file=sys.stderr)
    print("word2lemma length:", len(word2lemma),  file=sys.stderr)

    return word2lemma, lemma2word

def lemma_averages(vectors, l2w):
    d = {}
    for lemma, words in l2w.items():
        vs = [vectors[word] for word in words if word in vectors]
        if len(vs) > 0:
            d[lemma] = np.mean(vs, axis=0)
    return d

def get_lemma(w2l, word):
    word = word.casefold()
    if word in w2l:
        return random.choice(w2l[word])
    else:
        return None

def make_lemma_vectors(vectors, w2l, l2w):
    d = OrderedDict()
    lemma_vectors = lemma_averages(vectors, l2w)
    dimension = len(first(lemma_vectors.values()))
    for w, v in vectors.items():
        lemma = get_lemma(w2l, w)
        if lemma is not None:
            d[w] = lemma_vectors[lemma]
        else:
            d[w] = np.zeros(dimension)
    return d

def print_word2vec(d):
    dimension = len(first(d.values()))
    print(str(len(d)) + " " + str(dimension), end="\n")
    for w, v in d.items():
        print(w, end=" ")
        number_strs = ["%.4f" % v_i for v_i in v]
        print(" ".join(number_strs), end="\n")

def get_vectors(x):
    with myopen(x, mode='rt') as infile:
        vectors = load_vectors(infile)
    return vectors
    
def main(vec_filename, lemmas_filename):
    print("Loading vectors from: %s" % vec_filename, file=sys.stderr)
    with myopen(vec_filename, mode='rt') as infile:
        vectors = load_vectors(infile)
    print("Loading lemmas from: %s" % lemmas_filename, file=sys.stderr)
    with myopen(lemmas_filename, mode='rt') as infile:
        w2l, l2w = load_lemmas(infile)
    print("Calculating lemma embeddings...", file=sys.stderr)
    result = make_lemma_vectors(vectors, w2l, l2w)
    print_word2vec(result)
    return 0
    
if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
        
