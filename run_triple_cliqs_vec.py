import sys
from collections import Counter, namedtuple, defaultdict, OrderedDict
import itertools
import random
import cliqs.depgraph as depgraph
import cliqs.corpora
import numpy as np
import pandas as pd
import os

from generate_lemma_embeddings import get_vectors
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract SVO triples from corpora

flat = itertools.chain.from_iterable

VEC_DIM = 300
#SUBJ = torch.empty(64).to('cuda')
#nn.init.normal_(SUBJ)
#NOTSUBJ = torch.empty(64).to('cuda')
#nn.init.normal_(NOTSUBJ)
UNK = np.random.rand(VEC_DIM)

def aggregate_by_key(f, xs):
    d = {}
    for x in xs:
        key = f(x)
        if key in d:
            d[key].append(x)
        else:
            d[key] = [x]
    return d


SVOTriple = namedtuple(
    'Triple', "verb_index verb_word verb_lemma subj_index subj_word subj_lemma subj_pos obj_index obj_word obj_lemma obj_pos order sentence".split())

VectorTriple = namedtuple(
    'VectorTriple', "SVOTriple subj_vector obj_vector verb_vector random_so_vectors s_first label".split())

def extract_triples_from_sentence(s):
    for verb in s.nodes():
        if s.node[verb].get('pos') == 'VERB':
            deps = list(depgraph.dependents_of(s, verb))
            deps_by_type = aggregate_by_key(
                lambda d: s.edge[verb][d]['deptype'], deps)
            if ('nsubj' in deps_by_type
                    and 'obj' in deps_by_type
                    and len(deps_by_type['nsubj']) == 1
                    and len(deps_by_type['obj']) == 1):
                subj = deps_by_type['nsubj'][0]
                subj_word = s.node[subj]['word']
                subj_lemma = s.node[subj]['lemma']
                subj_pos = s.node[subj]['pos']
                obj = deps_by_type['obj'][0]
                obj_word = s.node[obj]['word']
                obj_lemma = s.node[obj]['lemma']
                obj_pos = s.node[obj]['pos']
                ordered = sorted([verb, subj, obj])
                order = ""
                for word in ordered:
                    if word == verb:
                        order += 'v'
                    elif word == subj:
                        order += 's'
                    elif word == obj:
                        order += 'o'
                verb_word = s.node[verb]['word']
                verb_lemma = s.node[verb]['lemma']
                yield SVOTriple(
                    verb,
                    verb_word.lower(),
                    verb_lemma.lower(),
                    subj,
                    subj_word.lower(),
                    subj_lemma.lower(),
                    subj_pos,
                    obj,
                    obj_word.lower(),
                    obj_lemma.lower(),
                    obj_pos,
                    order,
                    s,
                )

def extract_triples_from_corpus(corpus, **kwds):
    return list(flat(map(extract_triples_from_sentence, corpus.sentences(**kwds))))


class VectorEncoding:
    def __init__(self, triples, vectors, word_or_lemma="word"):
        self.vectors = vectors
        self.triples = triples
        self.word_or_lemma = word_or_lemma
        self.subj_first = SUBJ
        self.not_subj_first = NOTSUBJ

    def get_from_vectors(self, w):
        if w in self.vectors:
            return self.vectors[w]
        else:
            return UNK

    def get_values(self, include_order=True):
        """ Return list of vectors, labels (subj/obj), index with sent num, lemma dict."""
        vector_triples = []
        for sent_num, triple in enumerate(self.triples):
            s_first = triple.order in {'svo', 'sov', 'vso'}
            layer = []
            if self.word_or_lemma == "word":
                subj_vector = self.get_from_vectors(triple.subj_word)
                obj_vector = self.get_from_vectors(triple.obj_word)
                verb_vector = self.get_from_vectors(triple.verb_word)
            else:
                subj_vector = self.get_from_vectors(triple.subj_lemma)
                obj_vector = self.get_from_vectors(triple.obj_lemma)
                verb_vector = self.get_from_vectors(triple.verb_lemma)
            if include_order:
                if s_first:
                    order_vec = self.subj_first
                else:
                    order_vec = self.not_subj_first
            else:
                order_vec = self.subj_first
            if random.random() > .5:
                label = 1
                random_so_vectors = [subj_vector, obj_vector]
            else:
                label = 0
                random_so_vectors = [obj_vector, subj_vector]
            vector_triples += [VectorTriple(triple,
                                        subj_vector,
                                        obj_vector,
                                        verb_vector,
                                        random_so_vectors,
                                        order_vec,
                                        label)]
        return vector_triples

class _classifier(nn.Module):
    def __init__(self, num_in, num_out, num_hidden_1=64, num_hidden_2=16):
        print(num_hidden_1, num_hidden_2, file=sys.stderr)
        super(_classifier, self).__init__()
        self.hidden = nn.Linear(num_in, num_hidden_1)
        self.hidden_relu = nn.ReLU()
        self.hidden2 = nn.Linear(num_hidden_1 * 3, num_hidden_2)
        self.hidden3 = nn.ReLU()
        self.hidden4 = nn.Linear(num_hidden_2, num_out)

    def forward(self, input1, input2, order_vec):
        x1 = self.hidden(input1)
        x2 = self.hidden(input2)
        x1 = self.hidden_relu(x1)
        x2 = self.hidden_relu(x2)
        h = self.hidden2(torch.cat([x1.squeeze(), x2.squeeze(), order_vec.squeeze()], dim=0))
        h = self.hidden3(h)
        h = self.hidden4(h)
        return h


class TrainDevTestData:

    def __init__(self, triples_train, triples_dev, triples_test, encoder, vectors, word_or_lemma, include_order=False):
        self.triples_train = triples_train
        self._train = encoder(self.triples_train, vectors, word_or_lemma)
        self.train_triple_vectors = self._train.get_values(include_order=include_order)

        self.triples_dev = triples_dev
        self._dev = encoder(self.triples_dev, vectors, word_or_lemma)
        self.dev_triple_vectors = self._dev.get_values(include_order=include_order)

        self.triples_test = triples_test
        self._test = encoder(self.triples_test, vectors, word_or_lemma)
        self.test_triple_vectors = self._test.get_values(include_order=include_order)


class TripleClassifier:
    def __init__(self, data, num_epochs=200, **kwds):
        self.data = data
        self.do_train(num_epochs, **kwds)

    def do_train(self, num_epochs=200, verbose=True, early_stopping=False, num_hidden_1=64, num_hidden_2=16, **kwds):
        nlabel = 2
        train_triples = self.data.train_triple_vectors
        num_inputs = train_triples[0].subj_vector.shape[0]
        self.classifier = _classifier(num_inputs, nlabel, num_hidden_1=num_hidden_1, num_hidden_2=num_hidden_2).cuda()

        optimizer = optim.Adam(self.classifier.parameters(), **kwds)
        criterion = nn.CrossEntropyLoss()
        prev_dev_loss = float('inf')
        dev_loss = prev_dev_loss

        for epoch in range(num_epochs):
            losses = []
            for i, sample in enumerate(train_triples):
                inputv1 = torch.tensor(sample.random_so_vectors[0]).unsqueeze(0).to("cuda")
                inputv2 = torch.tensor(sample.random_so_vectors[1]).unsqueeze(0).to("cuda")
                inputv3 = sample.s_first.clone().detach().unsqueeze(0).to("cuda")
                labelsv = torch.tensor([sample.label])
                output = self.classifier(inputv1.float(), inputv2.float(), inputv3.float())
                loss = criterion(output.unsqueeze(0),  labelsv.to("cuda"))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            #   if verbose:
            #        losses.append(loss.data.mean())
            #print('[%d/%d] Train Loss: %.3f' %
            #          (epoch+1, num_epochs, np.mean(losses)), file=sys.stderr)

    def predict_test_set(self):
        # work on subj/obj dyads
        correct = []
        with torch.no_grad():
            for sample in self.data.test_triple_vectors:
                inputv1 = torch.tensor(
                    sample.random_so_vectors[0]).unsqueeze(0).to("cuda")
                inputv2 = torch.tensor(
                    sample.random_so_vectors[1]).unsqueeze(0).to("cuda")
                inputv3 = sample.s_first.clone().detach().unsqueeze(0).to("cuda")
                output = self.classifier(inputv1.float(), inputv2.float(), inputv3.float())
                correct.append(int((output[0] < output[1]) ==  sample.label))
        return correct

def get_data(lang, vectors, word_or_lemma, include_order):
    d = {}
    d[lang] = TrainDevTestData(train_triples[lang],
                                   dev_triples[lang],
                                   test_triples[lang],
                                   VectorEncoding,
                                   vectors, 
                                   word_or_lemma,
                                   include_order=include_order)
    return d        


def accuracy(data, **hypers):
    classifier = TripleClassifier(data, **hypers)
    token_acc = np.mean(classifier.predict_test_set())
    return (token_acc)

if __name__ == "__main__":
    
    curlang = sys.argv[1]
    df = pd.read_csv("iso_codes.txt").set_index("dep")

    word_or_lemma = sys.argv[2]
    assert word_or_lemma in ["word", "lemma"]
    if word_or_lemma == "word":
        code = "wiki.{}.align.vec".format(df.loc[curlang]["wiki"])
        path = os.path.join("/home/canjo/data/wikivecs/", code)
    else:
        code = df.loc[curlang]["iso"]
        path = os.path.join("/home/canjo/data/wikivec_lemmas/", code)

    hypers = OrderedDict([
        ('num_hidden_1', int(sys.argv[3])),
        ('num_hidden_2', int(sys.argv[4])),
        ('lr', float(sys.argv[5])),
    ])    

    hypers_str = "_".join(map(str, hypers.values()))

    global SUBJ
    global NOTSUBJ
    SUBJ = torch.empty(hypers['num_hidden_1']).to('cuda')
    nn.init.normal_(SUBJ)
    NOTSUBJ = torch.empty(hypers['num_hidden_1']).to('cuda')
    nn.init.normal_(NOTSUBJ)            

    vectors = {curlang: get_vectors(path)}

    all_corpora = (
        set(cliqs.corpora.all_ud_dev_corpora)
        & set(cliqs.corpora.all_ud_test_corpora)
        & set(cliqs.corpora.all_ud_train_corpora)
    )


    #print(curlang, all_corpora, "ALL")

    assert curlang in all_corpora

    # Load Data
    test_triples = {
        curlang: extract_triples_from_corpus(cliqs.corpora.all_ud_test_corpora[curlang])
    }    

    def take(xs, n):
        return itertools.islice(xs, None, n)


    CUTOFF = 200
    SIZE_LIMIT = CUTOFF * 8
    good_langs = {lang for lang in test_triples if len(
        test_triples[lang]) >= CUTOFF}
    print(good_langs, file=sys.stderr)

    assert curlang in good_langs

    train_triples = {
        curlang: list(take(extract_triples_from_corpus(
            cliqs.corpora.all_ud_train_corpora[curlang]), SIZE_LIMIT))
    }

    dev_triples = {
        curlang: extract_triples_from_corpus(cliqs.corpora.all_ud_dev_corpora[curlang])
    }

    d_with_order = get_data(curlang, vectors[curlang], word_or_lemma, True)
    d_without_order = get_data(curlang, vectors[curlang], word_or_lemma, False)

    order_data = {
        (curlang, "order"): accuracy(d_with_order[curlang], **hypers)
    }

    no_order_data = {
        (curlang, "no_order"): accuracy(d_without_order[curlang], **hypers) 
    }

    # output format: {('UD_Bulgarian-BTB', 'order'): 0.8679775280898876}

    print(order_data, file=sys.stderr)
    print(no_order_data, file=sys.stderr)

    for data in [order_data, no_order_data]:
        for key, value in data.items():
            print(",".join(list(map(str, key)) + list(map(str, hypers.values())) + [str(value)]))

    #pickle.dump(order_data, open("data_vecs/order/" + curlang + "_" + word_or_lemma + "_" + hypers_str, "wb"))

    #pickle.dump(no_order_data, open("data_vecs/no_order/" + curlang + "_" + word_or_lemma + "_" + hypers_str, "wb"))
