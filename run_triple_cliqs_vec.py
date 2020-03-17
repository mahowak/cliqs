import sys
from collections import Counter, namedtuple, defaultdict
import itertools
import random
import cliqs.depgraph as depgraph
import cliqs.corpora
import numpy as np
import pandas as pd

from transformers import BertTokenizer, BertModel, BertForMaskedLM
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


BERT_DIM = 768

# Load pre-trained model tokenizer (vocabulary)
# Crucially, do not do basic tokenization; PTB is tokenized. Just do wordpiece tokenization.
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

bert_model = BertModel.from_pretrained('bert-base-multilingual-cased',
                                       output_hidden_states=True)
bert_model.to('cuda')
bert_model.eval()


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

# Calculate how predictable subjecthood is from wordforms, by the simplest method

def subj_prob_mle(triples):
    subj_word_counts = Counter(triple.subj_word for triple in triples)
    obj_word_counts = Counter(triple.obj_word for triple in triples)
    word_counts = subj_word_counts + obj_word_counts
    p_subj_given_word = {word: subj_word_counts[word]/word_count for word, word_count in word_counts.items()}
    return p_subj_given_word

def mle_model(probs):
    def score(triple):
        return probs[triple.subj_word]
    return score
    

# How much uncertainty is there about subjecthood in the triple?
def subj_identification_accuracy(triples, model):
    num_correct = 0
    n = 0
    for triple in triples:
        n += 1
        if model(triple) > .5:
            num_correct += 1
    if n == 0:
        return None
    else:
        return num_correct/n


def get_words(s):
    return [s.node[n]['word'] for n in sorted(s.nodes())[1:]]

SUBJ = torch.empty(64).to('cuda')
nn.init.normal_(SUBJ)
NOTSUBJ = torch.empty(64).to('cuda')
nn.init.normal_(NOTSUBJ)

class ContextualBertEncoding:
    def __init__(self, triples, tokenizer=bert_tokenizer, model=bert_model):
        """Pass a BERT tokenizer, model, and triples (a list of SVOTriple object)"""
        self.tokenizer = tokenizer
        self.model = model
        self.triples = triples
        self.indexed_tokens, self.orig_tok_d = self.get_indexed_tokens(
            self.triples)
        self.outputs = self.get_outputs()
        self.subj_first = SUBJ
        self.not_subj_first = NOTSUBJ
        
    def get_indexed_tokens(self, triples):
        """Returned the indexed BERT tokens and the dictionary for interpreting.
        orig_tok_d is a list of maps, one map per sentence. It maps from orig tokens 
        to BERT tokens."""
        orig_tokens = [get_words(t.sentence) for t in triples]
        bt = []
        orig_tok_d = []
        for i, ot in enumerate(orig_tokens):
            ### Output
            bert_tokens = []

            # Token map will be an int -> int mapping between the `orig_tokens` index and
            # the `bert_tokens` index.
            orig_to_tok_map = [0]

            bert_tokens.append("[CLS]")
            for orig_token in ot:
                orig_to_tok_map.append(len(bert_tokens))
                bert_tokens.extend(self.tokenizer.tokenize(orig_token))
            orig_to_tok_map.append(len(bert_tokens))
            bert_tokens = bert_tokens[:511]
            bert_tokens.append("[SEP]")
            bt += [bert_tokens]
            orig_tok_d += [orig_to_tok_map]

        indexed_tokens = [self.tokenizer.convert_tokens_to_ids(b) for b in bt]
        self.bert_tokens = bt
        return indexed_tokens, orig_tok_d

    def get_train_values(self, avg_vec=False, include_order=True):
        """ Return list of BERT encodings, labels (subj/obj), index with sent num, lemma dict."""
        train = []
        labels = []
        idx = []
        lemma_dict = {"subj": defaultdict(list), "obj": defaultdict(list)}
        word_dict = {"subj": defaultdict(list), "obj": defaultdict(list)}

        for sent_num, output in enumerate(self.outputs):
            embed_layer = output[-1]
            triple = self.triples[sent_num]
            s_first = triple.order in {'svo', 'sov', 'vso'}
            subj_info = (self.orig_tok_d[sent_num][triple.subj_index],
                         self.orig_tok_d[sent_num][triple.subj_index + 1],
                         triple.subj_word,
                         triple.subj_lemma,
                         s_first,  # Whether this one is the first of the two
                         "subj")
            obj_info = (self.orig_tok_d[sent_num][triple.obj_index],
                        self.orig_tok_d[sent_num][triple.obj_index + 1],
                        triple.obj_word,
                        triple.obj_lemma,
                        not s_first,  # Whether this one is the first of the two
                        "obj")
            layer = []
            for (i, j, word, lemma, order, label) in [subj_info, obj_info]:
                if avg_vec:
                    l = (embed_layer.squeeze()[i:j]).max(0)[0].to("cuda")
                    layer += [l]
                else:
                    l = embed_layer.squeeze()[i].to("cuda")
                    layer += [l]
                if include_order:
                    if s_first:
                        order_vec = self.subj_first
                    else:
                        order_vec = self.not_subj_first
                else:
                    order_vec = self.subj_first
                lemma_dict[label][lemma] += [l]
                word_dict[label][word] += [l]

            layer += [order_vec]
            if random.random() > .5:
                labels += [1]
            else:
                labels += [0]
                layer[0], layer[1] = layer[1], layer[0]
            train += [layer]
            idx += [(sent_num, i, j)]
        return train, labels, idx, word_dict, lemma_dict

    def get_outputs(self):
        """ Do the BERT encodings."""
        outputs = []
        with torch.no_grad():
            for input_id in self.indexed_tokens:
                encoded_layers, _, hidden_layers = self.model(
                    torch.tensor(input_id).unsqueeze(0).to("cuda"))
                outputs += [encoded_layers]
        return outputs


def one_hot(K, v):
    a = torch.zeros(len(K), v)
    for i, k in enumerate(K):
        a[i, k] = 1
    return a


class OneHotEncoding(ContextualBertEncoding):

    def get_outputs(self):
        """ Do the BERT encodings."""
        V = self.tokenizer.vocab_size + len(self.tokenizer.special_tokens_map)
        outputs = []
        for input_id in self.indexed_tokens:
            layers = one_hot(input_id, V).unsqueeze(0)
            outputs += [layers]
        return outputs


class AveragedWordEncoding(ContextualBertEncoding):
    pass


class AveragedLemmaEncoding(ContextualBertEncoding):
    pass

        
class _classifier2(nn.Module):
    def __init__(self, num_in, num_out):
        super(_classifier2, self).__init__()
        self.hidden = nn.Linear(num_in, 64)
        self.hidden_relu = nn.ReLU()
        self.hidden2 = nn.Linear(64 * 3, 16)
        self.hidden3 = nn.ReLU()
        self.hidden4 = nn.Linear(16, num_out)

    def forward(self, input1, input2, order_vec):
        x1 = self.hidden(input1)
        x2 = self.hidden(input2)
        x1 = self.hidden_relu(x1)
        x2 = self.hidden_relu(x2)
        h = self.hidden2(torch.cat([x1.squeeze(), x2.squeeze(), order_vec.squeeze()], dim=0))
        h = self.hidden3(h)
        h = self.hidden4(h)
        return h

class _classifier(nn.Module):
    def __init__(self, num_in, num_out):
        super(_classifier, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(num_in, 64),
            nn.ReLU(),
            nn.Linear(64, num_out),
            nn.Dropout(.1)
        )

    def forward(self, input):
        return self.main(input)


class TrainDevTestData:

    def __init__(self, triples_train, triples_dev, triples_test, encoder, include_order=False, avg_vec=True):
        self.triples_train = triples_train
        self.fin_train = encoder(self.triples_train)
        self.train, self.train_labels, self.train_idx, self.train_lemmas, self.train_words = self.fin_train.get_train_values(
            avg_vec=avg_vec, include_order=include_order)

        self.triples_dev = triples_dev
        self.fin_dev = encoder(self.triples_dev)
        self.dev, self.dev_labels, self.dev_idx, self.dev_lemmas, self.dev_words = self.fin_dev.get_train_values(
            avg_vec=avg_vec, include_order=include_order)

        self.triples_test = triples_test
        self.fin_test = encoder(self.triples_test)
        self.test, self.test_labels, self.test_idx, self.test_lemmas, self.test_words = self.fin_test.get_train_values(
            avg_vec=avg_vec, include_order=include_order)


class TripleClassifier:
    def __init__(self, data, num_epochs=200, **kwds):
        self.data = data
        self.do_train(num_epochs, **kwds)

    def do_train(self, num_epochs=200, verbose=False, early_stopping=False, **kwds):

        nlabel = 2
        num_inputs = self.data.train[0][0].shape[0]
        self.classifier = _classifier2(num_inputs, nlabel).cuda()

        optimizer = optim.Adam(self.classifier.parameters(), **kwds)
        criterion = nn.CrossEntropyLoss()
        prev_dev_loss = float('inf')
        dev_loss = prev_dev_loss
        for epoch in range(num_epochs):
            losses = []
            for i, sample in enumerate(self.data.train):
                inputv1 = torch.tensor(sample[0]).unsqueeze(0).to("cuda")
                inputv2 = torch.tensor(sample[1]).unsqueeze(0).to("cuda")
                inputv3 = torch.tensor(sample[2]).unsqueeze(0).to("cuda")
                labelsv = torch.tensor([self.data.train_labels[i]])
                output = self.classifier(inputv1, inputv2, inputv3)
                loss = criterion(output.unsqueeze(0),  labelsv.to("cuda"))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if verbose:
                    losses.append(loss.data.mean())
            if verbose:
                print('[%d/%d] Train Loss: %.3f' %
                      (epoch+1, num_epochs, np.mean(losses)), file=sys.stderr)

    def predict_test_set(self):
        # work on subj/obj dyads
        correct = []
        with torch.no_grad():
            for i in range(len(self.data.test)):
                inputv1 = torch.tensor(self.data.test[i][0])
                inputv2 = torch.tensor(self.data.test[i][1])
                inputv3 = torch.tensor(self.data.test[i][2])
                output = self.classifier(inputv1, inputv2, inputv3)
                correct.append(int((output[0] < output[1]) ==  self.data.test_labels[i]))
        return correct

    def do_test_using_lemma_embeddings(self, word_or_lemma="word"):
        """Use the training set lemmas, average the embeddings across all
        subj and obj to get a new embedding. Ignore test set embedding."""
        correct = []

        with torch.no_grad():
            for triple in self.data.triples_test:
                if word_or_lemma == "word":
                    arg1 = triple.subj_word
                    arg2 = triple.obj_word
                    d = self.data.train_words

                elif word_or_lemma == "lemma":
                    arg1 = triple.subj_lemma
                    arg2 = triple.obj_lemma
                    d = self.data.train_lemmas

                arg1_len = len(d["subj"][arg1] + d["obj"][arg1])
                arg2_len = len(d["subj"][arg2] + d["obj"][arg2])
                if (arg1_len + arg2_len) == 0:
                    correct += [None]
                    continue

                if arg1_len > 0:
                    subj_inputv = torch.stack(
                        d["subj"][arg1] + d["obj"][arg1]).mean(0)
                else:
                    subj_inputv = torch.empty(BERT_DIM)
                    nn.init.normal_(subj_inputv)

                if arg2_len > 0:
                    obj_inputv = torch.stack(
                        d["subj"][arg2] + d["obj"][arg2]).mean(0)
                else:
                    obj_inputv = torch.empty(BERT_DIM)
                    nn.init.normal_(obj_inputv)
                
                if triple.order in ["svo", "sov", "vso"]:
                    order_vec = SUBJ
                else:
                    order_vec = NOTSUBJ
                
                # DO CLASSIFICATION
                output = self.classifier(subj_inputv.to('cuda'), obj_inputv.to('cuda'), order_vec.to('cuda'))
                correct.append(int(output[0] < output[1]))

        return correct


def get_data(langs, include_order):
    d = {}
    for lang in langs:
        d[lang] = TrainDevTestData(train_triples[lang], dev_triples[lang],
                                   test_triples[lang], ContextualBertEncoding, include_order=include_order)
    return d        


def accuracy(data):
    classifier = TripleClassifier(data)
    token_acc = np.mean(classifier.predict_test_set())
    _word_acc = classifier.do_test_using_lemma_embeddings("word")
    word_acc = np.mean([float(i) for i in _word_acc if i is not None])
    _lemma_acc = classifier.do_test_using_lemma_embeddings("lemma")
    lemma_acc = np.mean([float(i) for i in _lemma_acc if i is not None])
    return (token_acc, word_acc, lemma_acc, 
            len(_word_acc), sum([i == None for i in _word_acc]), 
            len(_lemma_acc), sum([i == None for i in _lemma_acc]))

if __name__ == "__main__":
    
    curlang = sys.argv[1]
        
    all_corpora = (
        set(cliqs.corpora.all_ud_dev_corpora)
        & set(cliqs.corpora.all_ud_test_corpora)
        & set(cliqs.corpora.all_ud_train_corpora)
    )

    print(curlang, all_corpora, "ALL")

    # Load Data
    test_triples = {
        lang: extract_triples_from_corpus(cliqs.corpora.all_ud_test_corpora[lang])
        # if "Turkish" in lang #and "UD_Romanian-RRT" not in lang)
        for lang in all_corpora if curlang in lang
        # for lang in all_corpora if "UD_German-GSD" in lang
    }


    def take(xs, n):
        return itertools.islice(xs, None, n)


    CUTOFF = 200
    SIZE_LIMIT = CUTOFF * 8
    good_langs = {lang for lang in test_triples if len(
        test_triples[lang]) >= CUTOFF}
    print(good_langs)

    train_triples = {
        lang: list(take(extract_triples_from_corpus(
            cliqs.corpora.all_ud_train_corpora[lang]), SIZE_LIMIT))
        for lang in good_langs
    }

    dev_triples = {
        lang: extract_triples_from_corpus(cliqs.corpora.all_ud_dev_corpora[lang])
        for lang in good_langs
    }

    d_with_order = get_data(good_langs, True)
    d_without_order = get_data(good_langs, False)

    order_data = {
        (lang, "order"): accuracy(d_with_order[lang]) for lang in good_langs
    }

    no_order_data = {
        (lang, "no_order"): accuracy(d_without_order[lang]) for lang in good_langs
    }

    print(order_data)
    print(no_order_data)

    pickle.dump(order_data, open("data3/order/" + curlang, "wb"))

    pickle.dump(no_order_data, open("data3/no_order/" + curlang, "wb"))
