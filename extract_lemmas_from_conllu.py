import sys
from collections import defaultdict

def main(filename):
    l2w = defaultdict(set)
    with open(filename) as infile:
        for line in infile:
            line = line.strip()
            if line and not line.startswith("#"):
                _, word, lemma, *_ = line.split("\t")
                l2w[lemma.casefold()].add(word.casefold())
    for lemma, words in l2w.items():
        for word in words:
            print(lemma, word, sep="\t")

if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
