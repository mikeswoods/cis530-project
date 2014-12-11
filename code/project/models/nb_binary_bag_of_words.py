import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from itertools import chain
from collections import Counter

import project.CoreNLP
from project.text import filename_to_id, strip_non_words

################################################################################
#
# Simple naive bayes bag-of-words classifier
#
################################################################################


def preprocess(train_files, test_files):

    # Use all files (train and test) as the corpus:
    return [project.CoreNLP.tokenize_keys(train_files + test_files)]


def train(files, ids, Y, tokens_by_file):

    # tokens_by_file is passed by classify.py from this models's preprocess()

    # Strip all the junk:
    corpus = strip_non_words(chain.from_iterable(tokens_by_file.values()))
    counts = Counter(corpus)

    keys = counts.keys()
    keys.sort()
    keys_indices = {k:i for (i,k) in enumerate(keys, start=0)}

    n = len(keys)
    m = len(ids)

    # Observations
    X = np.zeros((m,n), dtype=np.int)

    for (i,ob_id) in enumerate(ids, start=0):
        for token in strip_non_words(tokens_by_file[ob_id]):
            j = keys_indices[token]
            X[i][j] = 1

    nb = MultinomialNB()
    nb.fit(X, Y)
 
    return (nb, keys_indices)


def predict(model, files, ids, tokens_by_file):

    # tokens_by_file is passed by classify.py from this models's preprocess()

    keys_indices = model[1]
    model = model[0]

    # Strip all the junk:
    corpus = strip_non_words(chain.from_iterable(tokens_by_file.values()))
    counts = Counter(corpus)

    n = len(keys_indices)
    m = len(ids)

    # Observations
    X = np.zeros((m,n), dtype=np.int)
    for (i,ob_id) in enumerate(ids, start=0):
        for token in strip_non_words(tokens_by_file[ob_id]):
            if token in keys_indices:
            	j = keys_indices[token]
            	X[i][j] = 1

    return model.predict(X)
