import numpy as np
from sklearn.naive_bayes import GaussianNB

from itertools import chain
from collections import Counter


import project.CoreNLP
from project.text import filename_to_id, strip_non_words

################################################################################
#
# Simple naive bayes bag-of-words classifier
#
################################################################################


def train(train_files, train_labels):
    """
    @returns GaussianNB
    """
    tokens_by_file = project.CoreNLP.tokenize_keys(train_files)
    ob_ids         = filename_to_id(train_files)

    # Strip all the junk:
    corpus = strip_non_words(chain.from_iterable(tokens_by_file.values()))
    counts = Counter(corpus)

    keys = counts.keys()
    keys.sort()
    keys_indices = {k:i for (i,k) in enumerate(keys, start=0)}

    n = len(keys)
    m = len(ob_ids)
    print n
    print m
    print len(keys_indices), "keys indices"

    # Observations
    X = np.zeros((m,n), dtype=np.int)

    for (i,ob_id) in enumerate(ob_ids, start=0):
        for token in strip_non_words(tokens_by_file[ob_id]):
            j = keys_indices[token]
            X[i][j] = 1

    # Labels
    Y = np.zeros((m,), dtype=np.int)
    for (i,ob_id) in enumerate(ob_ids, start=0):
        Y[i] = train_labels[ob_id]

    gnb = GaussianNB()
    gnb.fit(X, Y)
    print X.shape

    return (gnb, keys_indices, tokens_by_file)


def predict(model, test_files):
    """
    """
    keys_indices = model[1]
    tokens_by_file = model[2]
    model = model[0]
    tokens_by_file = project.CoreNLP.tokenize_keys(test_files)
    ob_ids = filename_to_id(test_files)
    # Strip all the junk:
    corpus = strip_non_words(chain.from_iterable(tokens_by_file.values()))
    counts = Counter(corpus)

    n = len(keys_indices)
    m = len(ob_ids)

    # Observations
    X = np.zeros((m,n), dtype=np.int)
    for (i,ob_id) in enumerate(ob_ids, start=0):
        for token in strip_non_words(tokens_by_file[ob_id]):
            if token in keys_indices:
            	j = keys_indices[token]
            	X[i][j] = 1
    print X.shape
    print model.predict(X)
    return model.predict(X)
