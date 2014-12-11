import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from itertools import chain
from collections import Counter

import project.CoreNLP
from project.text import filename_to_id, strip_non_words

from project import features

################################################################################
#
# Binary naive bayes bag-of-words classifier
#
################################################################################


def preprocess(train_files, test_files):

    # Use all files (train and test) as the corpus:
    return [project.CoreNLP.tokenize_keys(train_files + test_files)]


def train(train_files, train_ids, Y, all_tokens_dict):

    # all_tokens_dict returned by preprocess() above:

    F = features.build('binary_bag_of_words', train_ids, all_tokens_dict)
    X = features.featurize('binary_bag_of_words', F, train_ids)

    nb = MultinomialNB()
    nb.fit(X, Y)
 
    return (nb, F)


def predict(model, test_files, test_ids, all_tokens_dict):

    # tokens_by_file is passed by classify.py from this models's preprocess()

    (nb, F) = model

    X = features.featurize('binary_bag_of_words', F, test_ids)

    return nb.predict(X)
