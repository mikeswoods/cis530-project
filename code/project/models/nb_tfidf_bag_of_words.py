import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB

import project.CoreNLP
from project import features


def preprocess(train_files, test_files):

    # Use all files (train and test) as the corpus:
    return [project.CoreNLP.tokens_with_key(train_files + test_files)]


def train(train_files, train_ids, Y, all_tokens_dict):

    # all_tokens_dict returned by preprocess() above:

    # See project.feature.binary_bag_of_words.py
    F = features.build('tfidf_bag_of_words', train_ids, all_tokens_dict)
    X = features.featurize('tfidf_bag_of_words', F, train_ids)

    nb = MultinomialNB()
    nb.fit(X, Y)
 
    return (nb, F)


def predict(model, test_files, test_ids, all_tokens_dict):

    # all_tokens_dict is ignored

    (nb, F) = model

    X = features.featurize('tfidf_bag_of_words', F, test_ids)

    return nb.predict(X)