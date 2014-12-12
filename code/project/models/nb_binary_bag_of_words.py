import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB

import project.CoreNLP
from project import features


def preprocess(train_files, test_files):

    # Use all files (train and test) as the corpus:
    return [project.CoreNLP.tokens_with_key(train_files + test_files)]


def train(train_files, train_ids, Y, all_tokens_dict):

    X1 = features.featurize('sentence_info', None, train_files)

    # See project.feature.binary_bag_of_words.py
    F  = features.build('binary_bag_of_words', train_ids, all_tokens_dict)
    X2 = features.featurize('binary_bag_of_words', F, train_ids)

    X = (X1, X2)

    nb = MultinomialNB()
    nb.fit(np.hstack(X), Y)
 
    return (nb, F)


def predict(model, test_files, test_ids, all_tokens_dict):

    # all_tokens_dict is ignored

    (nb, F) = model

    X1 = features.featurize('sentence_info', None, test_files)
    X2 = features.featurize('binary_bag_of_words', F, test_ids)

    X  = (X1, X2)

    return nb.predict(np.hstack(X))
