import numpy as np
from sklearn import svm

import project.CoreNLP
from project import features
from os.path import dirname
from project.utils.files import resolve
from project.utils.lists import index_of, pick


def preprocess(train_files, test_files):

    # Use all files (train and test) as the corpus:

    CoreNLP_data    = project.CoreNLP.all_sentences()
    all_tokens_dict = project.CoreNLP.tokens_with_key(train_files + test_files)

    BAG_OF_WORDS = features.build('binary_bag_of_words', all_tokens_dict)

    return [CoreNLP_data, BAG_OF_WORDS]


def train(train_files, train_ids, Y, CoreNLP_data, BAG_OF_WORDS, *args, **kwargs):

    LIWC = features.build('liwc', resolve(dirname(__file__), '..', '..', '..', 'data', 'LIWC', 'train_data_LIWC.dat'))

    X1 = features.featurize('binary_bag_of_words', BAG_OF_WORDS, train_ids)
    X2 = features.featurize('CoreNLP_sentence_info', None, train_files, CoreNLP_data)
    X3 = features.featurize('liwc', LIWC, train_ids)

    X = (X1, X2, X3)

    svm_model = svm.LinearSVC()
    svm_model.fit(np.hstack(X), Y)

    return (svm_model,)


def predict(model, test_files, test_ids, CoreNLP_data, BAG_OF_WORDS, *args, **kwargs):

    (svm_model,) = model

    LIWC = features.build('liwc', resolve(dirname(__file__), '..', '..', '..', 'data', 'LIWC', 'train_data_LIWC.dat'))

    X1 = features.featurize('binary_bag_of_words', BAG_OF_WORDS, test_ids)
    X2 = features.featurize('CoreNLP_sentence_info', None, test_files, CoreNLP_data)
    X3 = features.featurize('liwc', LIWC, test_ids)

    X = (X1, X2, X3)

    return svm_model.predict(np.hstack(X))
