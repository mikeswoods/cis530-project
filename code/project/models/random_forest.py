import numpy as np
from sklearn.linear_model import LogisticRegression, BayesianRidge

import project.CoreNLP
from project import features
from os.path import dirname
from project.utils.files import resolve
from project.utils.lists import index_of, pick


def preprocess(train_files, test_files):

    # Use all files (train and test) as the corpus:

    CoreNLP_data    = project.CoreNLP.all_sentences()
    all_tokens_dict = project.CoreNLP.tokens_with_key(train_files + test_files)
    liwc_data       = resolve(dirname(__file__), '..', '..', '..', 'data', 'LIWC', 'all_data_LIWC.dat')

    F = {
         'bag_of_words': features.build('binary_bag_of_words', all_tokens_dict)
        ,'LIWC':  features.build('liwc', liwc_data)
    }

    return [CoreNLP_data, F]


def train(train_files, train_ids, Y, CoreNLP_data, F, *args, **kwargs):

    #X1 = features.featurize('binary_bag_of_words', F['bag_of_words'], train_ids)
    X2 = features.featurize('CoreNLP_sentence_info', None, train_files, CoreNLP_data)
    X3 = features.featurize('liwc', F['LIWC'], train_ids)
    X  = np.hstack((X2, X3))
    #X = preprocessing.scale(X3)

    M = LogisticRegression()
    M.fit(X, Y)

    return (M,)


def predict(model, test_files, test_ids, CoreNLP_data, F, *args, **kwargs):

    (M,) = model

    #X1 = features.featurize('binary_bag_of_words', F['bag_of_words'], test_ids)
    X2 = features.featurize('CoreNLP_sentence_info', None, test_files, CoreNLP_data)
    X3 = features.featurize('liwc', F['LIWC'], test_ids)
    X  = np.hstack((X2, X3))
    #X = preprocessing.scale(X3)

    return M.predict(X)
