import numpy as np
from sklearn import svm
from sklearn import preprocessing

import project.CoreNLP
from project import features
from project import resources
from project.utils.lists import index_of, pick


def preprocess(train_files, test_files):

    CoreNLP_data    = project.CoreNLP.all_sentences()
    all_tokens_dict = project.CoreNLP.tokens_with_key(CoreNLP_data, train_files + test_files)

    F = {
         'bag_of_words': features.build('binary_bag_of_words', all_tokens_dict)
        ,'LIWC':  features.build('liwc', resources.liwc_data)
        ,'MRC_bag_of_words':  features.build('MRC_bag_of_words', resources.mrc_words_file, all_tokens_dict)
    }

    return [CoreNLP_data, F]


def train(train_files, train_ids, Y, CoreNLP_data, F, *args, **kwargs):

    X = np.hstack([
         features.featurize('binary_bag_of_words', F['bag_of_words'], train_ids)
        ,features.featurize('CoreNLP_sentence_info', None, train_files, CoreNLP_data)
        ,features.featurize('liwc', F['LIWC'], train_ids)
        ,features.featurize('MRC_bag_of_words', F['MRC_bag_of_words'], train_ids)
    ])
    X = preprocessing.scale(X)

    M = svm.LinearSVC()
    M.fit(X, Y)

    return (M,)


def predict(model, test_files, test_ids, CoreNLP_data, F, *args, **kwargs):

    (M,) = model

    X = np.hstack([
         features.featurize('binary_bag_of_words', F['bag_of_words'], test_ids)
        ,features.featurize('CoreNLP_sentence_info', None, test_files, CoreNLP_data)
        ,features.featurize('liwc', F['LIWC'], test_ids)
        ,features.featurize('MRC_bag_of_words', F['MRC_bag_of_words'], test_ids)
    ])
    X = preprocessing.scale(X)

    return M.predict(X)
