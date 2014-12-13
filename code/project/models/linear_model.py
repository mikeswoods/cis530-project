import numpy as np
from sklearn.linear_model import LogisticRegression, BayesianRidge, RANSACRegressor

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
        ,'production_rules':  features.build('production_rules', CoreNLP_data)
    }

    return [CoreNLP_data, F]


def train(train_files, train_ids, Y, CoreNLP_data, F, *args, **kwargs):

    X = np.hstack([
    #     features.featurize('binary_bag_of_words', F['bag_of_words'], train_ids)
         features.featurize('CoreNLP_sentence_info', None, train_files, CoreNLP_data)
        ,features.featurize('liwc', F['LIWC'], train_ids)
        ,features.featurize('MRC_bag_of_words', F['MRC_bag_of_words'], train_ids, binary=True)
        ,features.featurize('production_rules', F['production_rules'], train_ids, CoreNLP_data, binary=True)
    ])

    M = LogisticRegression(penalty='l1', class_weight={1: 0.58, -1:0.42})
    M.fit(X, Y)

    return (M,)


def predict(model, test_files, test_ids, CoreNLP_data, F, *args, **kwargs):

    (M,) = model

    X = np.hstack([
    #     features.featurize('binary_bag_of_words', F['bag_of_words'], test_ids)
          features.featurize('CoreNLP_sentence_info', None, test_files, CoreNLP_data)
         ,features.featurize('liwc', F['LIWC'], test_ids)
         ,features.featurize('MRC_bag_of_words', F['MRC_bag_of_words'], test_ids, binary=True)
         ,features.featurize('production_rules', F['production_rules'], test_ids, CoreNLP_data, binary=True)
    ])

    return M.predict(X)
