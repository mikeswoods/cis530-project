import numpy as np
from sklearn import svm
from sklearn import preprocessing

from logging import info
import project.CoreNLP
from project import features
from project import resources
from project.utils.lists import index_of, pick


def preprocess(train_files, test_files):

    CoreNLP_train_data    = project.CoreNLP.all_sentences('train')

    # If we're in submit mode, use the actual train and test sets,
    # otherwise just use train for both

    if resources.SUBMIT_MODE:
        CoreNLP_test_data = project.CoreNLP.all_sentences('test')
    else:
        CoreNLP_test_data = CoreNLP_train_data
        
    #all_train_tokens_dict = project.CoreNLP.tokens_with_key(CoreNLP_train_data)

    F = {
      # 'bag_of_words': features.build('binary_bag_of_words', all_train_tokens_dict)
       'LIWC':  features.build('liwc', resources.liwc_data)
      ,'MRC_bag_of_words':  features.build('MRC_bag_of_words', resources.mrc_words_file)
      ,'dependency_relations': features.build('dependency_relations', CoreNLP_train_data)
      ,'production_rules':  features.build('production_rules', CoreNLP_train_data)
    }

    return [CoreNLP_train_data, CoreNLP_test_data, F]


def train(train_files, train_ids, Y, CoreNLP_train_data, CoreNLP_test_data, F, *args, **kwargs):

    X = preprocessing.scale(np.hstack([
      # features.featurize('binary_bag_of_words', F['bag_of_words'], train_ids)
       features.featurize('CoreNLP_sentence_info', None, train_files, CoreNLP_train_data)
      ,features.featurize('liwc', F['LIWC'], train_ids)
      ,features.featurize('MRC_bag_of_words', F['MRC_bag_of_words'], train_ids, project.CoreNLP.tokens_with_key(CoreNLP_train_data), binary=True)
      ,features.featurize('dependency_relations', F['dependency_relations'], train_ids, CoreNLP_train_data, binary=True)
      ,features.featurize('production_rules', F['production_rules'], train_ids, CoreNLP_train_data, binary=True)
    ]))

    M = svm.LinearSVC(class_weight={1: 0.58, -1: 0.42})
    M.fit(X, Y)

    return (M,)


def predict(model, test_files, test_ids, CoreNLP_train_data, CoreNLP_test_data, F, *args, **kwargs):

    (M,) = model

    X = preprocessing.scale(np.hstack([
      # features.featurize('binary_bag_of_words', F['bag_of_words'], test_ids)
       features.featurize('CoreNLP_sentence_info', None, test_files, CoreNLP_test_data)
      ,features.featurize('liwc', F['LIWC'], test_ids)
      ,features.featurize('MRC_bag_of_words', F['MRC_bag_of_words'], test_ids, project.CoreNLP.tokens_with_key(CoreNLP_test_data), binary=True)
      ,features.featurize('dependency_relations', F['dependency_relations'], test_ids, CoreNLP_test_data, binary=True)
      ,features.featurize('production_rules', F['production_rules'], test_ids, CoreNLP_test_data, binary=True)
    ]))

    return M.predict(X)
