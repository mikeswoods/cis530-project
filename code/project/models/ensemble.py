import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing

import project.CoreNLP
from project import features
from project import resources
from project.utils.lists import index_of, pick


def majority_vote(Y1, Y2, Y3):
  assert(len(Y1) == len(Y2) and len(Y2) == len(Y3))
  n = len(Y1)
  Y = np.zeros((n,), dtype=np.int)
  for i in range(n):
    l1_count = 0 # Label: 1
    l2_count = 0 # Label: -1
    if Y1[i] == 1:
      l1_count += 1
    else:
      l2_count += 1
    if Y2[i] == 1:
      l1_count += 1
    else:
      l2_count += 1
    if Y3[i] == 1:
      l1_count += 1
    else:
      l2_count += 1
    if l1_count > l2_count:
      Y[i] = 1
    else:
      Y[i] = -1

  return Y


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

    X = np.hstack([
       # features.featurize('binary_bag_of_words', F['bag_of_words'], train_ids)
       features.featurize('CoreNLP_sentence_info', None, train_files, CoreNLP_train_data)
      ,features.featurize('liwc', F['LIWC'], train_ids)
      ,features.featurize('MRC_bag_of_words', F['MRC_bag_of_words'], train_ids, project.CoreNLP.tokens_with_key(CoreNLP_train_data), binary=True)
      ,features.featurize('dependency_relations', F['dependency_relations'], train_ids, CoreNLP_train_data, binary=True)
      ,features.featurize('production_rules', F['production_rules'], train_ids, CoreNLP_train_data, binary=True)
    ])

    M1 = LogisticRegression(class_weight={1: 0.58, -1:0.42})
    M2 = LinearSVC(class_weight={1: 0.58, -1: 0.42})
    M3 = BernoulliNB()

    M1.fit(X, Y)
    M2.fit(preprocessing.scale(X), Y)
    M3.fit(X, Y)

    return (M1, M2, M3)


def predict(model, test_files, test_ids, CoreNLP_train_data, CoreNLP_test_data, F, *args, **kwargs):

    (M1, M2, M3) = model

    X = np.hstack([
       # features.featurize('binary_bag_of_words', F['bag_of_words'], test_ids)
       features.featurize('CoreNLP_sentence_info', None, test_files, CoreNLP_test_data)
      ,features.featurize('liwc', F['LIWC'], test_ids)
      ,features.featurize('MRC_bag_of_words', F['MRC_bag_of_words'], test_ids, project.CoreNLP.tokens_with_key(CoreNLP_test_data), binary=True)
      ,features.featurize('dependency_relations', F['dependency_relations'], test_ids, CoreNLP_test_data, binary=True)
      ,features.featurize('production_rules', F['production_rules'], test_ids, CoreNLP_test_data, binary=True)
    ])

    Y1 = M1.predict(X)
    Y2 = M2.predict(preprocessing.scale(X))
    Y3 = M3.predict(X)
    Y  = majority_vote(Y1, Y2, Y3)

    y1n = len(Y1)
    y2n = len(Y2)
    y3n = len(Y3)
    yn  = len(Y)
    assert(y1n == y2n and y2n == y3n and y3n == yn)

    return Y
