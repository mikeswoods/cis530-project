import numpy as np
from sklearn import svm

import project.CoreNLP
from project import features


def preprocess(train_files, test_files):

    # Use all files (train and test) as the corpus:

    CoreNLP_data    = project.CoreNLP.all_sentences()
    all_tokens_dict = project.CoreNLP.tokens_with_key(train_files + test_files)

    BAG_OF_WORDS = features.build('binary_bag_of_words', all_tokens_dict)

    return [CoreNLP_data, BAG_OF_WORDS]


def train(train_files, train_ids, Y, CoreNLP_data, BAG_OF_WORDS, *args, **kwargs):

    X1 = features.featurize('CoreNLP_sentence_info', None, train_files, CoreNLP_data)
    X2 = features.featurize('binary_bag_of_words', BAG_OF_WORDS, train_ids)

    X = (X1, X2)

    svm_model = svm.LinearSVC()
    svm_model.fit(np.hstack(X), Y)
 
    return (svm_model,)


def predict(model, test_files, test_ids, CoreNLP_data, BAG_OF_WORDS, *args, **kwargs):

    # all_tokens_dict is ignored

    (svm_model,) = model

    X1 = features.featurize('CoreNLP_sentence_info', None, test_files, CoreNLP_data)
    X2 = features.featurize('binary_bag_of_words', BAG_OF_WORDS, test_ids)

    X  = (X1, X2)

    return svm_model.predict(np.hstack(X))
