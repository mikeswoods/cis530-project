import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

from logging import info
import project.CoreNLP
from project import features
from project import resources
from project.utils.lists import index_of, pick

################################################################################
#
# A logistic regression classifier run on a reduced dimension feature space 
# after PCA has been applied and gridsearch has been used for parameter tuning
#
################################################################################

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
        ,'production_rules':  features.build('production_rules', CoreNLP_train_data)
    }

    return [CoreNLP_train_data, CoreNLP_test_data, F]


def train(train_files, train_ids, Y, CoreNLP_train_data, CoreNLP_test_data, F, *args, **kwargs):

    X = np.hstack([
        # features.featurize('binary_bag_of_words', F['bag_of_words'], train_ids)
         features.featurize('CoreNLP_sentence_info', None, train_files, CoreNLP_train_data)
        ,features.featurize('liwc', F['LIWC'], train_ids)
        ,features.featurize('MRC_bag_of_words', F['MRC_bag_of_words'], train_ids, project.CoreNLP.tokens_with_key(CoreNLP_train_data), binary=True)
        ,features.featurize('production_rules', F['production_rules'], train_ids, CoreNLP_train_data, binary=True)
    ])

    pca = PCA()
    M   = LogisticRegression()

    info("> Running PCA...")
    pca.fit(X)
    pipe = Pipeline(steps=[('pca', pca), ('logistic', M)])

    n_components = [25, 50, 100, 250, 500]
    Cs = np.logspace(-4, 4, 3)

    info("Running gridsearch...")
    estimator = GridSearchCV(pipe, {'pca__n_components': n_components, 'logistic__C': Cs})

    info("Fitting...")
    estimator.fit(X, Y)

    return (estimator,)


def predict(model, test_files, test_ids, CoreNLP_train_data, CoreNLP_test_data, F, *args, **kwargs):

    (M,) = model

    X = np.hstack([
         # features.featurize('binary_bag_of_words', F['bag_of_words'], test_ids)
          features.featurize('CoreNLP_sentence_info', None, test_files, CoreNLP_test_data)
         ,features.featurize('liwc', F['LIWC'], test_ids)
         ,features.featurize('MRC_bag_of_words', F['MRC_bag_of_words'], test_ids, project.CoreNLP.tokens_with_key(CoreNLP_test_data), binary=True)
         ,features.featurize('production_rules', F['production_rules'], test_ids, CoreNLP_test_data, binary=True)
    ])

    info("Predicting...")

    return estimator.predict(X)
