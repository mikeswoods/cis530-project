# -*- coding: utf-8 -*-

import numpy as np
from sklearn import cross_validation

from importlib import import_module
from random import shuffle, randint
from logging import debug, info, warn

from project import resources
from project.text import filename_to_id
from project.utils.files import resolve

################################################################################
# Classification pipeline code goes in here
################################################################################

# ...

################################################################################
#
# Simple interface for interacting with all models.
#
# All models in <Base>/models/<model>.py must implement the following three
# functions:
#
# - train()
# - test()
# - predict()
#
# Example: train('nb_bag_of_words')
#
# Note: Models are resolved relative to the project.models modu
#
# Conceptual workflow:
#
# (1) my_data = preprocess('<MyModel>')
# 
#     (calls <MyModel>.preprocess if it exists. If not, my_data is set to []. 
#      In either case, <MyModel>.preprocess is expected to return a list)
#
# (2) model = train('<MyModel>', X_train, to_ids(X_train), Y_train, *my_data)
#
#     (calls <MyModel>.train)
#
# (3) Y_predict = predict('<MyModel>', model, X_test, to_ids(X_test), *my_data)
#
#     (calls <MyModel>.predict)
#
################################################################################

def get_model(name):
    """
    Loads the model module specified by name relative the project's
    relative to the project.models module

    @returns module
    """
    return import_module('project.models.{}'.format(name))


def preprocess(model_name, train_files, test_files):
    """
    Common model preprocessing interface -- calls the specified model's 
    preprocess(), if it exists, yielding a list of values that will be used
    as the positional arguments *args in later calls to the model's train()
    and predict() functions.

    preprocess(model_name, [str]:train_files, [str]:test_files) -> [*])

    if no preprocess() function is defined for the model specified by 
    model_name, [] will be returned
    """
    assert(isinstance(model_name, str))

    model_module = get_model(model_name)

    # Does the preprocess() function exist for the given model? Look in its
    # module
    if 'preprocess' in dir(model_module):
        
        data = model_module.preprocess(train_files, test_files)

        # Sanity check: data must be a list
        assert(isinstance(data, list))

        return data

    else:
        return []


def train(model_name, files, ids, labels, *args, **kwargs):
    """
    Common model train interface -- calls the specified model's train()
    function, yielding a trained model instance, whatever that may be

    train(str:model_name, [str]:files, [str]:ids, numpy.array(int):labels
         ,*args, **kwargs) -> model

    @param str:model_name The name of the model to load from the models dir
    @param [str]:files The list of filenames to train on containing observations
    @param [str]:files Observation identifiers defined by filenames. The
        identifier at index i corresponds to the filename at index i
    @param [*]:*args Additional positional arguments to pass down to the
        called model train() function
    @param {*:*}:**kwargs Additional keyword arguments to pass down to the
        called model train() function

    """
    # Sanity checking:
    assert(isinstance(model_name, str) and len(files) == len(ids))

    model_module = get_model(model_name)

    return model_module.train(files, ids, labels, *args, **kwargs)


def predict(model_name, trained_model, files, ids, *args, **kwargs):
    """
    Common model prediction interface -- calls the specified model's predict()
    function, yielding an numpy.array of predicted labels

    predict(str:model_name, *:model, *args, **kwargs) -> numpy.array(int)

    @param str:model_name The name of the model to load from the models dir
    @param *:trained_model The trained model instance produced by the previous
        call to train()
    @param [str]:files The list of filenames that predictions are to be made 
        for
    @param [str]:files Observation identifiers defined by filenames. The
        identifier at index i corresponds to the filename at index i
    @param [*]:*args Additional positional arguments to pass down to the
        called model predict() function
    @param {*:*}:**kwargs Additional keyword arguments to pass down to the
        called model predict() function
    """
    # Sanity checking:
    assert(isinstance(model_name, str) and len(files) == len(ids))

    model_module = get_model(model_name)

    Y = model_module.predict(trained_model, files, ids, *args, **kwargs)

    assert(len(files) == len(Y))

    return Y


def test(model_name, test_size=0.1, suppress_output=False, show_results=False, *args, **kwargs):
    """
    Runs a full test cycle for the given model

    Options:

    n_folds:         Number of cross-validation folds to produce
    suppress_output: If True, no output will be produced
    show_results:    If True, individual prediction results will be printed.
                     suppress_output must also be False for this option to work

    @returns (float:accuracy, int:correct_count, int:incorrect_count)
    """
    observations = resources.train_data_files('text')
    labels       = resources.train_data_labels()

    # Fed into sklearn cross_validation
    Y = np.array([labels[ob_id] for ob_id in filename_to_id(observations)])

    # Divide the observation data into two sets for training and testing:
    #(train_files, test_files, hold_out_fold) = nfold_xval(observations, n=n_folds)

    (train_files, test_files, train_labels, true_labels) = \
        cross_validation.train_test_split(observations, Y, test_size=test_size)
 
    assert(len(train_files) == len(train_labels) and len(test_files) == len(true_labels))

    info("> N-fold test size: {}, |train| (kept): {}, |test| (held out): {}"\
         .format(test_size, len(train_files), len(test_files)))
    
    # Get any preprocessing data and pass it to train() anbd predict() later:
    data = preprocess(model_name, train_files, test_files)

    # Generate training features:
    trained_model = train(model_name \
                         ,train_files \
                         ,filename_to_id(train_files) \
                         ,train_labels \
                         ,*data)

    # Same as training: the observation ID is just the basename of the input
    test_observation_ids = filename_to_id(test_files)

    # Use the trained model to make predictions:
    predicted_labels = predict(model_name \
                              ,trained_model \
                              ,test_files \
                              ,test_observation_ids \
                              ,*data)

    # Compare the accuracy of the prediction against the actual labels:
    correct_count      = 0 
    total_observations = len(predicted_labels)

    # For every test observation:
    for i in range(len(test_files)):

        # Get the test observsation ID:
        test_ob_id = test_observation_ids[i]

        # and compare it to the predicted label:
        result = true_labels[i] == predicted_labels[i]

        # Correct:
        if result:
            correct_count += 1

        if not suppress_output and show_results:
            debug("  [{}]  {}\t{}\t{}"\
                  .format('x' if result else ' ', observation_id, predicted_labels[i], true_labels[i]))

    incorrect_count = total_observations - correct_count
    accuracy        = 0 if total_observations == 0 \
                        else (float(correct_count) / float(total_observations)) * 100.0

    if not suppress_output:
        line = '*' * 80
        print 
        print line
        print "Accuracy: {}%".format(accuracy, correct_count)
        print "Correct: {} / {}".format(correct_count, total_observations)
        print line
        print 

    return (accuracy, correct_count, incorrect_count)


def test_iterations(model_name, N, n_folds=10, *args, **kwargs):
    """
    Like test, but runs N iterations, averaging the results
    """
    total_correct_count   = 0
    total_incorrect_count = 0        

    for i in range(N):
        info('Iteration: {}'.format(i+1))
        (accuracy, correct_count, incorrect_count) = test(model_name \
                                                         ,n_folds=10 \
                                                         ,suppress_output=True \
                                                         ,show_results=False \
                                                         ,*args \
                                                         ,**kwargs)

        total_correct_count += correct_count
        total_incorrect_count += incorrect_count

    total_observations = total_correct_count + total_incorrect_count
    total_accuracy     = (float(total_correct_count) / float(total_observations)) * 100.0

    print total_accuracy

