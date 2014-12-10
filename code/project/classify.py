# -*- coding: utf-8 -*-

from importlib import import_module
from random import shuffle, randint
from logging import debug, info, warn

from project import resources
from project.text import filename_to_id
from project.utils.files import resolve

################################################################################
# Classification pipeline code goes in here
################################################################################

def nfold_xval(training_set, n=10):
    """
    If training_set is a list, assume it is a list of file name (observation
    identifiers)

    If training_set is a dict, take training_set.items() and using the first
    key    
    """
    ob_ids  = None
    is_dict = False
    if isinstance(training_set, dict):
        ob_ids  = map(lambda (key,_): key, training_set.items())
        is_dict = True
    elif isinstance(training_set, list):
        ob_ids = training_set
    else:
        raise ValueError('training_set: not dict or list')

    # [(fold, observation-id)]
    ob_ids_folds = [(i % n, ob_id) for (i, ob_id) in enumerate(ob_ids, start=0)]

    # Pick a random fold to hold out:
    hold_out_fold = randint(0, n-1)

    # Mix it up
    shuffle(ob_ids_folds)

    # Partition into held and kept sets:
    kept     = [ob_id for (fold, ob_id) in ob_ids_folds if fold != hold_out_fold]
    held_out = [ob_id for (fold, ob_id) in ob_ids_folds if fold == hold_out_fold]

    if is_dict:
        # Reconstruct the dicts:
        kept_dict     = {ob_id: training_set[ob_id] for ob_id in kept} 
        held_out_dict = {ob_id: training_set[ob_id] for ob_id in held_out}
        return (kept_dict, held_out_dict, hold_out_fold)
    else:
        return (kept, held_out, hold_out_fold)

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
# Note: Models are resolved relative to the project.models module
#
################################################################################

def get_model(name):
    """
    Loads the model module specified by name relative the project's
    relative to the project.models module

    @returns module
    """
    return import_module('project.models.{}'.format(name))


def train(model_name, train_files, *args, **kwargs):
    """
    train(model_name, train_files, *args, **kwargs) -> model
    """
    model_module = get_model(model_name)
    return model_module.train(train_files, *args, **kwargs)


def predict(model_name, trained_model, test_files, *args, **kwargs):
    """
    predict(model_name, model, *args, **kwargs) -> {str: int}
    """
    model_module = get_model(model_name)
    return model_module.predict(trained_model, test_files, *args, **kwargs)


def test(model_name, n_folds=10, suppress_output=False, show_results=False, *args, **kwargs):
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

    # Divide the observation data into two sets for training and testing:
    (train_files, test_files, hold_out_fold) = nfold_xval(observations, n=n_folds)

    assert(len(observations) == (len(train_files) + len(test_files)))

    info("> N-fold xval: #folds: {}, |train| (kept): {}, |test| (held out): {}, fold: {}"\
         .format(n_folds, len(train_files), len(test_files), hold_out_fold))
    
    # Build the training model on all data but the held out data
    trained_model    = train(model_name, train_files, *args, **kwargs)

    # Use the trained model to make predictions
    predicted_labels = predict(model_name, trained_model, test_files, *args, **kwargs)

    # Compare the accuracy of the prediction against the actual labels:
    correct_count      = 0 
    total_observations = len(predicted_labels)

    for (observation_id, predicted_label) in predicted_labels.items():

        actual_label = labels[observation_id]
        result       = predicted_label == actual_label

        # Correct:
        if result:
            correct_count += 1

        if not suppress_output and show_results:
            debug("  [{}]  {}\t{}\t{}"\
                  .format('x' if result else ' ', observation_id, predicted_label, actual_label))

    incorrect_count = total_observations - correct_count
    accuracy        = (float(correct_count) / float(total_observations)) * 100.0

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

