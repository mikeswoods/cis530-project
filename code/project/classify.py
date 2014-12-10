from random import shuffle, randint


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


def classify():
    pass
