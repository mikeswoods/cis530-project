import numpy as np

################################################################################
#
# Sample dumb classifier that always predicts '1'
#
################################################################################


def train(files, ids, labels):

    return None


def predict(model, ids, files):

    return np.array([1 for ob_id in ids])
