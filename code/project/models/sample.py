from project.text import filename_to_id

################################################################################
#
# Sample dumb classifier that always predicts '1'
#
################################################################################


def train(train_files):
    """
    No model to build
    """
    return {}

def predict(model, test_files):
    """
    Always predicts 1
    """
    return {observation_id: 1 for observation_id in filename_to_id(test_files)}
