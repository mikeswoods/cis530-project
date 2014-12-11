from importlib import import_module

################################################################################
# Methods of featurizing text go here
#
# All features are expected to define two functions for their interface:
#
# - build(*args, **kwargs) => Creates the featurizer, F
#
# - featurize(F, *args, **kwargs) => Featurizes the input based on the features
#     F created with build() prior
#
################################################################################

# ...

def get_feature(name):
    """
    Loads the feature module specified by name relative the project's
    relative to the project.features module

    @returns module
    """
    return import_module('project.features.{}'.format(name))


def build(name, *args, **kwargs):
	"""
	Builds the feature set given by name with the arguments passed
	"""
	return get_feature(name).build(*args, **kwargs)


def featurize(name, F, *args, **kwargs):
	"""
	Featurizes the input using F condstructed from build() earlier
	"""
	return get_feature(name).featureize(F, *args, **kwargs)
