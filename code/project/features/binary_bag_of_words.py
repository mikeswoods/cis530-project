import numpy as np

from itertools import chain
from collections import Counter

from project.text import strip_non_words


################################################################################
# Binary bag of words featurizer
################################################################################


def build(train_ids, all_tokens_dict):

	# Strip all the junk:
	corpus = strip_non_words(chain.from_iterable(all_tokens_dict.values()))

	# Get all unique words:
	unique_words = list(set(corpus))

	# Sort in ascending order from A..Z
	unique_words.sort()

	# Assign an index to each word
	word_indices = {k:i for (i,k) in enumerate(unique_words, start=0)}

	return (all_tokens_dict, word_indices)


# observation_ids can be training IDs or test observation IDs
def featureize(F, observation_ids):

	(all_tokens_dict, word_indices) = F

	n = len(word_indices)
	m = len(observation_ids)

	 # Observations
	X = np.zeros((m,n), dtype=np.int)

	for (i,ob_id) in enumerate(observation_ids, start=0):
	    for token in strip_non_words(all_tokens_dict[ob_id]):
	        j = word_indices[token]
	        X[i][j] = 1

	return X
