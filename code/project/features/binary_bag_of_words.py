# -*- coding: utf-8 -*-

import numpy as np

from itertools import chain
from collections import Counter

from project.utils.text import strip_junk_tokens

################################################################################
# Binary bag of words featurizer
################################################################################

def build(all_tokens_dict):

	# Strip all the junk:
	corpus = strip_junk_tokens(chain.from_iterable(all_tokens_dict.values()))

	# Get all unique words:
	unique_words = list(set(corpus))

	# Sort in ascending order from A..Z
	unique_words.sort()

	# Assign an index to each word
	corenlp_words_index = {k:i for (i,k) in enumerate(unique_words)}

	return (all_tokens_dict, corenlp_words_index)


# observation_ids can be training IDs or test observation IDs
def featureize(F, observation_ids):

	(all_tokens_dict, corenlp_words_index) = F

	n = len(corenlp_words_index)
	m = len(observation_ids)

	# Observations
	X = np.zeros((m,n), dtype=np.float)

	for (i,ob_id) in enumerate(observation_ids, start=0):

	    for token in strip_junk_tokens(all_tokens_dict[ob_id]):

	    	# Binary indicator:
	        X[i][corenlp_words_index[token]] = 1

	return X
