import numpy as np

from itertools import chain
from collections import Counter

from project import resources
from project.text import filename_to_id
from project.utils.files import resolve

import project.CoreNLP
from project.text import filename_to_id, strip_non_words

################################################################################
# Binary bag of words featurizer
################################################################################


def build(train_ids, all_tokens_dict):

	# Strip all the junk:
	corpus = strip_non_words(chain.from_iterable(all_tokens_dict.values()))

	counts = Counter(corpus)

	keys = counts.keys()
	keys.sort()
	keys_indices = {k:i for (i,k) in enumerate(keys, start=0)}

	return (corpus, all_tokens_dict, counts, keys_indices)


def featureize(F, train_ids):

	(corpus, all_tokens_dict, counts, keys_indices) = F

	n = len(keys_indices)
	m = len(train_ids)

	 # Observations
	X = np.zeros((m,n), dtype=np.int)

	for (i,ob_id) in enumerate(train_ids, start=0):
	    for token in strip_non_words(all_tokens_dict[ob_id]):
	        j = keys_indices[token]
	        X[i][j] = 1

	return X
