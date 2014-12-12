import numpy as np

from itertools import chain
from collections import Counter

from nltk.stem import PorterStemmer

from project.text import strip_non_words


################################################################################
# Binary bag of words featurizer
################################################################################

# Helpers

def process_words(all_words):

	# Additionall, remove tokens like "'s", "'nt"
	remove = set(["'s", "'nt"])

	return [word for word in strip_non_words(all_words) if word not in remove]


################################################################################

def build(train_ids, all_tokens_dict):

	# Strip all the junk:
	corpus = process_words(chain.from_iterable(all_tokens_dict.values()))

	#stemmer = PorterStemmer()
	#corpus = map(stemmer.stem, corpus)

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

	stemmer = PorterStemmer()

	for (i,ob_id) in enumerate(observation_ids, start=0):
	    for token in process_words(all_tokens_dict[ob_id]):
	    	#j = word_indices[stemmer.stem(token)]
	        j = word_indices[token]
	        X[i][j] = 1

	return X
