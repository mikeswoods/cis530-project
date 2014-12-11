import numpy as np
import math

from itertools import chain
from collections import Counter

from project.utils.counting import counts_to_probs
from project.text import strip_non_words


################################################################################
# Binary bag of words featurizer
################################################################################


def compute_TF(all_tokens_dict):
	"""
	Compute term frequeny per sentence

	@returns {str:sentence : {str:word: float:P(word|sentence)}}
	"""
	return {sentence: counts_to_probs(Counter(strip_non_words(words))) \
	        for (sentence, words) in all_tokens_dict.items()}


def compute_DF(all_tokens_dict):
	"""
	Compute document frequency per word

	@returns {str:word : int:document-count}
	"""
	df_counts = Counter() # Number of times a word occurs in all observations

	# Tabulate the number of documents each word appears in
	for words in all_tokens_dict.values():

		for word in set(strip_non_words(words)):

			if word not in df_counts:
				df_counts[word] = 1
			else:
				df_counts[word] += 1

	return df_counts


def compute_TFIDF(all_tokens_dict):
	"""
	Computes TDIDF for a given word

	@returns {str:word : float:TFIDF-score}
	"""
	sentences = all_tokens_dict.keys()

	TF = compute_TF(all_tokens_dict)
	DF = compute_DF(all_tokens_dict)
	N  = float(len(sentences))

	TFIDF = {}

	for sentence in sentences:

		TFIDF[sentence] = {}

		for (word, tf) in TF[sentence].items():

			IDF = math.log(N / DF[word])
			TFIDF[sentence][word] = tf * IDF

	return TFIDF


def build(train_ids, all_tokens_dict):

	# Strip all the junk:
	corpus = strip_non_words(chain.from_iterable(all_tokens_dict.values()))

	# Get all unique words:
	unique_words = list(set(corpus))

	# Term frequeny probabilities per sentence
	TFIDF = compute_TFIDF(all_tokens_dict)

	# Unknown token IDF value
	UNK = math.log(len(all_tokens_dict))

	# Sort in ascending order from A..Z
	unique_words.sort()

	# Assign an index to each word
	word_indices = {k:i for (i,k) in enumerate(unique_words, start=0)}

	return (all_tokens_dict, word_indices, TFIDF, UNK)


# observation_ids can be training IDs or test observation IDs
def featureize(F, observation_ids):

	(all_tokens_dict,  word_indices, TFIDF, UNK) = F

	n = len(word_indices)
	m = len(observation_ids)

	 # Observations
	X = np.zeros((m,n), dtype=np.int)

	for (i,ob_id) in enumerate(observation_ids, start=0):
	    for token in strip_non_words(all_tokens_dict[ob_id]):
	        j = word_indices[token]
	        X[i][j] = TFIDF[ob_id][token]

	return X
