import numpy as np
from itertools import *
from project.utils.lists import index_of, pick

# ['Seg', 'WC', 'WPS', 'Sixltr', 'Dic', 'funct', 'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'article', 'verb', 'auxverb', 'past', 'present', 'future', 'adverb', 'preps', 'conj', 'negate', 'quant', 'number', 'swear', 'social', 'family', 'friend', 'humans', 'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad', 'cogmech', 'insight', 'cause', 'discrep', 'tentat', 'certain', 'inhib', 'incl', 'excl', 'percept', 'see', 'hear', 'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'relativ', 'motion', 'space', 'time', 'work', 'achieve', 'leisure', 'home', 'money', 'relig', 'death', 'assent', 'nonfl', 'filler', 'Period', 'Comma', 'Colon', 'SemiC', 'QMark', 'Exclam', 'Dash', 'Quote', 'Apostro', 'Parenth', 'OtherP', 'AllPct']
# http://www.liwc.net/descriptiontable1.php

def read_data(filepath, exclude_lst):
	f = open(filepath, 'r')
	exclude_idx_lst = [False] * 81 #81 is number of features in liwc
	feature_tuple_lst = []
	idx = 0

	for line in f:

		line = line.strip()
		if len(line) == 0:
			continue

		if idx == 0: #skip line with labels
			labels = line.split()[1:]
			for exclusion in exclude_lst:
				exclude_idx = labels.index(exclusion)
				exclude_idx_lst[exclude_idx] = True
		else:
			line     = line.split()
			obs_id   = line[0]
			features = [float(x) for x in line[1:]]
			filtered_features = list(compress(features, map(lambda x: not x, exclude_idx_lst)))
			feature_tuple_lst.append((obs_id, filtered_features))

		idx += 1

	return feature_tuple_lst


def build(filepath, exclude_lst=[]):
	return read_data(filepath, exclude_lst)


def featureize(feature_tuple_lst, observation_ids):

	n = len(feature_tuple_lst)
	d = len(feature_tuple_lst[0][1])
	X = np.zeros((n,d), dtype=np.float)

	obs_list = []

	for idx, row in enumerate(X):
		X[idx] = feature_tuple_lst[idx][1]
		obs_list.append(feature_tuple_lst[idx][0])

	I = index_of(observation_ids, obs_list)

	return pick(np.zeros((len(I), X.shape[1])), X, I)

# if __name__ == "__main__":
# 	feature_tuple_lst = read_data("/Users/stuwagsmac/Desktop/nlp_final_project/cis530-project/data/LIWC/test_data_LIWC.dat", ['WC'])
# 	print featurize(feature_tuple_lst)
