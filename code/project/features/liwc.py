import numpy as np
from itertools import *

# ['Seg', 'WC', 'WPS', 'Sixltr', 'Dic', 'funct', 'pronoun', 'ppron', 'i', 'we', 'you', 'shehe', 'they', 'ipron', 'article', 'verb', 'auxverb', 'past', 'present', 'future', 'adverb', 'preps', 'conj', 'negate', 'quant', 'number', 'swear', 'social', 'family', 'friend', 'humans', 'affect', 'posemo', 'negemo', 'anx', 'anger', 'sad', 'cogmech', 'insight', 'cause', 'discrep', 'tentat', 'certain', 'inhib', 'incl', 'excl', 'percept', 'see', 'hear', 'feel', 'bio', 'body', 'health', 'sexual', 'ingest', 'relativ', 'motion', 'space', 'time', 'work', 'achieve', 'leisure', 'home', 'money', 'relig', 'death', 'assent', 'nonfl', 'filler', 'Period', 'Comma', 'Colon', 'SemiC', 'QMark', 'Exclam', 'Dash', 'Quote', 'Apostro', 'Parenth', 'OtherP', 'AllPct']
# http://www.liwc.net/descriptiontable1.php

def read_data(filepath, exclude_lst):
	f = open(filepath, 'r')
	exclude_idx_lst = [False] * 81 #81 is number of features in liwc
	feature_tuple_lst = []
	for idx, line in enumerate(f):
		if idx == 0: #skip line with labels
			labels = line.split()[1:]
			for exclusion in exclude_lst:
				exclude_idx = labels.index(exclusion)
				exclude_idx_lst[exclude_idx] = True
		else:
			line = line.split()
			obs_id = line[0]
			features = [float(x) for x in line[1:]]
			filtered_features = list(compress(features, map(lambda x: not x, exclude_idx_lst)))
			feature_tuple_lst.append((obs_id, filtered_features))
	return feature_tuple_lst

def featurize(feature_tuple_lst):
	n = len(feature_tuple_lst)
	d = len(feature_tuple_lst[0][1])
	X = np.zeros((n,d), dtype=np.float)
	for idx, row in enumerate(X):
		X[idx] = feature_tuple_lst[idx][1]
	return X


if __name__ == "__main__":
	feature_tuple_lst = read_data("/Users/stuwagsmac/Desktop/nlp_final_project/cis530-project/data/LIWC/test_data_LIWC.dat", ['WC'])
	featurize(feature_tuple_lst)