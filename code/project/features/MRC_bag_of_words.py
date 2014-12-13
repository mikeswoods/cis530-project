# -*- coding: utf-8 -*-

import numpy as np

from itertools import chain
from collections import Counter

from project.utils.files import read_file
from project.utils.files import resolve
from project.utils.text import strip_junk_tokens

################################################################################
# MRC bag of words
# 
# From "Detecting Information-Dense Texts in Multiple News Domain"
# by Ani Nenkova & Yinfei Yang
#
# "MRC Database The MRC Psycholinguistic Database
# (Wilson ) is a machine usable dictionary containing 150,837
# words, different subsets of which are annotated for 26 linguistic
# and psycholinguistic attributes. We select a subset
# of 4,923 words normed for age of acquisition, imagery,
# concreteness, familiarity and ambiguity. We use the list of
# all normed words in a bag-of-words representations of the
# leads. This representation is appealing because the feature
# space is determined independently of the training data and
# is thus more general than the alternative lexical representation
# that we explore.
# The value of each feature is equal to the number of times it
# appeared in the lead divided by the number of words in the
# lead. We observe the words with higher familiarity scores
# scores, such as mother, money, watch are more characteristic
# for the non-informative texts, and appeared to be among the
# best indicators for the class of the lead."
################################################################################

def test():

    mrc_words_file = resolve('..', 'data', 'MRC', 'MRC_words')
    mrc_words  = [word.strip() for word in read_file(mrc_words_file).split() if word.strip() != '']
    mrc_words.sort()

    mrc_words_index = {word: i for (i,word) in enumerate(mrc_words)}


def build(mrc_words_file, all_tokens_dict):

    mrc_words  = [word.strip() for word in read_file(mrc_words_file).split() if word.strip() != '']
    mrc_words.sort()

    mrc_words_index = {word: i for (i,word) in enumerate(mrc_words)}

    return (all_tokens_dict, mrc_words_index)


def featureize(F, observation_ids, binary=False):
    """
    If binary = True, the X[i][j] position will be set to 1 if some word in the 
    observation appears in the MRC word list at position j, otherwise 0

    If binary = False, the X[i][j] position will be set to 
    (word count / number of tokens in the observation), otherwise 0.0
    """
    (all_tokens_dict, mrc_words_index) = F

    n = len(mrc_words_index)
    m = len(observation_ids)

     # Observations
    X = np.zeros((m,n), dtype=np.float)

    for (i,ob_id) in enumerate(observation_ids, start=0):

        N = len(all_tokens_dict[ob_id])

        for token in all_tokens_dict[ob_id]:

            if token in mrc_words_index:

                if binary:
                    X[i][mrc_words_index[token]] = 1
                else:    
                    X[i][mrc_words_index[token]] += 1.0

        if not binary:
            # Normalize by the number of tokens in each observation
            for j in range(0, N):
                X[i][j] /= float(N)

    return X
