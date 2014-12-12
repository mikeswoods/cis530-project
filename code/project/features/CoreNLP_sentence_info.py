# -*- coding: utf-8 -*-

import numpy as np

from itertools import chain
from collections import Counter

from project.CoreNLP import all_sentences

from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from project.text import strip_non_words
from project.utils.files import read_file
from project.text import filename_to_id

################################################################################
#
# Generates sentence features with CoreNLP. It is assumed that all observation
# files featurized will have been processed by CoreNLP earlier
#
# Computes 4 features: 
# 1 (0) - Number of sentences per file
# 2 (1) - Number of tokens per file
# 3 (2) - Number of NERs per file
# 4 (3) - Number of nouns per file
# 5 (4) - Number of words over 6 letters
################################################################################

def featureize(F, observation_files, CoreNLP_data):

    m = len(observation_files)

    # Observations
    X = np.zeros((m, 5), dtype=np.int)

    for (i,filename) in enumerate(observation_files,start=0):

        # Convert the filename to an observation ID
        ob_id = filename_to_id(filename)
        assert(ob_id in CoreNLP_data)

        sent_data   = CoreNLP_data[ob_id]
        sent_count  = len(sent_data)
        token_count = 0
        ner_count   = 0
        noun_count  = 0
        over6_count = 0

        # Token data is a dict of the form:
        #   {'lemma': 'count', 'ner': None, 'pos': 'NNS', 'word': 'counts'}
        #     - or -
        #   {'lemma': 'i.b.m.', 'ner': 'ORGANIZATION', 'pos': 'NNP', 'word': 'i.b.m.'}
        for token_data in chain(*[sd['tokens'] for sd in sent_data]):
             
            if len(token_data['word']) > 6:
                over6_count += 1

            if token_data['ner'] is not None:
                ner_count += 1

            # Count the number of nouns:
            # - NN Noun, singular or mass
            # - NNS Noun, plural
            # - NNP Proper noun, singular
            # - NNPS Proper noun, plural
            pos = token_data['pos']
            if pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS':
                noun_count += 1

            token_count += 1

        X[i][0] = sent_count
        X[i][1] = token_count
        X[i][2] = ner_count
        X[i][3] = noun_count
        X[i][4] = over6_count

    return X
