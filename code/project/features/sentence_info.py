# -*- coding: utf-8 -*-

import numpy as np

from itertools import chain
from collections import Counter

from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from project.text import strip_non_words
from project.utils.files import read_file

################################################################################
#
# Generates sentence features without CoreNLP, meaning it will work with any
# file
#
# Computes 2 features: 
# 1 (0) - Number of sentences per file
# 2 (1) - Number of tokens per file
#
################################################################################

def featureize(F, observation_files):

    word_tokenizer = PunktSentenceTokenizer()
    sent_tokenizer = PunktSentenceTokenizer()

    m = len(observation_files)

    # X is Nx2
    X = np.zeros((m,2), dtype=np.float)

    for (i,filename) in enumerate(observation_files,start=0):

        file_text  = read_file(filename).decode('string_escape')

        try:
            num_sents = len(sent_tokenizer.sentences_from_text(file_text))
        except UnicodeDecodeError:
            num_sents = 2

        #num_tokens = len(word_tokenize(file_text))
        num_tokens = len(file_text.split())

        # Return two features: 
        # 1 (0) - Number of sentences per file
        # 2 (1) - Number of tokens per file
        X[i][0] = num_sents
        X[i][1] = num_tokens

    return X
