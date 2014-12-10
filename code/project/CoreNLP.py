import re
import sys
from itertools import chain
from contextlib import contextmanager
try:
   import cPickle as pickle
except:
   import pickle
from logging import debug, info, warn, error
from collections import Counter
from itertools import chain, product, groupby, permutations, combinations
try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET
from os.path import exists

from project import resources
from project.text import filename_to_id
from project.utils.files import resolve

################################################################################
# Generate CoreNLP output
################################################################################


USE_ANNOTATORS = ['tokenize','ssplit', 'pos', 'lemma', 'ner', 'parse', 'sentiment', 'relation']

def _create_CoreNLP_trainxml():
    """
    Generates CoreNLP output for the training data (run once)
    """
    input_files = resources.train_data_files()
    command.run_corenlp(resolve('~', 'corenlp')
                       ,input_files
                       ,resolve('..', 'data', 'CoreNLP', 'train_data')
                       ,annotators=USE_ANNOTATORS)


def _create_CoreNLP_test_xml():
    """
    Generates CoreNLP output for the training data (run once)
    """
    input_files = resources.test_data_files()
    command.run_corenlp(resolve('~', 'corenlp')
                       ,input_files
                       ,resolve('..', 'data', 'CoreNLP', 'test_data')
                       ,annotators=USE_ANNOTATORS)


################################################################################


def all_sentences():
    """
    Returns a dict of all sentences data derived from CoreNLP. The key
    is the truncated filename (observation-ID), and the value is the output
    sentence data generated by parse_sentences() for that particular file.

    @returns: {str: <sentence-data>}
    """
    corenlp_bin_data = resolve(resources.CoreNLP_base, 'sentences.bin')
    
    # If there's cached data, load it:
    if exists(corenlp_bin_data):

        debug('> Loading cached CoreNLP data from {}'.format(corenlp_bin_data))

        with open(corenlp_bin_data, 'r') as f:
            return pickle.load(f)

    # Otherwise, generate the output from parse_sentences()
    debug('> CoreNLP data {} not found; caching...'.format(corenlp_bin_data))

    filenames = resources.train_data_files('CoreNLP')
    # parse_sentences(filename)[1] means to only keep the actual sentence data,
    # not the file name/observation identifier
    data      = {filename_to_id(filename): parse_sentences(filename)[1] for filename in filenames}

    with open(corenlp_bin_data, 'w') as f:
        pickle.dump(data, f)

    debug('> CoreNLP data cached to {}'.format(corenlp_bin_data))

    return data


################################################################################
# CoreNLP related parsing
################################################################################

def parse_sentences(filename):
    """
    Parses a CoreNLP output sentence, returning a tuple of 
    (str:observation-id, [dict:<sentence-data>]) where sentences is a list of 
    dicts, where each <sentence-data> dict has the following keys:

    - 'tokens' : [{'word':str, 'lemma':str, 'POS':str, 'NER':str|None}]
    - 'sentiment' : str
    - 'parse' : str

    @returns (str:observation-id, [dict:<sentence-data>])
    """
    sentences = []
    tree = ET.parse(filename)

    for s in tree.findall('.//sentence'):

        sentence = {
             'tokens': []
            ,'sentiment': s.get('sentiment').lower()
            ,'parse': s.find('.//parse').text
        }

        for t in s.findall('.//token'):

            word    = t.find('word').text.lower()
            lemma   = t.find('lemma').text.lower()
            pos_tag = t.find('POS').text
            ner_tag = t.find('NER').text
            if ner_tag == "O":
                ner_tag = None

            data = {'word':word, 'lemma':lemma, 'pos':pos_tag, 'ner':ner_tag}
            sentence['tokens'].append(data)

        sentences.append(sentence)

    return (filename_to_id(filename), sentences)
