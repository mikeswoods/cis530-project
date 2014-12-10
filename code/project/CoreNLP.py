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
# CoreNLP related parsing
################################################################################

def parse_sentences(filename):
    """
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

    return sentences
