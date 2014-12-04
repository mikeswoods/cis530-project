#!/usr/bin/env python

from nltk.tokenize.punkt import PunktWordTokenizer

import logging
from logging import debug, info, warn

import os.path

from project import resources
from project import text
from project.utils import command
from project.utils.files import resolve

################################################################################

logging.basicConfig(level=logging.DEBUG)

################################################################################

USE_ANNOTATORS = ['tokenize','ssplit', 'pos', 'lemma', 'ner', 'sentiment', 'depparse', 'relation']

def create_CoreNLP_trainxml():
	"""
	Generates CoreNLP output for the training data
	"""
	input_files = resources.train_data_files()
	command.run_corenlp(resolve('~', 'corenlp')
		               ,input_files
		               ,resolve('..', 'data', 'CoreNLP', 'train_data')
		               ,annotators=USE_ANNOTATORS)

def create_CoreNLP_test_xml():
	"""
	Generates CoreNLP output for the training data
	"""
	input_files = resources.test_data_files()
	command.run_corenlp(resolve('~', 'corenlp')
		               ,input_files
		               ,resolve('..', 'data', 'CoreNLP', 'test_data')
		               ,annotators=USE_ANNOTATORS)

################################################################################
# Main
################################################################################

if __name__ == "__main__":

	create_CoreNLP_trainxml()
	create_CoreNLP_test_xml()
