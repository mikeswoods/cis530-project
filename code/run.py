#!/usr/bin/env python

from nltk.tokenize.punkt import PunktWordTokenizer

import logging
from logging import debug, info, warn

import os.path

from project import resources
from project import text
from project import CoreNLP
from project.utils import command
from project.utils.files import resolve

################################################################################

logging.basicConfig(level=logging.DEBUG)

################################################################################


################################################################################
# Main
################################################################################

if __name__ == "__main__":

	xml_file = '/Users/mike/src/school/cis530-project/data/CoreNLP/train_data/2006_12_29_1815122.txt.xml'
	CoreNLP.parse_sentences(xml_file)
	
	