#!/usr/bin/env python

import logging
from logging import debug, info, warn

import os.path

from project import resources
from project import text
from project import CoreNLP
from project import classify
from project.utils import command
from project.utils.files import resolve

################################################################################

logging.basicConfig(level=logging.DEBUG)

################################################################################


################################################################################
# Main
################################################################################

if __name__ == "__main__":

    #xml_file = '/Users/mike/src/school/cis530-project/data/CoreNLP/train_data/2006_12_29_1815122.txt.xml'
    #S = CoreNLP.parse_sentences(xml_file)

    print CoreNLP.all_sentences()['2006_12_13_1811545.txt.xml']

    #train_labels =  dict(resources.train_data_labels().items()[1:10])

    #(kept, held, fold) = classify.nfold_xval(train_labels, n=5)
    #print kept
    #print held

    pass