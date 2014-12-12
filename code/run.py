#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    #classify.test('sample')
    #classify.test_iterations('sample', 50)

    classify.test('nb_binary_bag_of_words', test_size=0.125)
    #classify.test('svm_binary_bag_of_words', test_size=0.125)
    
