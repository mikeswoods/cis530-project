#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from logging import debug, info, warn

import os.path
import sys

from project import resources
from project import text
from project import CoreNLP
from project import classify
from project.utils import command
from project.utils.files import resolve

################################################################################

logging.basicConfig(level=logging.DEBUG)

#CoreNLP.regenerate_cache()

if __name__ == "__main__":

    #classify.test('sample')
    #classify.test_iterations('sample', 50)

    # use_model = 'linear_svm'
    # #use_model = 'naive_bayes'
    # test_size = 0.125

    # if len(sys.argv) > 1:
    #     classify.test_iterations(use_model, int(sys.argv[1]), test_size=test_size)
    # else:
    #     classify.test(use_model, test_size=test_size)


    classify.make_submission('naive_bayes')
