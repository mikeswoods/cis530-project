#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from logging import debug, info, warn
import argparse
import os.path
import sys

from project import resources
from project import text
from project import CoreNLP
from project import classify
from project.utils import command
from project.utils.files import resolve

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--submit', dest='submit', action='store_const', const=True, default=False, help="Generate a submission")
parser.add_argument('--iters', dest='iterations', action='store', type=int, default=1, help="Run N test iterations, averaging the results")
parser.add_argument('-T', dest='test_size', action='store', type=float, default=0.1, help="xval test size (default = 0.1")
parser.add_argument('--recache', dest='recache', action='store_const', const=True, default=False, help="Regenerate cache data")
parser.add_argument('model', nargs=1, help="The model to run")
args = parser.parse_args()

################################################################################

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":

    if args.recache:
        CoreNLP.regenerate_cache()

    run_model = args.model[0]

    info(">> Running model \"{}\" (test_size={})".format(run_model, args.test_size))

    if args.submit:
        classify.make_submission(run_model)
    else:
        if args.iterations > 1:
            classify.test_iterations(run_model, args.iterations, test_size=args.test_size)
        else:
            classify.test(run_model, test_size=args.test_size)
