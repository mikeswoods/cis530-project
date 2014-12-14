#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from logging import debug, info, warn
import argparse
import sys

from project import resources
from project import CoreNLP
from project import classify
from project import features

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--submit', dest='submit', action='store_const', const=True, default=False, help="Generate a submission")
parser.add_argument('--iters', dest='iterations', action='store', type=int, default=1, help="Run N test iterations, averaging the results")
parser.add_argument('-T', dest='test_size', action='store', type=float, default=0.1, help="xval test size (default = 0.1")
parser.add_argument('-F', dest='test_feature', action='store', type=str, default=None, help="Call the feature's test() function")
parser.add_argument('--recache', dest='recache', action='store_const', const=True, default=False, help="Regenerate cache data")
parser.add_argument('model', nargs=1, help="The model to run")
args = parser.parse_args()

################################################################################

logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":

    if args.recache:
        CoreNLP.regenerate_cache()

    if args.test_feature is not None:
        features.test(args.test_feature)
        sys.exit(0)

    run_model = args.model[0]

    info(">> Running model \"{}\" (test_size={})".format(run_model, args.test_size))

    if args.submit:
        resources.SUBMIT_MODE = True
        classify.make_submission(run_model)
    else:
        if args.iterations > 1:
            classify.test_iterations(run_model, args.iterations, test_size=args.test_size)
        else:
            classify.test(run_model, test_size=args.test_size)
