#!/usr/bin/env python

from nltk.tokenize.punkt import PunktWordTokenizer

from logging import debug, info, warn

from project import resources
from project import text

################################################################################
# Main
################################################################################

if __name__ == "__main__":

    article = text.read_file('train', 89)

    tokenizer = PunktWordTokenizer()
    print tokenizer.tokenize(article)