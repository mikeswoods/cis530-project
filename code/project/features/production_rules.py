# -*- coding: utf-8 -*-

import numpy as np
import re
from itertools import chain
from collections import Counter

from project.CoreNLP import all_sentences

################################################################################

class ParseNode(object):

    def __init__(self, name):
        self.name     = name
        self.children = []
        self.parent   = None
        self.terminal = False

    def add_child(self, child):
        self.children.append(child)
        child.set_parent(self)

    def set_parent(self, parent):
        self.parent = parent

    def get_path(self):
        path    = [self]
        current = self.parent
        while current is not None:
            path.insert(0, current)
            current = current.parent
        return path

    def __str__(self):
        return "{}(\"{}\", {})".format('T' if self.terminal else 'N' \
                                      ,self.name \
                                      ,len(self.children))

    def __repr__(self):
        return self.__str__()

class ParseTree(object):

    def __init__(self):
        self.root  = None
        self.nodes = []

    def tokenize_chunk(self, chunk):
        return filter(lambda s: len(s) > 0, re.split(r'([\(\)])', chunk, flags=re.I))

    def tokenize_rule(self, rule):
        return list(chain.from_iterable(map(self.tokenize_chunk, rule.split())))

    def parse(self, rule_string):
        tokens = self.tokenize_rule(rule_string)
        stack  = []

        self.root  = None
        self.nodes = []

        while len(tokens) > 0:

            head  = tokens.pop(0)
            empty = len(tokens) == 0 

            # Start of rule
            if head == "(":
                if len(tokens) > 0:
                    new_node  = tokens.pop(0)
                    tree_node = ParseNode(new_node)
                    self.nodes.append(tree_node)
                    stack.insert(0, tree_node)
            # End of rule
            elif head == ")":
                if len(stack) > 0 and len(tokens) > 0:
                    child  = stack.pop(0)
                    parent = stack[0]
                    parent.add_child(child)
            # Non-terminal contents of whatever node is on top of the stack
            else:
                if len(stack) > 0:
                    stack[0].terminal = True

        self.root = stack[0]
        return self

    def get_production_rules(self):
        return ['_'.join([node.name] + map(lambda c: c.name, node.children)) \
                for node in self.nodes if not node.terminal]

    def __traverse(self, node, depth):
        print "{}{}".format('  ' * depth, node)
        for child in node.children:
            self.__traverse(child, depth+1)

    def traverse(self):
        if self.root is not None:
            self.__traverse(self.root, 0)
        else:
            print "*** Tree is empty ***"
        return self


def to_production_rules(parse):

    return ParseTree().parse(parse).get_production_rules()

################################################################################

def test():

    S      = all_sentences('train')
    parses = (sent_data['parse'] for sentences in S.values() for sent_data in sentences)
    rules  = set(chain.from_iterable(map(to_production_rules, parses)))

    rules_index = {rule: i for (i,rule) in enumerate(rules, start=0)}

    print rules_index


def build(CoreNLP_data):

    parses      = (sent_data['parse'] for sentences in CoreNLP_data.values() for sent_data in sentences)
    rules       = set(chain.from_iterable(map(to_production_rules, parses)))
    rules_index = {rule: i for (i,rule) in enumerate(rules, start=0)}

    return (rules_index,)


def featureize(F, observation_ids, CoreNLP_data, binary=False):
    """
    If binary = True, the X[i][j] position will be set to 1 if some production
    rule in the observation appears in the producntion word list at position j, 
    otherwise 0

    If binary = False, the X[i][j] position will be set to 
    (rule count / number of rules in the observation), otherwise 0.0
    """
    (rules_index,) = F

    n = len(rules_index)
    m = len(observation_ids)

     # Observations
    X = np.zeros((m,n), dtype=np.float)

    for (i,ob_id) in enumerate(observation_ids, start=0):

        assert(ob_id in CoreNLP_data)
        
        observation_rules = list(chain(*[to_production_rules(sentence['parse']) for sentence in CoreNLP_data[ob_id]]))

        N = len(observation_rules)

        for rule in observation_rules:
            
            if rule in rules_index:

                if binary:
                    X[i][rules_index[rule]] = 1
                else:
                    X[i][rules_index[rule]] += 1.0

        if not binary:
            # Normalize by the number of tokens in each observation
            for j in range(0, N):
                X[i][j] /= float(N)

    return X



