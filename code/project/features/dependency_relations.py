# -*- coding: utf-8 -*-

import numpy as np
import re
from itertools import chain
from collections import Counter

from project.CoreNLP import all_sentences

################################################################################

def test():

    CoreNLP_data = all_sentences('train')
    dependencies = (sent_data['dependencies'] for sentences in CoreNLP_data.values() for sent_data in sentences)
    counts       = Counter(chain.from_iterable(dependencies))
    
    top_deps  = map(lambda (t,_): t, counts.most_common(5000))
    dep_index = {dep: i for (i,dep) in enumerate(top_deps, start=0)}

    print dep_index


def build(CoreNLP_data):

    dependencies = (sent_data['dependencies'] for sentences in CoreNLP_data.values() for sent_data in sentences)
    counts       = Counter(chain.from_iterable(dependencies))
    
    top_deps  = map(lambda (t,_): t, counts.most_common(5000))
    dep_index = {dep: i for (i,dep) in enumerate(top_deps, start=0)}

    return (dep_index,)


def featureize(F, observation_ids, CoreNLP_data, binary=True):

    (dep_index,) = F

    n = len(dep_index)
    m = len(observation_ids)

     # Observations
    X = np.zeros((m,n), dtype=np.float)

    for (i,ob_id) in enumerate(observation_ids, start=0):

        assert(ob_id in CoreNLP_data)
        
        dependencies = list(chain.from_iterable([sentence['dependencies'] for sentence in CoreNLP_data[ob_id]]))

        N = len(dependencies)

        for dep in dependencies:
            
            if dep in dep_index:

                if binary:
                    X[i][dep_index[dep]] = 1
                else:
                    X[i][dep_index[dep]] += 1.0

        if not binary:
            # Normalize by the number of tokens in each observation
            for j in range(0, N):
                X[i][j] /= float(N)

    return X



