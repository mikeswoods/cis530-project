import math

################################################################################
# Probability and counting related utility functions
################################################################################

def n_choose_k(n,k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))


def counts_to_probs(counter, total=None):
    """
    Counter({str: int}) -> {str: float[0,1]}
    """
    if len(counter) == 0:
    	return {}

    if total is None:
        total = float(sum(counter.values()))

    P = dict([(word, float(count) / total) for (word, count) in counter.items()])

    assert(abs(1.0 - sum(P.values())) < 1.0e-8)

    return P
