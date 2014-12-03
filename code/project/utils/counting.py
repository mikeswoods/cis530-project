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
    if total is None:
        total = float(sum(counter.values()))

    return dict([(word, float(count) / total) for (word, count) in counter.items()])
