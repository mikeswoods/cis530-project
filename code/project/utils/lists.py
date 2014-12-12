
################################################################################
# List processing functions go here
################################################################################

def index_of(search_list, index_list):
    """
    Example:

    X = ["B", "F", "G"]
    Y  = ["A", "B", "C", "D", "E", "F", "G"]

    index_of(X, Y) => [1, 6, 7]    
    """
    I = {k:i for (i,k) in enumerate(index_list)}
    return [I[j] for j in search_list if j in I]


def pick(Y, X, I):
    """
    Picks rows from X into Y, such that for i [0..n], Y[i] = X[I[i]]
    """
    assert(len(I) == len(Y) and len(X) >= len(Y))

    for (i,j) in enumerate(I):
        Y[i] = X[j]

    return Y
