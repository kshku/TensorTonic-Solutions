import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    classes, frequencies = np.unique(y, return_counts=True)
    total = len(y)
    entropy = 0
    for freq in frequencies:
        pi = freq / total
        entropy += pi * np.log2(pi)
    return float(-entropy)