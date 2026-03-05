import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    numerator = np.dot(a, b)
    denomenator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(numerator / denomenator if denomenator != 0 else 0)
