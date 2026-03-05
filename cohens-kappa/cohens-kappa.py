import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    # Write code here
    rater1 = np.array(rater1)
    rater2 = np.array(rater2)
    n = len(rater1)
    
    po = sum(1 for r1, r2 in zip(rater1, rater2) if r1 == r2) / n

    classes1, frequencies1 = np.unique(rater1, return_counts=True)
    classes2, frequencies2 = np.unique(rater2, return_counts=True)

    # Hoping frequencies are in order for both raters
    pe = sum((freq1 * freq2) / (n ** 2) for freq1, freq2 in zip(frequencies1, frequencies2))

    return float((po - pe) / (1 - pe) if pe != 1 else 1)