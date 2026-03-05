import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    if sum(p) != 1:
        raise ValueError("Invalid probabilities")
    return sum(xi * pi for xi, pi in zip(x, p))
