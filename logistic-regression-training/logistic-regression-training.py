import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    w = np.zeros(len(X[0]))
    b = 0.0
    N = len(X)

    for i in range(steps):
        # p = _sigmoid(X @ w + b)
        p = _sigmoid(X.dot(w) + b)
        loss = (-1 / N) * np.sum(y * np.log2(p) + (1 - y) * np.log2(1 - p))
        grad_w = X.T.dot(p - y) / N
        grad_b = np.sum(p - y) / N
        w -= lr * grad_w
        b -= lr * grad_b

    return (w, b)
    
    