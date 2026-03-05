import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    """
    predictions = np.array(predictions)
    predictions = predictions.T
    ans = []
    for sample_preds in predictions:
        classes, freqs = np.unique(sample_preds, return_counts=True)
        ans.append(classes[freqs.argmax()])
    return ans