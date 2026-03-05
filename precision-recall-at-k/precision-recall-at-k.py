def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    relevant_recommends = len(set(recommended[:k]) & set(relevant))
    return [relevant_recommends / k, relevant_recommends / len(relevant)]