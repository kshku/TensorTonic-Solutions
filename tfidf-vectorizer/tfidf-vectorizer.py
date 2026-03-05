import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    documents = [doc.lower().split() for doc in documents]
    total_len = sum(len(doc) for doc in documents)
    vocab = set()
    for doc in documents:
        vocab |= set(doc)
    vocab = sorted(list(vocab))
    
    freq = dict()
    for term in vocab:
        for doc in documents:
            if term in doc:
                freq[term] = freq.get(term, 0) + 1

    # matrix = [[(doc.count(term) / len(doc)) * (np.log(len(documents) / freq[term])) for term in vocab] for doc in documents]
    matrix = np.zeros((len(documents), len(vocab)))
    for i, doc in enumerate(documents):
        counter = Counter(doc)
        for j, term in enumerate(vocab):
            matrix[i][j] = (counter[term] / len(doc)) * (math.log(len(documents) / freq[term]))

    return (matrix, vocab)