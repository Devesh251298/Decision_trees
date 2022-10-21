import numpy as np

def calculate_entropy(dataset):
    """Calculate entropy of a label distribution.

    The entropy, S, will be calculated as the sum of the probabilities times 
    the log of the probabilities, i.e. S = -sum(pk * log(pk), axis=axis).

    Args:
        dataset (np.ndarray)

    Returns:
        S: entropy (float)
    """
    
    # Get the target variable and its lenght 
    labels = dataset[:, -1]

    # Entropy will be zero if labels distribution is equal or less than 1 in length
    if len(dataset[:, -1]) <= 1:
        return 0

    # Number of ocurrences of a class k in the dataset
    count_k = np.unique(labels, return_counts=True)[1]
    
    # Compute the probability values for each class k
    probability_k = count_k / np.sum(count_k)

    n_classes = np.count_nonzero(probability_k)

    # Entropy will be zero if number of classes in the distribution is equal or less than 1 in length
    if n_classes <= 1:
        return 0

    # Compute entropy
    S = -np.sum(probability_k*np.log2(probability_k))

    return S