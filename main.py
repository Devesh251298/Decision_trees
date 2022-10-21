import numpy as np

def load_data(filepath):
    """Load data.

    Extended description of function.

    Args:
        filename (str): filepath

    Returns:
        dataset (np.ndarray(Nx8)): dataset

    """
    File_data = np.loadtxt(filepath, dtype=int)
    return File_data
    


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
    
   


def find_split(dataset):
    """Load data.

    Extended description of function.

    Args:
        filename (str): 

    Returns:
        attribute (int): integer going from 
        value (float): 
        left_dataset (np.ndarray, shape Nleft x 8)
        right_dataset (np.ndarray, shape Nright x 8)

    """



def decision_tree_learning(training_dataset, depth):
    """Load data.

    Extended description of function.

    Args:
        training_dataset (np.ndarray)
        depth (int)

    Returns:
        node (Node)
    """

class DecisionTree:
    def __init__(self, attribute, value, left, right, depth, leaf, label=None):
        """ Attributes:
            attribute (int): integer between 1 and 7
            value (float): value used to split this attribute
            left (Node): left node
            right (Node): right node
            depth (int): depth
            leaf (bool): whether it is a leaf or not
            label (int): room
        - 
        """


def get_confusion_matrix(y, y_pred):

def get_precision(y, y_pred):

def get_recall(y, y_pred):

def get_f1_score(y, y_pred):


def evaluate(classifier, X_test, y_test):
    """ Evaluate.
    
    Args:
        classifier (DecisionTreeClassifier)
        X_test (np.ndarray)
    Returns:
        confusion_matrix
        accuracy
        recall
        precision
        f1_measure
    """
    y_pred = classifier.predict(X_test)
    confusion_matrix = get_confusion_matrix(y_pred, y_test)


def cross_validation(dataset, k=10):
    """ Evaluate.
    
    1) Split dataset into 10 parts
    2) Loop with 10 iterations. For each iteration:
        - train decision tree with training dataset
        - get evaluation metrics for the test dataset (get confusio
    
    Args:
        dataset (np.ndarray) 
    Returns:
        confusion_matrix
        accuracy
        recall
        precision
        f1_measure
    """


class DecisionTreeClassifier():
    def __init__(self):
        self.decision_tree = None
    
    def decision_tree_learning(training_dataset, depth):
        """Load data.

        Extended description of function.

        Args:
            training_dataset (np.ndarray)
            depth (int)

        Returns:
            decision_tree (DecisionTree)
        """
        pass
        
    def fit(self, X_train, y_train):
        self.decision_tree = self.decision_tree_learning(..)
        pass
    
    def predict(self, X_test):
        """Predict labels according to the decision tree model.

        Args:
            X_test (np.ndarray)

        Returns:
            y_pred (np.ndarray)
        """
        pass

    def prune(self):
        pass
