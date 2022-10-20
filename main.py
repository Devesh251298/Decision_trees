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
    """Calculate entropy of dataset.

    Extended description of function.

    Args:
        dataset (np.ndarray): 

    Returns:
        entropy (float)
    """
    return 0


def find_split(dataset):
    """Find split that maximizes information gain in dataset.

    Args:
        dataset (np.ndarray, size N x 8): 
            dataset[:,:7] -> attributes (float)
            dataset[:,7] -> label (int from 1 to 4)
        

    Returns:
        split_attribute (int): integer from 0 to 6 (both included)
        split_value (float): value used to split the chosen split_attribute
        split_left_dataset (np.ndarray, shape Nleft x 8)
        split_right_dataset (np.ndarray, shape (N-Nleft) x 8)
    """
    # get the number of attributes in dataset (= 7)
    N, num_attributes = dataset.shape[0], dataset.shape[1] - 1
    # calculate the entropy of the entire dataset
    dataset_entropy = calculate_entropy(dataset)
    
    # initialize the target variables defining split
    max_info_gain = -1
    
    # iterate over all attributes to find the splitting attribute
    for attribute in range(num_attributes):
        # get values of that attribute
        values = dataset[:, attribute]
        
        # get indices of the dataset sorted according to that attribute
        sorted_indices = np.argsort(values)
        
        # iterate from smallest to largest value and find optimal splitting point
        for i, index in enumerate(sorted_indices):
            value = dataset[index, attribute]
            
            # split dataset on the condition
            # left_dataset -> attribute <= value
            # right_dataset ->  attribute > value
            left_dataset = dataset[sorted_indices[:i+1]]
            right_dataset = dataset[sorted_indices[i+1:]]

            # calculate left and right entropies
            left_entropy = calculate_entropy(left_dataset)
            right_entropy = calculate_entropy(right_dataset)
            
            # calculate information gain
            info_gain = dataset_entropy - (i/N * left_entropy + (N-i)-N * right_entropy)
            
            # if information gain is larger than the current maximum, update variables
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                split_attribute, split_value = attribute, value
                split_left_dataset = left_dataset
                split_right_dataset = right_dataset
    
    return split_attribute, split_value, split_left_dataset, split_right_dataset


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
