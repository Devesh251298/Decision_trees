import numpy as np
#from decision_tree_classifier import DecisionTreeClassifier, parse_tree


def evaluate(classifier, test_dataset):
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
    X_test, y_test = test_dataset[:,:-1], test_dataset[:,-1]
    y_pred = classifier.predict(X_test)
    confusion_matrix = get_confusion_matrix(y_test, y_pred)
    accuracy = get_accuracy(confusion_matrix)
    precision = get_precision(confusion_matrix)
    recall = get_recall(confusion_matrix)
    f1_score = get_f1_score(precision, recall)
    return confusion_matrix, accuracy, precision, recall, f1_score


def get_confusion_matrix(actual, predicted):
    # Extract the different classes
    classes = np.unique(actual)
    
    # Size of confusion matrix 
    N = len(classes)

    # Initialize the (N x N) confusion matrix
    confusion_matrix = np.zeros((N, N))

    # Loop through all combination of actual and predicted classes
    for i in range(N):
        for j in range(N):
           # Count the number of instances in each combination of actual and predicted classes
           confusion_matrix[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))
  
    return confusion_matrix


def get_accuracy(conf_matrix):
    """ Get accuracy by class from confusion matrix."""
    # Compute the total number of instances in each class
    actual_num = conf_matrix.sum()
    # Compute the total number of correct predicitons per class
    correct_num = np.diagonal(conf_matrix).sum()
    return correct_num / actual_num


def get_precision(conf_matrix):
    """ Get precision by class from confusion matrix."""
    # Compute the total number of instances in each class
    actual_num_per_class = conf_matrix.sum(axis=1)
    # Compute the total number of correct predicitons per class
    correct_num_per_class = np.diagonal(conf_matrix)
    return correct_num_per_class / actual_num_per_class


def get_recall(conf_matrix):
    """ Get recall by class from confusion matrix."""
    # Compute the total number of instances in each class
    pred_num_per_class = conf_matrix.sum(axis=0)
    # Compute the total number of correct predicitons per class
    correct_num_per_class = np.diagonal(conf_matrix)
    return correct_num_per_class / pred_num_per_class


def get_f1_score(precision, recall):
    """ Get f1-score by class from precision and recall metrics."""
    return 2* (precision*recall) / (precision+recall)


class Metrics:
    def __init__(self):
        self.metric_dict = {
            'confusion_matrix': [],
            'precision': [],
            'recall': [],
            'accuracy': [],
            'f1_score': [],
            'depth': []
        }
    
    def add_metrics(self, conf, acc, prec, rec, f1, dep):
        """ Adds values to the metric dictionary."""
        self.metric_dict['confusion_matrix'].append(conf)
        self.metric_dict['precision'].append(prec)
        self.metric_dict['recall'].append(rec)
        self.metric_dict['accuracy'].append(acc)
        self.metric_dict['f1_score'].append(f1)
        self.metric_dict['depth'].append(dep)
    
    def get_avg_metrics(self):
        """ Return avg metrics dictionary."""
        avg_metrics = {}
        avg_metrics['confusion_matrix'] = np.average(
            np.array(self.metric_dict['confusion_matrix']), axis=0)
        avg_metrics['accuracy']  = np.average(
            np.array(self.metric_dict['accuracy']), axis=0)
        avg_metrics['recall']  = np.average(
            np.array(self.metric_dict['recall']), axis=0)
        avg_metrics['precision']  = np.average(
            np.array(self.metric_dict['precision']), axis=0)
        avg_metrics['f1_score']  = np.average(
            np.array(self.metric_dict['f1_score']), axis=0)
        avg_metrics['depth']  = np.average(
            np.array(self.metric_dict['depth']), axis=0)
        
        return avg_metrics


def test_get_confusion_matrix():
    dataset = np.loadtxt("wifi_db/clean_dataset.txt", dtype=float)
    dtree = DecisionTreeClassifier()
    dtree.fit(dataset)
    parse_tree(dtree.dtree)
    output, actual = dtree.predict(dataset)
    return get_confusion_matrix(actual, output)


if __name__ == "__main__":
    # test_evaluate()
    confusion_matric = test_get_confusion_matrix()
    precision = get_precision(confusion_matric)
    recall = get_recall(confusion_matric)
    f1_score = get_f1_score(precision, recall)

    print(confusion_matric)
    print(precision)
    print(recall)
    print(f1_score)
