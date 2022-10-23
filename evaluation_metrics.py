import numpy as np
#from decision_tree_classifier import DecisionTreeClassifier, parse_tree

def get_confusion_matrix(actual, predicted):
    # extract the different classes
    classes = np.unique(actual)
    
    # size of matrix
    N = len(classes)

    # initialize the (N x N) confusion matrix
    confusion_matrix = np.zeros((N, N))

    # loop across the different combinations of actual / predicted classes
    for i in range(N):
        for j in range(N):
           # count the number of instances in each combination of actual / predicted classes
           confusion_matrix[i, j] = np.sum((actual == classes[i]) & (predicted == classes[j]))

    return confusion_matrix


def get_accuracy(conf_matrix):
    """ Get accuracy by class from confusion matrix."""
    # compute the total number of instances in each class
    actual_num = conf_matrix.sum()
    # compute the total number of correct predicitons per class
    correct_num = np.diagonal(conf_matrix).sum()
    return correct_num / actual_num

def get_precision(conf_matrix):
    """ Get precision by class from confusion matrix."""
    # compute the total number of instances in each class
    actual_num_per_class = conf_matrix.sum(axis=1)
    # compute the total number of correct predicitons per class
    correct_num_per_class = np.diagonal(conf_matrix)
    return correct_num_per_class / actual_num_per_class

def get_recall(conf_matrix):
    """ Get recall by class from confusion matrix."""
    # compute the total number of instances in each class
    pred_num_per_class = conf_matrix.sum(axis=0)
    # compute the total number of correct predicitons per class
    correct_num_per_class = np.diagonal(conf_matrix)
    return correct_num_per_class / pred_num_per_class

def get_f1_score(precision, recall):
    """ Get f1-score by class from precision and recall metrics."""
    return 2* (precision*recall) / (precision+recall)



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
