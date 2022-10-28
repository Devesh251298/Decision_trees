import numpy as np
from decision_tree_classifier import DecisionTreeClassifier, parse_tree

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
    print(confusion_matric)