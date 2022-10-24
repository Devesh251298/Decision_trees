from find_split import find_split
from evaluation_metrics import get_confusion_matrix, get_accuracy, get_precision, get_recall, get_f1_score
import numpy as np
from sklearn import tree

class DecisionTree:
    def __init__(self, attribute=0, value=-1, left=None, right=None, depth=-1, leaf=False, label=None):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.depth = depth
        self.leaf = leaf
        self.label = label

    def __repr__(self):
        return f"DecisionTree(Attr : {self.attribute}, Value : {self.value}, Depth : {self.depth}, Label : {self.label})"

class DecisionTreeClassifier():
    def __init__(self):
        self.dtree = None
        self.depth = 0

    def fit(self, dataset):
        self.dtree, self.depth = self.decision_tree_learning(dataset,0)
    
    def decision_tree_learning(self, training_dataset, depth):
        output = np.unique(training_dataset[:,-1])
        if output.shape[0]==1:
            return DecisionTree(leaf = True, label = output[0], depth = depth), depth

        split_attribute, split_value, split_left_dataset, split_right_dataset = find_split(training_dataset)
        dtree = DecisionTree(attribute = split_attribute, value = split_value, depth = depth)
        dtree.left, left_depth = self.decision_tree_learning(split_left_dataset, depth+1)
        dtree.right, right_depth = self.decision_tree_learning(split_right_dataset, depth+1)

        return dtree, max(left_depth, right_depth)

    def predict(self, y_test):
        output = []
        for i in range(y_test.shape[0]):
            tree = self.dtree
            while tree.leaf!=True:
                attr = tree.attribute
                val = tree.value
                if y_test[i][attr] >= val:
                    tree = tree.right
                else:
                    tree = tree.left
            output.append(tree.label)
        return output, y_test[:,-1]

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
    confusion_matrix = get_confusion_matrix(y_test, y_pred)
    accuracy = get_accuracy(confusion_matrix)
    precision = get_precision(confusion_matrix)
    recall = get_recall(confusion_matrix)
    f1_score = get_f1_score(precision, recall)
    return confusion_matrix, accuracy, precision, recall, f1_score

def get_list_preleaf_nodes(classifier):
    pass

def pruning(classifier, test_dataset):
    """ Prune the classifier to maximize accuracy.
    
    Algorithm:
    
    while True:
        accuracy = get_accuracy(classifier, y_test, y_pred)
        pre_leaf_nodes = get_list_of_preleaf_nodes(classifier)
        max_accuracy = accuracy
        for node in pre_leaf_nodes:
            new_classifier = prune(node, classifier)
            new_accuracy = get_accuracy(new_classifier)
            if new_accuracy > accuracy:
            return classifier
        else:
            classifier <- pruned classifier
            continue
    
    
    Args:
        classifier (DecisionTreeClassifier)
        X_test
        y_test
    
    Returns:
        pruned_classifier (DecisionTreeClassifier)
        evaluation_metrics: dictionary with keys:
            confusion_matrix
            accuracy
            precision
            recall
            f1_score    
    """
    return None

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
    np.random.shuffle(dataset)
    batches = np.split(dataset, k)
    confusion_matrix, accuracy, recall, precision, f1_measure = [], [], [], [], []
   

    for i in range(k):
        d = None
        if k!=1:
            for j in range(k):
                if i==j:
                    continue
                if not isinstance(d,np.ndarray):
                    d = batches[j]
                else:
                    d = np.append(d,batches[j],axis=0)

        else:
            d = dataset

        classifier = DecisionTreeClassifier()
        classifier.fit(d)
        c, a, p, r, f = evaluate(classifier, batches[i][:,:-1], batches[i][:,-1])
        output, actual = classifier.predict(batches[i])
        count = 0


        for i1 in range(len(output)):
            if output[i1] == actual[i1]:
                count+=1

        print(f"Accuracy for Batch {i+1} = ", (count*100)/len(output))
        confusion_matrix.append(c)
        accuracy.append(a)
        recall.append(r)
        precision.append(p)
        f1_measure.append(f)
        
        pruned_metrics = pruning(classifier, batches[i])

    confusion_matrix = np.average(np.array(confusion_matrix), axis=0)
    accuracy  = np.average(np.array(accuracy))
    recall  = np.average(np.array(recall))
    precision  = np.average(np.array(precision))
    f1_measure  = np.average(np.array(f1_measure))

    return confusion_matrix, accuracy, recall, precision, f1_measure


def test_decision_tree():
    dataset = np.loadtxt("wifi_db/clean_dataset.txt", dtype=float)
    dtree = DecisionTreeClassifier()
    dtree.fit(dataset)
    print(dtree.depth)
    # parse_tree(dtree.dtree)
    output, actual = dtree.predict(dataset)
    print(len(output), len(actual))

    count = 0
    for i in range(len(output)):
        if output[i] == actual[i]:
            count+=1
    print("Overall Accuracy = ", (count*100)/len(output))

    print(cross_validation(dataset, 10))


    #print(f"attribute: {attribute}, value: {value}, left dataset: {left_dataset.shape}, right dataset: {right_dataset.shape}")

def parse_tree(node):
    if node.left!=None:
        parse_tree(node.left)
    if node.right!=None:
        parse_tree(node.right)
    print(node)

if __name__ == "__main__":
    test_decision_tree()

