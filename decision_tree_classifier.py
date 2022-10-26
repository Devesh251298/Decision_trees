from find_split import find_split
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

class DecisionTree:
    def __init__(self, attribute=0, value=-1, left=None, right=None, 
                    depth=-1, leaf=False, label=None, parent = None, 
                    n_instances=0):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.depth = depth
        self.leaf = leaf
        self.label = label
        self.parent = parent
        self.n_instances = n_instances

    def __repr__(self):
        return f"DecisionTree(Attr : {self.attribute}, Value : {self.value}, Depth : {self.depth}, Label : {self.label}, Leaf: {self.leaf}, Instances: {self.n_instances})"
    
    def is_preleaf(self):
        if self.left.leaf == True and self.right.leaf == True:
            return True
        return False

class DecisionTree_Classifier():
    def __init__(self):
        self.dtree = None
        self.depth = 0

    def fit(self, dataset):
        self.dtree, self.depth = self.decision_tree_learning(dataset,0)
    
    def decision_tree_learning(self, training_dataset, depth):
        output = np.unique(training_dataset[:,-1])
        n_instances = training_dataset.shape[0]
        if output.shape[0]==1:
            return DecisionTree(leaf = True, label = output[0], depth = depth, n_instances = n_instances), depth

        split_attribute, split_value, split_left_dataset, split_right_dataset = find_split(training_dataset)
        dtree = DecisionTree(
            attribute = split_attribute, value = split_value, 
            depth = depth, n_instances = n_instances)
        dtree.left, left_depth = self.decision_tree_learning(split_left_dataset, depth+1)
        dtree.right, right_depth = self.decision_tree_learning(split_right_dataset, depth+1)
        dtree.left.parent = dtree
        dtree.right.parent = dtree

        return dtree, max(left_depth, right_depth)

    def predict(self, x_test):
        y_pred = np.zeros(x_test.shape[0])
        for i in range(x_test.shape[0]):
            tree = self.dtree
            while tree.leaf!=True:
                attr = tree.attribute
                val = tree.value
                if x_test[i][attr] >= val:
                    tree = tree.right
                else:
                    tree = tree.left
            y_pred[i] = tree.label
        return y_pred
    
    def compute_accuracy(self, x, y):
        y_pred = self.predict(x)
        n_correct = (y_pred == y).sum()
        n_total = y.shape[0]
        return n_correct/n_total
    


def test_decision_tree():
    dataset = np.loadtxt("wifi_db/noisy_dataset.txt", dtype=float)
    dtree = DecisionTree_Classifier()
    dtree.fit(dataset)
    # parse_tree(dtree.dtree)
    output, actual = dtree.predict(dataset)

    count = 0
    for i in range(len(output)):
        if output[i] == actual[i]:
            count+=1

    print(cross_validation(dataset, 10))

    clf = DecisionTreeClassifier(criterion = "entropy")
    print(dataset[:,:-1].shape,dataset[:,-1].shape)
    clf.fit(dataset[:,:-1], dataset[:,-1])

    predictions = clf.predict(dataset[:,:-1])

    from sklearn.metrics import accuracy_score
    print("Overall Accuracy = ", (count*100)/len(output))
    print(accuracy_score(dataset[:,-1], predictions))

    text_representation = tree.export_text(clf)
    print(text_representation)

    parse_tree(dtree.dtree)

    print(dtree.depth)
    print(clf.tree_.max_depth)


def parse_tree(node):

    if node.left!=None:
        parse_tree(node.left)
    if node.right!=None:
        parse_tree(node.right)

    print(node)

if __name__ == "__main__":
    test_decision_tree()

