from find_split import find_split
import numpy as np

class DecisionTree:
    def __init__(self, attribute=0, value=-1, left=None, right=None, depth=-1, leaf=False, label=None):
        self.attribute = attribute
        self.value = value
        self.left = left
        self.right = right
        self.depth = depth
        self.leaf = leaf
        self.label = leaf

    def __repr__(self):
        return f"DecisionTree({self.attribute}, {self.value})"

class DecisionTreeClassifier():
    def __init__(self):
        self.dtree = None
        self.depth = 0

    def fit(self, dataset):
        self.dtree, self.depth = self.decision_tree_learning(dataset,0)
    
    def decision_tree_learning(self, training_dataset, depth):
        print(training_dataset.shape, depth)
        output = np.unique(training_dataset[:,-1])
        if output.shape[0]==1:
        	return DecisionTree(leaf = True, label = output[0], depth = depth), depth

        split_attribute, split_value, split_left_dataset, split_right_dataset = find_split(training_dataset)
        dtree = DecisionTree(attribute = split_attribute, value = split_value, depth = depth)
        dtree.left, left_depth = self.decision_tree_learning(split_left_dataset, depth+1)
        dtree.right, right_depth = self.decision_tree_learning(split_right_dataset, depth+1)

        return dtree, max(left_depth, right_depth)
        
def test_decision_tree():
    dataset = np.loadtxt("wifi_db/clean_dataset.txt", dtype=float)
    dtree = DecisionTreeClassifier()
    dtree.fit(dataset)
    print(dtree.depth)
    #print(f"attribute: {attribute}, value: {value}, left dataset: {left_dataset.shape}, right dataset: {right_dataset.shape}")
    
if __name__ == "__main__":
    test_decision_tree()

