import numpy as np
import copy
from matplotlib.patches import BoxStyle
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from visualisation import visualise_decision_tree
from find_split import find_split

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

class DecisionTreeClassifier():
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

    def compute_depth(self, node, depth):
        if node.leaf:
            return depth
        return max(self.compute_depth(node.left,depth+1), self.compute_depth(node.right,depth+1))

    def save_fig(self, name):
        max_depth = self.compute_depth(self.dtree, 0)
        grid = np.zeros((max_depth+1,np.power(2,max_depth+1)))
        grid_x = np.zeros((max_depth+1,np.power(2,max_depth+1)))
        grid_y = np.zeros((max_depth+1,np.power(2,max_depth+1)))
        scale = 5
        step = scale/max_depth
        for i in range(max_depth+1):
            grid_y[i,:] = 0.1*i
        for j in range(np.power(2,max_depth+1)):
            grid_x[:,j] = (scale)*(j - int(np.power(2,max_depth+1)/2)) + 2
        
        fig, ax = plt.subplots(figsize=(10,10))
        visualise_decision_tree(node=self.dtree, tree=self, grid = grid, grid_x = grid_x, grid_y = grid_y,x=np.power(2,max_depth), y=max_depth, ax=ax, max_depth=max_depth, max_x=6, max_y=5)
        ax.margins(0.2, 0.2)  
        ax.axis('off')
       
        plt.ylim(0.1*(max_depth-5.5), 0.1*max_depth)
        plt.savefig(f'Plots/DT_{name}.png')


def test_decision_tree():
    dataset = np.loadtxt("wifi_db/noisy_dataset.txt", dtype=float)
    dtree = DecisionTreeClassifier()
    dtree.fit(dataset)
    # parse_tree(dtree.dtree)
    output, actual = dtree.predict(dataset)

    count = 0
    for i in range(len(output)):
        if output[i] == actual[i]:
            count+=1

    parse_tree(dtree.dtree)

def parse_tree(node):

    if node.left!=None:
        parse_tree(node.left)
    if node.right!=None:
        parse_tree(node.right)

    print(node)

if __name__ == "__main__":
    test_decision_tree()

