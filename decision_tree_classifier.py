from find_split import find_split


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
        return f"DecisionTree({self.attribute}, {self.value}, {self.leaf})"

class DecisionTreeClassifier():
    def __init__(self):
        self.dtree = None

    def fit(self, x_train, y_train):
        self.dtree = self.decision_tree_learning(np.concatenate((x_train, y_train), axis=0) ,0)
    
    def decision_tree_learning(self, training_dataset, depth):
        output = np.unique(training_dataset[:,-1]).shape[0]
        if output.shape[0]==1:
        	return DecisionTree(leaf = True, label = output[0], depth = 0)

        split_attribute, split_value, split_left_dataset, split_right_dataset = find_split(training_dataset)
        dtree = DecisionTree(attribute = split_attribute, value = split_value, depth = depth)
        dtree.left = self.decision_tree_learning(split_left_dataset, depth+1)
        dtree.right = self.decision_tree_learning(split_right_dataset, depth+1)
        dtree.depth = max(dtree.left.depth, dtree.right.depth)

        return dtree
        
