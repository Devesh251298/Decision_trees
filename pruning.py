import numpy as np


def get_pruning_result(node, val_dataset):
    """ Check if pruning the node improves accuracy on val_dataset."""
    x_val, y_val = val_dataset[:,:-1], val_dataset[:,-1]
    
    # get indices of the instances that do not end up in this node
    indices = np.full(y_val.shape[0],True)
    
    child_node = node
    parent_node = node.parent
    while parent_node is not None:
        value = parent_node.value
        attribute = parent_node.attribute
        # check which data instances do not fall in this child node
        if child_node == parent_node.left:
            indices[x_val[:, attribute] >= value] = False
        else:
            indices[x_val[:, attribute] < value] = False
        child_node = parent_node
        parent_node = child_node.parent
    
    # compute the change in correct predictions
    changed_instances = indices.sum()
    if changed_instances == 0:
        return 0
    
    # calculate the predictions of the classifier without pruning
    y_pred = np.zeros(shape=changed_instances)
    y_pred[x_val[indices][:,node.attribute] >= node.value] = node.right.label
    y_pred[x_val[indices][:,node.attribute] < node.value] = node.left.label
    
    # calculate the predictions of the classifier with pruning
    y_prun = np.zeros(shape=indices.sum())
    if node.left.n_instances > node.right.n_instances:
        y_prun[:] = node.left.label
    else:
        y_prun[:] = node.right.label
    
    # compare the number of correct predictions
    n_correct = (y_pred == y_val[indices]).sum()
    n_correct_prun = (y_prun == y_val[indices]).sum()

    #print(f"Pruning improves by {n_correct_prun - n_correct}")
    change = n_correct_prun - n_correct
    return change
    
    
def prune_tree(node, val_dataset):
    ''' Prune the tree recursively.
    
    Args:
        node (DecisionTree)
    
    Returns:
        None
    '''
    # parse the tree post-order style - because we need to check the
    # childs before the parent, since pruning a child might mean
    # that we also need to prune the parent afterwards
    
    # parse the left node if it is not a leaf
    if not node.left.leaf:
        prune_tree(node.left, val_dataset)
    # parse the right node if it is not a leaf
    if not node.right.leaf:
        prune_tree(node.right, val_dataset)
    
    # check if the current node is a preleaf node
    if node.is_preleaf():
        # check if pruning it improves accuracy on validation dataset
        n_correct_change = get_pruning_result(node, val_dataset)
        if n_correct_change >= 0:
            ###print(f"pruning node: {node} improves accuracy by {n_correct_change}")
            # prune the node
            node.leaf = True 
            
            # update the node label and n_instances
            if node.left.n_instances > node.right.n_instances:
                node.label = node.left.label
            else:
                node.label = node.right.label
            node.n_instances = node.left.n_instances + node.right.n_instances
