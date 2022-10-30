from evaluation_metrics import get_confusion_matrix, get_accuracy, get_precision, get_recall, get_f1_score
from decision_tree_classifier import DecisionTree, DecisionTree_Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import copy
import numpy as np

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
    # split dataset into k folds
    np.random.shuffle(dataset)
    splits = np.split(dataset, k)
    k_fold_metrics = {
        'confusion_matrix': [],
        'precision': [],
        'recall': [],
        'accuracy': [],
        'f1_score': []
    }

    
    # iterate over every fold
    for i in range(k):
        # test dataset is the i-th split of the dataset
        test_dataset = splits[i]
        
        # train dataset is the remaining portion of the dataset
        train_dataset = None
        if k!=1:
            for j in range(k):
                if i==j:
                    continue
                if not isinstance(train_dataset,np.ndarray):
                    train_dataset = splits[j]
                else:
                    train_dataset = np.append(train_dataset,splits[j],axis=0)

        else:
            train_dataset = dataset
        
        # create an instance of DecisionTree_Classifier
        classifier = DecisionTree_Classifier()
        
        # train the classifier with the dataset
        classifier.fit(train_dataset)
        
        # evaluate the classifier on the test dataset
        conf, acc, prec, rec, f1 = evaluate(classifier, test_dataset)
        # update the k fold metrics dictionary
        k_fold_metrics['confusion_matrix'].append(conf)
        k_fold_metrics['precision'].append(prec)
        k_fold_metrics['recall'].append(rec)
        k_fold_metrics['accuracy'].append(acc)
        k_fold_metrics['f1_score'].append(f1)

        '''
        # ??
        output, actual = classifier.predict(splits[i])
        count = 0

        for i1 in range(len(output)):
            if output[i1] == actual[i1]:
                count+=1

        print(f"Accuracy for Batch {i+1} = ", (count*100)/len(output), " Depth = ", classifier.depth)
        clf = DecisionTreeClassifier(criterion = "entropy")
        clf.fit(train_dataset[:,:-1], train_dataset[:,-1])
        predictions = clf.predict(splits[i][:,:-1])

        from sklearn.metrics import accuracy_score
        print(f"Bench Mark Accuracy = ",100*accuracy_score(splits[i][:,-1], predictions), " Depth = ", clf.tree_.max_depth)
        '''

        
    avg_metrics = {}
    avg_metrics['confusion_matrix'] = np.average(
        np.array(k_fold_metrics['confusion_matrix']), axis=0)
    avg_metrics['accuracy']  = np.average(np.array(k_fold_metrics['accuracy']), axis=0)
    avg_metrics['recall']  = np.average(np.array(k_fold_metrics['recall']), axis=0)
    avg_metrics['precision']  = np.average(np.array(k_fold_metrics['precision']), axis=0)
    avg_metrics['f1_score']  = np.average(np.array(k_fold_metrics['f1_score']), axis=0)
    return k_fold_metrics, avg_metrics


def train_test_k_fold(n_folds, n_instances):
    """ Split a dataset of length n_instances into n_folds.
    
    Args:
        n_folds (int): number of folds
        n_instances (int): length of the dataset
    
    Returns:
        split_indices (list[list[np.array]]: list of length n_folds
            where each element is a list containing two np arrays
            containing the train and test indices
    """
    rng = np.random.default_rng(12345)
    shuffled_indices = rng.permutation(n_instances)
    split_indices = np.array_split(shuffled_indices, n_folds)
    folds = []
    for k in range(n_folds):
        test_indices = split_indices[k]
        train_indices = np.concatenate(
            split_indices[:k] + split_indices[k+1:])
        folds.append([train_indices, test_indices])
    return folds



def get_prunning_result(node, val_dataset):
    """ Check if prunning the node improves accuracy on val_dataset."""
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
        n_correct_change = get_prunning_result(node, val_dataset)
        if n_correct_change >= 0:
            ###print(f"Prunning node: {node} improves accuracy by {n_correct_change}")
            # prune the node
            node.leaf = True 
            
            # update the node label and n_instances
            if node.left.n_instances > node.right.n_instances:
                node.label = node.left.label
            else:
                node.label = node.right.label
            node.n_instances = node.left.n_instances + node.right.n_instances
 
def nested_cross_validation(dataset, k=10):
    """Perform nested cross validation on a Decision Tree Classifier
        with pruning.
    
    Args:
        dataset (np.ndarray, shape Nx8)
    
    Returns:
        k_fold_metrics = dictionary containing the test metrics for 
            each fold
        avg_metrics = dictionary containing the average test metrics
            accross the k fold
    """
    n_outer_folds = k
    n_inner_folds = k-1
    n_instances = dataset.shape[0]
    
    # split the dataset into k folds
    outter_split_indices = train_test_k_fold(n_outer_folds, n_instances)
    
    k_fold_metrics = {
        'confusion_matrix': [],
        'precision': [],
        'recall': [],
        'accuracy': [],
        'f1_score': []
    }
    
    # outer cross validation loop
    for i, (trainval_indices, test_indices) in enumerate(outter_split_indices):
        trainval_dataset = dataset[trainval_indices,:]
        test_dataset = dataset[test_indices,:]

        # split the trainval dataset into k-1 folds
        n_inner_instances = trainval_dataset.shape[0]
        inner_split_indices = train_test_k_fold(
            n_inner_folds, n_inner_instances)
        
        # inner cross validation loop
        inner_metrics = {
                'confusion_matrix': [],
                'precision': [],
                'recall': [],
                'accuracy': [],
                'f1_score': []
        }
        for j, (train_indices, val_indices) in enumerate(inner_split_indices):
            train_dataset = trainval_dataset[train_indices,:]
            val_dataset = trainval_dataset[val_indices,:]

            
            # train a DecisionTree classifier on the train dataset
            classifier = DecisionTree_Classifier()
            classifier.fit(train_dataset)
            val_acc = classifier.compute_accuracy(val_dataset[:,:-1], val_dataset[:,-1])
            test_acc = classifier.compute_accuracy(test_dataset[:,:-1], test_dataset[:,-1])
            
            pre_depth = classifier.compute_depth(classifier.dtree, 0)
            # prune the classifier using the validation dataset
            prune_tree(classifier.dtree, val_dataset)

            post_val_acc = classifier.compute_accuracy(val_dataset[:,:-1], val_dataset[:,-1])
            post_test_acc = classifier.compute_accuracy(test_dataset[:,:-1], test_dataset[:,-1])
            post_depth = classifier.compute_depth(classifier.dtree, 0)

            print(f"batch ({i},{j}), val acc {val_acc} -> {post_val_acc}, test acc {test_acc} -> {post_test_acc}")
            print(f"batch ({i},{j}), depth {pre_depth} -> {post_depth}")
            
            # get evaluation metrics on test dataset
            conf, acc, prec, rec, f1 = evaluate(classifier, test_dataset)
            inner_metrics['confusion_matrix'].append(conf)
            inner_metrics['precision'].append(prec)
            inner_metrics['recall'].append(rec)
            inner_metrics['accuracy'].append(acc)
            inner_metrics['f1_score'].append(f1)
        
        # append avg evaluation metrics to the outer  metrics
        k_fold_metrics['confusion_matrix'].append(np.average(
            np.array(inner_metrics['confusion_matrix']), axis=0))
        k_fold_metrics['accuracy'].append(
            np.average(np.array(inner_metrics['accuracy']), axis=0))
        k_fold_metrics['recall'].append(
            np.average(np.array(inner_metrics['recall']), axis=0))
        k_fold_metrics['precision'].append(
            np.average(np.array(inner_metrics['precision']), axis=0))
        k_fold_metrics['f1_score'].append(
            np.average(np.array(inner_metrics['f1_score']), axis=0))
    
    avg_metrics = {}
    avg_metrics['confusion_matrix'] = np.average(
        np.array(k_fold_metrics['confusion_matrix']), axis=0)
    avg_metrics['accuracy']  = np.average(
        np.array(k_fold_metrics['accuracy']), axis=0)
    avg_metrics['recall']  = np.average(
        np.array(k_fold_metrics['recall']), axis=0)
    avg_metrics['precision']  = np.average(
        np.array(k_fold_metrics['precision']), axis=0)
    avg_metrics['f1_score']  = np.average(
        np.array(k_fold_metrics['f1_score']), axis=0)

    return k_fold_metrics, avg_metrics


