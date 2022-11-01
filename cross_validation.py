from evaluation_metrics import evaluate, Metrics
from decision_tree_classifier import DecisionTree, DecisionTree_Classifier
import numpy as np
from pruning import prune_tree


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
    n_instances = dataset.shape[0]
    split_indices = train_test_k_fold(k, n_instances)
    
    # list to store the trained classifiers 
    trees = []
    
    # instantiate a Metrics object to store the evaluation metrics 
    eval_metrics = Metrics()
    
    # iterate over every fold
    for i, (train_indices, test_indices) in enumerate(split_indices):
        # test dataset is the i-th split of the dataset
        train_dataset = dataset[train_indices,:]
        test_dataset = dataset[test_indices,:]
        
        # create an instance of DecisionTree_Classifier
        classifier = DecisionTree_Classifier()
        
        # train the classifier with the dataset
        classifier.fit(train_dataset)
        
        # evaluate the classifier on the test dataset
        conf, acc, prec, rec, f1 = evaluate(classifier, test_dataset)
    
        # compute depth of the trained tree
        dep = classifier.compute_depth(classifier.dtree, 0)

        # update the k fold metrics dictionary
        eval_metrics.add_metrics(conf, acc, prec, rec, f1, dep)
        trees.append(classifier)
        
    avg_metrics = eval_metrics.get_avg_metrics()

    return trees, avg_metrics




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
    
    # instantiate Metrics dict. store evaluation metrics for unprunned trees
    eval_metrics = Metrics()
    trees = []
    
    # outer cross validation loop
    for i, (trainval_indices, test_indices) in enumerate(outter_split_indices):
        trainval_dataset = dataset[trainval_indices,:]
        test_dataset = dataset[test_indices,:]

        # split the trainval dataset into k-1 folds
        n_inner_instances = trainval_dataset.shape[0]
        inner_split_indices = train_test_k_fold(
            n_inner_folds, n_inner_instances)

        # inner cross validation loop
        for j, (train_indices, val_indices) in enumerate(inner_split_indices):
            train_dataset = trainval_dataset[train_indices,:]
            val_dataset = trainval_dataset[val_indices,:]

            # train a DecisionTree classifier on the train dataset
            classifier = DecisionTree_Classifier()
            classifier.fit(train_dataset)
            
            # prune the classifier using the validation dataset
            prune_tree(classifier.dtree, val_dataset)
            
            # compute the evaluation metrics after pruning
            conf, acc, prec, rec, f1 = evaluate(classifier, test_dataset)
            dep = classifier.compute_depth(classifier.dtree, 0)
            eval_metrics.add_metrics(conf, acc, prec, rec, f1, dep)
            trees.append(classifier)
    
    avg_metrics = eval_metrics.get_avg_metrics()

    return trees, avg_metrics


