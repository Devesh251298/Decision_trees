from evaluation_metrics import evaluate, Metrics
from decision_tree_classifier import DecisionTree, DecisionTreeClassifier
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


def cross_validation(dataset_name, k=10):
    """ Train and test our DecisionTreeClassifier on the dataset 
    stored in the path 'wifi_db/{dataset_name}' using k-fold cross-
    validation.
    
    This function splits the dataset into k folds and performs k
    train/test iterations. At each iteration, one fold is used for test
    and the remaining k-1 folds are used to train the decision tree
    classifier.
    
    
    Args:
        dataset_name (str): name of the dataset we wish to use (which
            needs to be stored in the path 'wifi_db/{dataset_name}')
        k (int, default=10): number of folds used for train/test split

    Returns:
        trees (list[DecisionTreeClassifier]): list of length k containing
            the trained decision tree classifiers
        avg_metrics (dict, keys='str', values=np.ndarray): dictionary
            containing the evaluation metrics computed on the test set
             averaged over k trees.
        
    """
    # split dataset into k folds
    dataset = np.loadtxt(f'wifi_db/{dataset_name}')

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
        
        # create an instance of DecisionTreeClassifier
        classifier = DecisionTreeClassifier()
        
        # train the classifier with the dataset
        classifier.fit(train_dataset)
        
        # evaluate the classifier on the test dataset
        conf, acc, prec, rec, f1 = evaluate(classifier, test_dataset)
    
        # compute depth of the trained tree
        dep = classifier.compute_depth(classifier.dtree, 0)

        # saves the decision tree visualization in the Plots folder
        classifier.save_fig(dataset_name.split('.')[0]+'_'+str(i))

        # update the k fold metrics dictionary
        eval_metrics.add_metrics(conf, acc, prec, rec, f1, dep)
        trees.append(classifier)
        
    avg_metrics = eval_metrics.get_avg_metrics()

    return trees, avg_metrics




def nested_cross_validation(dataset, k=10):
    """ Train and test our DecisionTreeClassifier on the dataset 
    stored in the path 'wifi_db/{dataset_name}' using nested k-fold 
    cross-validation with pruning.
    
    This function splits the dataset into k folds and performs nested 
    cross-validation with an outer loop of k iterations and an inner loop
    of k-1 iterations (total of k*(k-1) iterations).
    
    For each outer loop iteration, 1 fold is used for test and the other
    k-1 folds are used for train/val. The train/val set is then split 
    again into k-1 folds.
    
    For each inner loop iteration, the function uses 1 fold of the
    train/val split for validation the remaining k-2 folds for training. 
    It trains a decision tree classifier using the train set, and 
    performs pruning using the validation set. After  pruning, the 
    function computes the evaluation metrics on the separate test set.
    
    Args:
        dataset_name (str): name of the dataset we wish to use (which
            needs to be stored in the path 'wifi_db/{dataset_name}')
        k (int, default=10): number of folds used for train/test split

    Returns:
        trees (list[DecisionTreeClassifier]): list of length k containing
            the trained decision tree classifiers
        avg_metrics (dict, keys='str', values=np.ndarray): dictionary
            containing the evaluation metrics computed on the test set
             averaged over k*(k-1) trees.
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
            classifier = DecisionTreeClassifier()
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


