from evaluation_metrics import evaluate
from decision_tree_classifier import DecisionTree, DecisionTree_Classifier
import numpy as np
from prunning import prune_tree

class Metrics:
    def __init__(self):
        self.metric_dict = {
            'confusion_matrix': [],
            'precision': [],
            'recall': [],
            'accuracy': [],
            'f1_score': [],
            'depth': []
        }
    
    def add_metrics(self, conf, acc, prec, rec, f1, dep):
        """ Adds values to a metric dictionary."""
        self.metric_dict['confusion_matrix'].append(conf)
        self.metric_dict['precision'].append(prec)
        self.metric_dict['recall'].append(rec)
        self.metric_dict['accuracy'].append(acc)
        self.metric_dict['f1_score'].append(f1)
        self.metric_dict['depth'].append(dep)
    
    def get_avg_metrics(self):
        """ Return avg metrics dictionary."""
        avg_metrics = {}
        avg_metrics['confusion_matrix'] = np.average(
            np.array(self.metric_dict['confusion_matrix']), axis=0)
        avg_metrics['accuracy']  = np.average(
            np.array(self.metric_dict['accuracy']), axis=0)
        avg_metrics['recall']  = np.average(
            np.array(self.metric_dict['recall']), axis=0)
        avg_metrics['precision']  = np.average(
            np.array(self.metric_dict['precision']), axis=0)
        avg_metrics['f1_score']  = np.average(
            np.array(self.metric_dict['f1_score']), axis=0)
        
        return avg_metrics



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
    tree_metrics = Metrics()
    
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
        tree_metrics.add_metrics(conf, acc, prec, rec, f1, dep)
        trees.append(classifier)
        
    avg_metrics = tree_metrics.get_avg_metrics()

    return trees, tree_metrics.metric_dict, avg_metrics




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
        'f1_score': [],
        'depth': []
    }
    k_fold_trees = []
    
    # outer cross validation loop
    for i, (trainval_indices, test_indices) in enumerate(outter_split_indices):
        trainval_dataset = dataset[trainval_indices,:]
        test_dataset = dataset[test_indices,:]

        # split the trainval dataset into k-1 folds
        n_inner_instances = trainval_dataset.shape[0]
        inner_split_indices = train_test_k_fold(
            n_inner_folds, n_inner_instances)
        
        inner_trees = []
        inner_metrics = {
                'confusion_matrix': [],
                'precision': [],
                'recall': [],
                'accuracy': [],
                'f1_score': [],
                'depth':[]
        }

        # inner cross validation loop
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


