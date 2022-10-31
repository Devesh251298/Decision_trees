import numpy as np
from cross_validation import cross_validation, nested_cross_validation

def load_data(filepath):
    """Load data.

    Extended description of function.

    Args:
        filename (str): filepath

    Returns:
        dataset (np.ndarray(Nx8)): dataset

    """
    dataset = np.loadtxt(filepath, dtype=float)
    return dataset




if __name__ == '__main__':
    clean_dataset = load_data('wifi_db/clean_dataset.txt')
    noisy_dataset = load_data('wifi_db/noisy_dataset.txt')
    
    # Perform 10-fold cross validation on the clean dataset
    print("Performing 10-fold cross validation on the clean dataset")
    trees, tree_metrics, avg_metrics = cross_validation(clean_dataset)
    print(f"Avg accuracy for clean dataset: {avg_metrics['accuracy']}")

    # Save results in npy pickles
    print("Saving results...")
    np.save("results/clean_tree_metrics.npy", tree_metrics, allow_pickle = True)
    np.save("results/clean_avg_metrics.npy", avg_metrics, allow_pickle = True)
    
    # Perform 10-fold cross validation on the clean dataset
    print("Performing 10-fold cross validation on the noisy dataset")
    noisy_trees, noisy_tree_metrics, noisy_avg_metrics = \
        cross_validation(noisy_dataset)
    print(f"Avg accuracy for noisy dataset: {noisy_avg_metrics['accuracy']}")

    # Save results in npy pickles
    print("Saving results...")
    np.save("results/noisy_tree_metrics.npy", noisy_tree_metrics, allow_pickle = True)
    np.save("results/noisy_avg_metrics.npy", noisy_avg_metrics, allow_pickle = True)
    
    '''
    # Perform nested 10-fold cross validation with pruning
    print("Performing 10-fold cross-validation with pruning...")
    nested_k_fold_metrics, nested_avg_metrics = nested_cross_validation(clean_dataset)
    nested_noisy_k_fold_metrics, nested_noisy_avg_metrics = nested_cross_validation(noisy_dataset)
    breakpoint()
    # Print evaluation metrics
    print(f"Avg accuracy for clean dataset: {nested_avg_metrics['accuracy']}")
    print(f"Avg accuracy for noisy dataset: {nested_noisy_avg_metrics['accuracy']}")
    '''
