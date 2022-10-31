import numpy as np
from cross_validation import cross_validation, nested_cross_validation

def load_data(filepath):
    """Load data."""
    dataset = np.loadtxt(filepath, dtype=float)
    return dataset


def display_metrics(metric_dict):
    """ Display metrics in metric_dict """
    print(f"Avg accuracy:  \n{metric_dict['accuracy']}")
    print(f"Avg precision by class:  \n{metric_dict['precision']}")
    print(f"Avg recall by class:  \n{metric_dict['recall']}")
    print(f"Avg f1-score by class:  \n{metric_dict['f1_score']}")
    print(f"Avg confusion matrix: \n{metric_dict['confusion_matrix']}")
    print(f"Avg tree depth: \n{metric_dict['depth']}")


if __name__ == '__main__':
    # Load data
    clean_dataset = load_data('wifi_db/clean_dataset.txt')
    noisy_dataset = load_data('wifi_db/noisy_dataset.txt')
    
    # Perform 10-fold cross validation 
    print("Running 10-fold cross-validation on clean dataset...")
    trees, avg_metrics = cross_validation(clean_dataset)
    print("Running 10-fold cross-validation on noisy dataset...")
    noisy_trees, noisy_avg_metrics = cross_validation(noisy_dataset)

    # Perform nested 10-fold cross validation with pruning
    print("Running nested 10-fold cross-validation on clean dataset...")
    unprun_trees, unprun_metrics, prun_trees, prun_metrics = \
        nested_cross_validation(clean_dataset)
    print("Running nested 10-fold cross-validation on noisy dataset...")
    noisy_unprun_trees, noisy_unprun_metrics, noisy_prun_trees, noisy_prun_metrics = \
        nested_cross_validation(noisy_dataset)
    
    # Print answers for the report
    
    #### Report - Step 3
    # Display evaluation metrics for 10-fold cv without pruning
    print(60*'-'+"\n Step 3 - 10-fold cross-validation:\n"+60*'-')
    print(60*'-'+"\nEvaluation metrics on clean dataset:\n"+60*'-')
    display_metrics(avg_metrics)
    print(60*'-'+"\nEvaluation metrics on noisy dataset:\n"+60*'-')
    display_metrics(noisy_avg_metrics)
    
    #### Report - Step 4
    # Display evaluation metrics for nested 10-fold cv with pruning
    print(60*'-'+"\n Step 4 - nested 10-fold cross-validation with pruning:\n"+60*'-')
    print(60*'-'+"\nEvaluation metrics on clean dataset (before pruning):\n"+60*'-')
    display_metrics(unprun_metrics)
    print(60*'-'+"\nEvaluation metrics on clean dataset (after pruning):\n"+60*'-')
    display_metrics(prun_metrics)
    print(60*'-'+"\nEvaluation metrics on noisy dataset (before pruning):\n"+60*'-')
    display_metrics(noisy_unprun_metrics)
    print(60*'-'+"\nEvaluation metrics on noisy dataset (after pruning):\n"+60*'-')
    display_metrics(noisy_prun_metrics)


