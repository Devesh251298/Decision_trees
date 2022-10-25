import numpy as np
from calculate_entropy import calculate_entropy
# def calculate_entropy(dataset):
#     return 2

def find_split(dataset):
    """Find split that maximizes information gain in dataset.

    Args:
        dataset (np.ndarray, size N x 8): 
            dataset[:,:7] -> attributes (float)
            dataset[:,7] -> label (int from 1 to 4)
        

    Returns:
        split_attribute (int): integer from 0 to 6 (both included)
        split_value (float): value used to split the chosen split_attribute
        split_left_dataset (np.ndarray, shape Nleft x 8)
        split_right_dataset (np.ndarray, shape (N-Nleft) x 8)
    """
    # get the number of attributes in dataset (= 7)
    N, num_attributes = dataset.shape[0], dataset.shape[1] - 1
    # calculate the entropy of the entire dataset
    dataset_entropy = calculate_entropy(dataset)
    
    # initialize the target variables defining split
    max_info_gain = -np.inf
    
    # iterate over all attributes to find the splitting attribute
    for attribute in range(num_attributes):
        # get values of that attribute
        values = dataset[:, attribute]
        
        # get indices of the dataset sorted according to that attribute
        sorted_indices = np.argsort(values)
        x = np.unique(values[sorted_indices])
        # Spliting set to mid points between data points
        x = x[:-1] + np.diff(x)/2

        # iterate from smallest to largest value and find optimal splitting point
        for i in x:
            value = i
            # split dataset on the condition
            # left_dataset -> attribute <= value
            # right_dataset ->  attribute > value

            right_indices = np.nonzero(values >= value)
            left_indices = np.nonzero(values < value)

            left_dataset = dataset[left_indices]
            right_dataset = dataset[right_indices]

            # calculate left and right entropies
            left_entropy = calculate_entropy(left_dataset)
            right_entropy = calculate_entropy(right_dataset)
            
            sl = len(left_dataset)
            # calculate information gain
            info_gain = dataset_entropy - (sl/N * left_entropy + (N-sl)/N * right_entropy)
            
            # if information gain is larger than the current maximum, update variables
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                split_attribute, split_value = attribute, value
                split_left_dataset = left_dataset
                split_right_dataset = right_dataset
    
    return split_attribute, split_value, split_left_dataset, split_right_dataset
    
def test_find_split():
    dataset = np.loadtxt("wifi_db/clean_dataset.txt", dtype=float)
    attribute, value, left_dataset, right_dataset = find_split(dataset)
    #print(f"attribute: {attribute}, value: {value}, left dataset: {left_dataset.shape}, right dataset: {right_dataset.shape}")
    
if __name__ == "__main__":
    test_find_split()
