# Decision Tree Coursework - Intro to ML


## Description

In this project a binary decision tree algorithm is implemented and used to determine one of the indoor locations based on WIFI signal strengths collected from a mobile phone.

## Usage

To make use of this algorithm, run ```python main.py``` in the command line. This will return the results from the decision tree classifier. The results of running the scrip consists of all the evaluation metrics (confusion matrix, accuracy, precision, recall, f1-score):

    1. After performing 10-fold cross validation

        1.1. Evaluation metrics on clean dataset and the average depth of the tree

        1.2. Evaluation metrics on noisy dataset and the average depth of the tree

    2. After performing nested 10-fold cross-validation with pruning

        2.1. Evaluation metrics on clean dataset and average depth of the tree after pruning

        2.2. Evaluation metrics on noisy dataset and average depth of the tree after pruning

# Description of scripts 

## Calculate Entropy

- This script calculates the entropy of a label distribution according to the formula:

$S = -\sum (p_k * log(p_k))$

## Find Split 

- This script contains the definition of a funtion that finds the split that maximises the information gain. This function sorts the values of the attribute, and then consider only splitting points that are between two examples in sorted order and counts the total number of examples of each class at each side of the splitting point. It takes a dataset as the input and outputs the splitting attribute, value and the two splitted datasets (left and right). 

## Decision Tree Classifier
            
Defines the decision tree and decision tree classifier classes and initialises them. The methods defined for the classifier are the following:
- Fit: fits the datasets

- Learning: uses the find_split function to determine two new datasets, left and right, and create a DecisionTree instance for each one of them recursively

- Predict: forecasts the class

- Compute Accuracy: Calculates the accuracy of the classifier

- Compute Depth: Calculates the depth of the decision tree

- Save Figure: Saves the visualisation of the decision tree into the Plots folder

## Evaluation Metrics

- Defines the functions used to evaluate the performance of the decision tree. It takes as inputs the classifier and the testing dataset (without labels), and returns the confusion matrix, accuracy, recall, precision, and f1-measure

## Cross Validation

This script contains the following three functions:
 
- Train, Test k-Fold: Splits a dataset into k-folds. Takes as inputs the number of folds and the length of the dataset, and outputs a list of lists containing the train and test indices.

- Cross Validation: Evaluates the performance of the decision tree by first splitting the dataset into k=10 parts, and then for each of the 10 iterations, training the decision three with the training dataset of the corresponding fold and getting the evaluation metrics for the test dataset of the fold.

- Nested Cross Validation: Performs nested cross validation on a Decision Tree Classifier with pruning, it takes as input a dataset, and outputs a list of k*(k-1) decision tree classifiers (for k=10, it contains 90 trees), and a dictionary containing the average test metrics accross the k*(k-1) fold.

The evaluation metrics that are returned by both the cross validation and nested cross validation functions are confusion matrixs, accuracy, recall, precision, and F1-measure. 


## Pruning

- Determine the advantages of replacing each node with a single leaf in relation to the validation error for each node that is directly connected to two leaves. A single leaf will take the place of the node if it reduces or does not modify the validation error. In order to optimise the efficiency of the model, the tree must be parsed numerous times until there are no longer any nodes connecting two leaves.

## Visualisation and Plots

- The visualisation script contains a funtion to visualise a decision tree using matplotlib. Once main is ran, the plots corresponding to each decision tree (before pruning, after nested 10-k fold cross validation, etc) will be stored in the folder Plots.

## Authors 
Mireia Hernandez Caralt, Joshan Dooki, Devesh Joshi and Teresa Delgado de las Heras

