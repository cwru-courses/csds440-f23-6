import random
import warnings
from typing import Tuple, Iterable
from sting.data import FeatureType

import numpy as np

"""
This is where you will implement helper functions and utility code which you will reuse from project to project.
Feel free to edit the parameters if necessary or if it makes it more convenient.
Make sure you read the instruction clearly to know whether you have to implement a function for a specific assignment.
"""


def count_label_occurrences(y: np.ndarray) -> Tuple[int, int]:
    """
    This is a simple example of a helpful helper method you may decide to implement. Simply takes an array of labels and
    counts the number of positive and negative labels.

    HINT: Maybe a method like this is useful for calculating more complicated things like entropy!

    Args:
        y: Array of binary labels.

    Returns: A tuple containing the number of negative occurrences, and number of positive occurences, respectively.

    """
    n_ones = (y == 1).sum()  # How does this work? What does (y == 1) return?
    n_zeros = y.size - n_ones
    return n_zeros, n_ones


def entropy(schema, feature_index, data, labels, split_criterion):
    # Implement this on your own!
    
    datatype = schema[feature_index].ftype
        

    
    #for value,count in zip(unique_values, counts):
        #print(value,':', count)
        
    # List of all unique feature values. For partitioning the data
    #tests = []
    
    '''
    # If the feature is continuous
    if datatype == FeatureType.CONTINUOUS:
        
        # Construct Testing List
        lastValChange = None # Index of the last value change
        for index, (value, label) in enumerate(zip(data, labels)):
            
            prevValue = data[index - 1] if index > 0 else None  # Get the previous value
            prevLabel = labels[index - 1] if index > 0 else None  # Get the previous label
            print(f"Index {index}: Value: {value}, Label: {label}, Previous Label: {prevLabel}")
            
            
            if prevValue != value:
                lastValChange = index
                
            if prevLabel != label:
                newTest = (value + data[lastValChange]) / 2 if prevValue != None else value     
                tests.append(newTest)
                   
    '''           
            
            
            
            
    
    # If the feature is discrete    
    # Calculate the unique values and their counts in the data
    unique_values, counts = np.unique(data, return_counts=True)
    # If calculating the entropy of the data with respect to itself
    if(np.array_equal(data, labels)):
            probabilities = counts / len(data)
            entropy_value = -np.sum(probabilities * np.log2(probabilities))
            return entropy_value
        
    else:
        # Calculate entropy using the formula: H(D) = -Σ(p_i * log2(p_i))
        entropy_value = 0.0
        
        for value, count in zip(unique_values, counts):
            # Filter the labels corresponding to the current unique value
            subset_labels = labels[data == value]
            
            # Printing the label metadata
            #n_zero, n_one = count_label_occurrences(subset_labels)
            #print(value,':', '0s:', n_zero, ',' , '1s:', n_one)

            # Calculate the probability of the current unique value
            probability = count / len(data)
            
            #Find no. of unique labels in subset_labels
            unique_labels = np.unique(subset_labels)
            
            # If there is only one unique label associated with this unique value
            if len(unique_labels) == 1:
                # Since there is only one label, the entropy is 0
                entropy_value += probability * 0 # entropy = 0
                
            # If there are two unique labels associated with this unique value 
            else:
                #Finding the entropy of the label w.r.t the unique feature value: H(Label | Feature = value)
                entropy_contribution = -np.sum(
                    (subset_labels == 0).sum() / len(subset_labels) * np.log2((subset_labels == 0).sum() / len(subset_labels)) +
                    (subset_labels == 1).sum() / len(subset_labels) * np.log2((subset_labels == 1).sum() / len(subset_labels))
                )
                # Adding on the H(Label | Feature = value) to the total entropy value (H(Label | Feature)
                entropy_value += probability * entropy_contribution

            # Update the total entropy value
        #return entropy_value            
        return entropy_value # Return the entropy value: H(Label | Feature) general entropy of the label w.r.t the feature
    
def infogain(schema, feature_index, data, labels):
    # Get the entropy of the labels w.r.t themself
    entropy_labels = entropy(schema, feature_index, labels, labels)
    
    # Get the entropy of the labels w.r.t the data
    entropy_labels_data = entropy(schema, feature_index, data, labels)
    
    # Calculate the information gain
    information_gain = entropy_labels - entropy_labels_data
    
    return information_gain
    
def cv_split(
        X: np.ndarray, y: np.ndarray, folds: int, stratified: bool = False
    ) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], ...]:
    """
    Conducts a cross-validation split on the given data.

    Args:
        X: Data of shape (n_examples, n_features)
        y: Labels of shape (n_examples,)
        folds: Number of CV folds
        stratified:

    Returns: A tuple containing the training data, training labels, testing data, and testing labels, respectively
    for each fold.

    For example, 5 fold cross validation would return the following:
    (
        (X_train_1, y_train_1, X_test_1, y_test_1),
        (X_train_2, y_train_2, X_test_2, y_test_2),
        (X_train_3, y_train_3, X_test_3, y_test_3),
        (X_train_4, y_train_4, X_test_4, y_test_4),
        (X_train_5, y_train_5, X_test_5, y_test_5)
    )

    """

    # Set the RNG seed to 12345 to ensure repeatability
    np.random.seed(12345)
    random.seed(12345)

    # HINT!
    if stratified:
        n_zeros, n_ones = count_label_occurrences(y)

    #warnings.warn('cv_split is not yet implemented. Simply returning the entire dataset as a single fold...')

    return (X, y, X, y),


def accuracy(y: np.ndarray, y_hat: np.ndarray) -> float:
    """
    Another example of a helper method. Implement the rest yourself!

    Args:
        y: True labels.
        y_hat: Predicted labels.

    Returns: Accuracy
    """

    if y.size != y_hat.size:
        raise ValueError('y and y_hat must be the same shape/size!')

    n = y.size

    return (y == y_hat).sum() / n


def precision(y: np.ndarray, y_hat: np.ndarray) -> float:
    raise NotImplementedError()


def recall(y: np.ndarray, y_hat: np.ndarray) -> float:
    raise NotImplementedError()


def roc_curve_pairs(y: np.ndarray, p_y_hat: np.ndarray) -> Iterable[Tuple[float, float]]:
    raise NotImplementedError()


def auc(y: np.ndarray, p_y_hat: np.ndarray) -> float:
    roc_pairs = roc_curve_pairs(y, p_y_hat)
    raise NotImplementedError()