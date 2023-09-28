import argparse
import os.path
import warnings

from typing import Optional, List

import numpy as np
from sting.classifier import Classifier
from sting.data import Feature, parse_c45

import util


# In Python, the convention for class names is CamelCase, just like Java! However, the convention for method and
# variable names is lowercase_separated_by_underscores, unlike Java.
class DecisionTree(Classifier):
    def __init__(self, schema: List[Feature]):
        """
        This is the class where you will implement your decision tree. At the moment, we have provided some dummy code
        where this is simply a majority classifier in order to give you an idea of how the interface works. Don't forget
        to use all the good programming skills you learned in 132 and utilize numpy optimizations wherever possible.
        Good luck!
        """

        #warnings.warn('The DecisionTree class is currently running dummy Majority Classifier code. ' +
                      #'Once you start implementing your decision tree delete this warning message.')

        self._schema = schema  # For some models (like a decision tree) it makes sense to keep track of the data schema
        self._majority_label = 0  # Protected attributes in Python have an underscore prefix

    def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        """
        This is the method where the training algorithm will run.

        Args:
            X: The dataset. The shape is (n_examples, n_features).
            y: The labels. The shape is (n_examples,)
            weights: Weights for each example. Will become relevant later in the course, ignore for now.
        """

        # In Java, it is best practice to LBYL (Look Before You Leap), i.e. check to see if code will throw an exception
        # BEFORE running it. In Python, the dominant paradigm is EAFP (Easier to Ask Forgiveness than Permission), where
        # try/except blocks (like try/catch blocks) are commonly used to catch expected exceptions and deal with them.
        
        
        # Implement Split Criterion for Decision Tree
        #try:
            #split_criterion = self._determine_split_criterion(X, y)
        #except NotImplementedError:
            #warnings.warn('This is for demonstration purposes only.')
            
            
        #if(split_criterion):
            # Implement split criterion of a continuous attribute
            
            
            
        #for i in range(0, len(X)):
            #print('Data:', X[i])
            #print('Label:', y[i])
            
        #Steps for ID3 Algorithm Training
        #Step 1. For Each Feature F, Calculate the Entropy of the Feature, and the Information Gain of the Feature
        
        #Step 2. Select the Feature with the Highest Information Gain
        
                

        n_zero, n_one = util.count_label_occurrences(y)
        
        #print('Total Labels: ', n_zero + n_one)
        #print('Total Ones: ', n_one)
        #print('Total Zeroes: ', n_zero)
        
        
        # Entropy of Voting Label = 0.99139
        # Entropy Calculation of the Label
        #entropy = util.entropy(y, y) # Passing in the label vector for both inputs as I want the entropy of just the label
        
        # Entropy of the First Feature w.r.t the label = 
        #entropy = util.entropy(X[:, 0], y) # Passing in the first column of data and the label vector
        #print('Entropy of Label w.r.t First Feature:', entropy)
        
        # Calculating the Information Gain of the First Feature w.r.t the label
        #information_gain = util.infogain(X[:, 0], y)
        #print('Information Gain of First Feature w.r.t the Labels:', information_gain)
        
        # Testing out faulty Entropy calculations for the 12th Feature in the volcano dataset
        entropy1 = util.entropy(X[:, 10], y)
        entropy2 = util.entropy(X[:, 11], y)
        
        print('Entropy of Label w.r.t 11th Feature:', entropy1)
        print('Entropy of Label w.r.t 12th Feature:', entropy2)
        
        #infogain = util.infogain(X[:, 11], y)
        #print('Information Gain of 12th Feature w.r.t the Labels:', infogain)
       
        
        # Loop through each column of data, calculate the entropy and information gain of each feature w.r.t the label
        # and select the feature with the highest information gain
        #infogains = {}
        #for i in range(0, len(X[0])):
            #entropy = util.entropy(X[:, i], y) # Entropy of the ith feature w.r.t the label
            #print('Entropy of Feature', i+1,':', entropy)
            #information_gain = util.infogain(X[:, i], y) # Information Gain of the ith feature w.r.t the label
            #print('IG of Feature', i+1,':', information_gain)
            #infogains[i] = information_gain
            #print('-------------------------')
            
        # Select the feature with the highest information gain
        #max_ig_index = max(infogains, key=infogains.get) # returns the index of the feature with the highest information gain
        #print('Max IG:', infogains[max_ig_index])
        
        
        # Calculating the information gain of each feature w.r.t the label
        #information_gain = util.information_gain(X[:, 0], y)

        if n_one > n_zero:
            self._majority_label = 1
        else:
            self._majority_label = 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        This is the method where the decision tree is evaluated.

        Args:
            X: The testing data of shape (n_examples, n_features).

        Returns: Predictions of shape (n_examples,), either 0 or 1
        """

        # Returns either all 1s or all 0s, depending on _majority_label.
        return np.ones(X.shape[0], dtype=int) * self._majority_label

    # In Python, instead of getters and setters we have properties: docs.python.org/3/library/functions.html#property
    @property
    def schema(self):
        """
        Returns: The dataset schema
        """
        return self._schema

    # It is standard practice to prepend helper methods with an underscore "_" to mark them as protected.
    def _determine_split_criterion(self, X: np.ndarray, y: np.ndarray):
        """
        Determine decision tree split criterion. This is just an example to encourage you to use helper methods.
        Implement this however you like!
        """
        
        
        raise NotImplementedError()


def evaluate_and_print_metrics(dtree: DecisionTree, X: np.ndarray, y: np.ndarray):
    """
    You will implement this method.
    Given a trained decision tree and labelled dataset, Evaluate the tree and print metrics.
    """

    y_hat = dtree.predict(X)
    acc = util.accuracy(y, y_hat)
    print(f'Accuracy:{acc:.2f}')
    print('Size:', 0)
    print('Maximum Depth:', 0)
    print('First Feature:', dtree.schema[0])

    #raise NotImplementedError()


def dtree(data_path: str, tree_depth_limit: int, use_cross_validation: bool = True, information_gain: bool = True):
    """
    It is highly recommended that you make a function like this to run your program so that you are able to run it
    easily from a Jupyter notebook. This function has been PARTIALLY implemented for you, but not completely!

    :param data_path: The path to the data.
    :param tree_depth_limit: Depth limit of the decision tree
    :param use_cross_validation: If True, use cross validation. Otherwise, run on the full dataset.
    :param information_gain: If true, use information gain as the split criterion. Otherwise use gain ratio.
    :return:
    """

    # last entry in the data_path is the file base (name of the dataset)
    path = os.path.expanduser(data_path).split(os.sep)
    file_base = path[-1]  # -1 accesses the last entry of an iterable in Python
    root_dir = os.sep.join(path[:-1])
    schema, X, y = parse_c45(file_base, root_dir)

    if use_cross_validation:
        datasets = util.cv_split(X, y, folds=5, stratified=True)
    else:
        datasets = ((X, y, X, y),)

    for X_train, y_train, X_test, y_test in datasets:
        decision_tree = DecisionTree(schema)
        decision_tree.fit(X_train, y_train)
        #evaluate_and_print_metrics(decision_tree, X_test, y_test)

    #raise NotImplementedError()


if __name__ == '__main__':
    """
    THIS IS YOUR MAIN FUNCTION. You will implement the evaluation of the program here. We have provided argparse code
    for you for this assignment, but in the future you may be responsible for doing this yourself.
    """

    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run a decision tree algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('depth_limit', metavar='DEPTH', type=int,
                        help='Depth limit of the tree. Must be a non-negative integer. A value of 0 sets no limit.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.add_argument('--use-gain-ratio', dest='gain_ratio', action='store_true',
                        help='Use gain ratio as tree split criterion instead of information gain.')
    parser.set_defaults(cv=True, gain_ratio=False)
    args = parser.parse_args()

    # If the depth limit is negative throw an exception
    if args.depth_limit < 0:
        raise argparse.ArgumentTypeError('Tree depth limit must be non-negative.')

    # You can access args with the dot operator like so:
    data_path = os.path.expanduser(args.path)
    tree_depth_limit = args.depth_limit
    use_cross_validation = args.cv
    use_information_gain = not args.gain_ratio

    dtree(data_path, tree_depth_limit, use_cross_validation, use_information_gain)
