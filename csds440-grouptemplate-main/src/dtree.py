import argparse
import os.path
import warnings

from typing import Optional, List

import numpy as np
from sting.classifier import Classifier
from sting.data import Feature, parse_c45
from sting.data import FeatureType
from decimal import Decimal


import util

class Node():
    def __init__(self, schema = None, tests = None, class_label = None):
        self.schema = schema
        self.tests = tests
        self.children = {} # Empty dictionary, Every child key will be a test and the value will be a node
        self.class_label = class_label
        
    # Adds a child node to the current node given a certain test
    # all children will have an associated test, from the parent node
    def add_child(self, test, node):
        self.children[test] = node
        
    # Returns the children of the node
    def get_children(self):
        return self.children
    
    # Returns the schema of the node
    def get_schema(self):
        return self.schema
    
    # Returns the list of tests for the node
    def get_tests(self):
        return self.tests
    
    # Returns the child node associated with the test
    def get_child(self, test):
        return self.children[test]
    
    # Returns whether node is a leaf or not
    def is_leaf(self):
        return len(self.children) == 0
    
    # Method to set a node as a leaf with a particular class label
    def set_as_leaf(self, class_label):
        self.class_label = class_label
        self.schema = None
        self.tests = []
        self.children = {}
    
    # Method to get the class label of the leaf node
    def get_class_label(self):
        return self.class_label


# In Python, the convention for class names is CamelCase, just like in Java! However, the convention for method and
# variable names is lowercase_separated_by_underscores, unlike Java.
class DecisionTree(Classifier):
    def __init__(self, schema: List[Feature]):
        """
        This is the class where you will implement your decision tree. At the moment, we have provided some dummy code
        where this is simply a majority classifier in order to give you an idea of how the interface works. Don't forget
        to use all the good programming skills you learned in 132 and utilize numpy optimizations wherever possible.
        Good luck!
        """
        
        self._schema = schema  # For some models (like a decision tree) it makes sense to keep track of the data schema
        self._majority_label = 0  # Protected attributes in Python have an underscore prefix
        
        self.root = None
        self.features_in_tree = []
        self.masked_indeces = []
        
        self.data = None
        self.labels = None

        

    def fit(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        """
        This is the method where the training algorithm will run.

        Args:
            X: The dataset. The shape is (n_examples, n_features).
            y: The labels. The shape is (n_examples,)
            weights: Weights for each example. Will become relevant later in the course, ignore for now.
        """

        # Implement Split Criterion for Decision Tree
        try:
            split_criterion = self._determine_split_criterion(X, y, self._schema)
            #split_criterion = self._determine_split_criterion(X, y, self._schema)
            #split_criterion = self._determine_split_criterion(X, y)
        except NotImplementedError:
            warnings.warn('This is for demonstration purposes only.')
        
        #entropies = util.calculate_column_entropy(self._schema, X, y, split_criterion)
        
        infogains = util.infogain(self._schema, X, y, split_criterion)
        
        # Masked infogains is the list of infogains that have not been used in the tree
        masked_infogains = [infogains[i] for i in range(len(infogains)) if i not in self.masked_indeces]        
        
        #max_ig_index = np.argmax(infogains)
        # Masked ig index is the index of the feature with the highest infogain that has not been used in the tree
        max_ig_index = np.where(infogains == masked_infogains[np.argmax(masked_infogains)])[0][0]
        
        if infogains[max_ig_index] == 0:
            #print("LEAF ACTIVATED")
            #print("Leaf Name:", self._schema[max_ig_index].name)
            majority_label = util.majority_label(y)
            # returns leafe node
            return Node(schema = self._schema[max_ig_index], tests=None, class_label=majority_label)
        
        #root = Node(self._schema[max_ig_index], split_criterion[max_ig_index]) # Passing the schema of the root feature only, not general schema
        root = Node(schema = self._schema[max_ig_index], tests = split_criterion[max_ig_index], class_label = None) # Passing the schema of the root feature only, not general schema
        
        #self.features_in_tree.append(self._schema[max_ig_index].name)
        self.masked_indeces.append(max_ig_index)
        
        # If the main root of the tree has yet been initialized, set it to the current root
        if self.root is None:
            self.root = root
        
        # Break point for debugging
        return 0
            
        #print("-------------------------")
        
        # Constructing children of root node
        # Testing construction of a child node manually first
        for test in root.get_tests():
            print("Current test:", test)
            # Create masked data and labels for the current test only have rows in which the root feature is equal to the test
            # FIX FOR CONTINUOUS DATA
            mask = X[:, max_ig_index] == test
            mask_X = X[mask]
            mask_y = y[mask]
            
            # if all label values are the same, then create a leaf node
            if len(np.unique(mask_y)) == 1:
                #print("LEAF ACTIVATED")
                return root.set_as_leaf(util.majority_label(mask_y))

            
            if len(self.masked_indeces) == len(self._schema): # no more attributes to test
                #print("LEAF ACTIVATED")
                return root.set_as_leaf(util.majority_label(mask_y))

            
            # recursive call to fit covered by else case
            else:
                child = self.fit(mask_X, mask_y)
                root.add_child(test, child)
            #if (mask_X.size == 0) or (np.array_equal(mask_X, X)):
                #print("OUGNFJDSFEHIUBJONCWHUDEFBDJN")
                #return root
            
            #entropies = util.calculate_column_entropy(self._schema, mask_X, mask_y, split_criterion)
            
            #infogains = util.infogain(self._schema, mask_X, mask_y, split_criterion)
        return root
        
        
        #if n_one > n_zero:
            #self._majority_label = 1
        #else:
            #self._majority_label = 0

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
    def _determine_split_criterion(self, X: np.ndarray, y: np.ndarray, schema: List[Feature]):
        """
        Determine decision tree split criterion. This is just an example to encourage you to use helper methods.
        Implement this however you like!
        """
        # Dictionary that associates each feature with a list of tests
        
        # Dictionary to track the test list for each feature
        test_dic = {}
        # Loop through each column of data and calculate the tests for each feature
        for i in range(X.shape[1]):  
            
            tests = []  # List to store test values for the current feature          
            datatype = schema[i] # Get the datatype of the current feature of the dataset
            
            # If the feature is continuous
            if datatype.ftype == FeatureType.CONTINUOUS:
            #if True:

                # Construct Testing List
                lastValChange = None # Index of the last value change
                # Loop through each entry of the current column of data, and the label vector
                for index, (value, label) in enumerate(zip(X[:, i], y)):
                    
                    prevValue = X[:, i][index - 1] if index > 0 else None  # Get the previous value
                    prevLabel = y[index - 1] if index > 0 else None  # Get the previous label
                    #print(f"Index {index}: Value: {value}, Label: {label}, Previous Label: {prevLabel}")
                    
                    
                    if prevValue != value and index > 0:
                        lastValChange = index-1 # Get the index of the last value change
                        
                    if prevLabel != label and index > 0:
                        newTest = ((value + X[:, i][lastValChange]) / 2.0)
                        tests.append(newTest)                         
                        
                        #print(value, X[:, i][lastValChange])
                        #print(newTest)
                        #print('-------------------------')
                
                test_dic[i] = tests
            
            # If the feature is discrete
            #elif datatype.ftype == FeatureType.DISCRETE:
            else:
                # Calculate the unique values and their counts in the data
                unique_values= np.unique(X[:, i])
                for value in unique_values:
                    tests.append(value)
                
                test_dic[i] = tests
                
                    
        return test_dic  
                                                

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
        
        
        #print('-------------------------')
        
        #print("Root:", decision_tree.root.get_schema().name)
        #print("Root tests:", decision_tree.root.get_tests())
        
        #print('+------------------------+')
        
        #for test, child in decision_tree.root.get_children().items():
            #print("Test:", test, "Name:", child.get_schema().name)
            #print("Child (branch,feature):", decision_tree.root.get_child())
            #print("Child tests:", decision_tree.root.get_child(child).get_tests())
            #print('-------------------------')
        
        #print(decision_tree.features_in_tree)
        
        #evaluate_and_print_metrics(decision_tree, X_test, y_test)

    #raise NotImplementedError()
    
    #print(schema[1].attribute[0])
    #print(schema[1].ftype) # gives us whether the second feature of the dataset is continuous or discrete


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
    
    
    
    
    

