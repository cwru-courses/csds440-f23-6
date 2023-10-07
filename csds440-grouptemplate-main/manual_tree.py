import numpy as np
import matplotlib.pyplot as plt

def generate_data(N):
    # Generate N points from (-1,1)^2
    X = np.random.uniform(-1, 1, (N, 2))
    
    # Label the points using the classifier y=sign(0.5x1+0.5x2)
    y = np.sign(0.5 * X[:, 0] + 0.5 * X[:, 1])
    
    return X, y

# Test the function
#data, labels = generate_data(5)


def entropy(y):
    """Compute the entropy for a list of labels."""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(y, y_left, y_right):
    """Compute the information gain of a split."""
    parent_entropy = entropy(y)
    left_entropy = entropy(y_left)
    right_entropy = entropy(y_right)
    
    p_left = len(y_left) / len(y)
    p_right = len(y_right) / len(y)
    
    return parent_entropy - p_left * left_entropy - p_right * right_entropy

def id3(X, y, depth=0):
    """A basic ID3 algorithm returning the depth of the tree."""
    # If the node is pure or no attributes left, stop
    if len(np.unique(y)) == 1:
        return depth
    
    gains = []
    for i in range(X.shape[1]):
        median = np.median(X[:, i])
        y_left = y[X[:, i] <= median]
        y_right = y[X[:, i] > median]
        
        gains.append(information_gain(y, y_left, y_right))
    
    # If no gain from any attribute, stop
    if max(gains) == 0:
        return depth
    
    # Split using the best attribute
    best_attr = np.argmax(gains)
    median = np.median(X[:, best_attr])
    
    X_left, y_left = X[X[:, best_attr] <= median], y[X[:, best_attr] <= median]
    X_right, y_right = X[X[:, best_attr] > median], y[X[:, best_attr] > median]
    
    # Recursively apply ID3 on the left and right children
    left_depth = id3(X_left, y_left, depth + 1)
    right_depth = id3(X_right, y_right, depth + 1)
    
    return max(left_depth, right_depth)

# Test the ID3 on a small dataset
X_test, y_test = generate_data(50)
depth = id3(X_test, y_test)

#print(depth)



# Sizes of datasets
N_values = [50, 100, 500, 1000, 5000]
depths = []


# Generate data and apply ID3 for each size
for N in N_values:
    X, y = generate_data(N)
    depth = id3(X, y)
    depths.append(depth)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(N_values, depths, marker='o')
plt.xlabel('Number of Data Points (N)')
plt.ylabel('Depth of Decision Tree')
plt.title('Depth of Decision Tree vs. Dataset Size')
plt.grid(True)
plt.xticks(N_values)
plt.show()







