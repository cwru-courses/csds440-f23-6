# CSDS440 Written Homework 4
**Instructions:** Each question is worth 10 points unless otherwise stated. Write your answers below the question. Each answer should be formatted so it renders properly on github. **Answers that do not render properly may not be graded.** Please comment the last commit with "FINAL COMMIT" and **enter the final commit ID in canvas by the due date.** 

When working as a group, only one answer to each question is needed unless otherwise specified. Each person in each group must commit and push their own work. **You will not get credit for work committed/pushed by someone else even if done by you.** Commits should be clearly associated with your name or CWRU ID (abc123). Each person is expected to do an approximately equal share of the work, as shown by the git logs. **If we do not see evidence of equal contribution from the logs for someone, their individual grade will be reduced.** 

Names and github IDs (if your github ID is not your name or Case ID):

1.	Show that the set $C=$ \{ $x|Ax\geq b$ \}, $A \in R^{m\times n}$, $x \in R^n$, $b \in R^m$, is a convex set. Note that this describes the constraint set of a linear program. (10 points)

Answer: 
We employ the definition of convexity to demonstrate the convexity of the set C = {x | Axe >= b}. If the line segment joining any two locations in the set is also totally contained inside the set, then the set is convex. Mathematically speaking, a set X is convex if the point λx₁ + (1-λ)x₂ is also in X for any two points x₁ and x₂ in X and for any λ in the range [0,1].

In this case, C is defined as follows:
C = {x | Ax ≥ b}

Where:
- A is a matrix of size M×n.
- x is a vector in Rⁿ.
- b is a vector in Rᴹ.

Now, let's take two points x₁ and x₂ in the set C, which means that Ax₁ ≥ b and Ax₂ ≥ b.

We want to show that for any λ in the range [0,1], the point λx₁ + (1-λ)x₂ is also in the set C, which means that A(λx₁ + (1-λ)x₂) ≥ b.

Let's prove this:
A(λx₁ + (1-λ)x₂) = λAx₁ + (1-λ)Ax₂

Since Ax₁ ≥ b and Ax₂ ≥ b, we have:
λAx₁ + (1-λ)Ax₂ ≥ λb + (1-λ)b = b (by the properties of linear combinations)

So, A(λx₁ + (1-λ)x₂) ≥ b, which means that the convex combination of x₁ and x₂, λx₁ + (1-λ)x₂, is also in the set C.

Therefore, the set C = {x | Ax ≥ b} is convex, which is a desirable property for the constraint set of a linear program.

2.	A function $f$ is said to have a global minimum at $x$ if for all $y$, $f(y) \geq f(x)$. It is said to have a local minimum at $x$ if there exists a neighborhood $H$ around $x$ so that for all $y$ in $H$, $f(y)\geq f(x)$. Show that, if $f$ is convex, every local minimum is a global minimum. [Hint: Prove by contradiction using Jensen’s inequality.] (10 points)

Answer: 
A function ƒ exhibiting convexity is a crucial characteristic supporting the idea that each local minimum is a global minimum. It's called the "non-negativity of the subdifferential."

Let's define the concept of the subdifferential:

All of the function's subgradients at a given point x make up the subdifferential of a convex function ƒ, which is represented as ∂̒(x). An arbitrary vector g that ensures the following inequality holds for every point y is a subgradient at x:

ƒ(y) ≥ ƒ(x) + gᵀ(y - x)

Now, let's consider a function ƒ that has a local minimum at point x, which means there exists a neighborhood H around x such that, for all y in H, ƒ(y) ≥ ƒ(x).

Our goal is to establish the global minimum of this local minimum.

If g is a subgradient at x for a convex function ƒ, then g belongs to the subdifferential ∂ƒ(x). According to the subdifferential's definition, at every point y, we have:

ƒ(y) ≥ ƒ(x) + gᵀ(y - x)

Now, for any y in the neighborhood H around x, we know that ƒ(y) ≥ ƒ(x). Thus, the inequality becomes:

ƒ(x) ≥ ƒ(x) + gᵀ(y - x)

By subtracting ƒ(x) from both sides, we get:

0 ≥ gᵀ(y - x)

Since this inequality applies when y = x, it also holds for all y in the neighborhood H. Therefore:
0 ≥ gᵀ(0) = 0

Thus, gᵀ(0) = 0 is implied. It follows that all subgradients in the ∂ƒ(x) are non-negative since this inequality holds for all subgradients g.

The subdifferential only contains non-negative subgradients, hence for a convex function, every local minimum is also a global minimum. This characteristic guarantees that the function cannot have any local minima or "bumps" inside the convex region.

3.	Consider the LP: $\min c^Tx$ s.t. $Ax \geq b, x \geq 0$, where $T$ is the transpose, $A$ is the 4x2 matrix: \[ 0 −1; −1 −1; −1 2; 1 −1\], $b$ is a 4x1 vector \[−5; −9;0; −3\] and $c$ is a 2x1 vector \[−1; −2\]. (a) Draw the feasible region in $R^2$. (b) Draw the contours of $c^Tx =−12$, $c^Tx =−14$ and $c^Tx =−16$ and determine the solution graphically. (10 points)

Answer: 
To visualize the feasible region and draw the contours, we'll first find the vertices of the feasible region by solving the given linear program:

Minimize cᵀx
Subject to Ax > b
x > 0

Given:
A = [ 0  -1;
     -1 -1;
     -1  2;
      1 -1]

b = [-5; -9; 0; -3]

c = [-1; -2]

(a) Finding the Feasible Region:

We'll start by solving the inequalities Ax > b, where x > 0. This will help us identify the feasible region. To do this, we need to find the vertices of the feasible region. The vertices are the points where the constraints intersect.

Let's solve these inequalities to find the vertices:

1. 0x - x ≥ -5   →  -x ≥ -5   →  x ≤ 5
2. -1x - 1x ≥ -9 →  -2x ≥ -9 →  x ≤ 4.5
3. -1x + 2x ≥ 0  →  x ≥ 0
4. 1x - 1x ≥ -3  →  0 ≥ -3 (always true)

So, we have the following vertices for the feasible region: (0, 0), (5, 0), and (4.5, 4.5).

Now, let's move on to part (b) to draw the contours of c₁x and determine the solution graphically.

(b) Drawing the Contours and Determining the Solution Graphically:

We will draw the contours for c₁x = -12, c₁x = -14, and c₁x = -16. To do this, we need to find the corresponding c₂ values for each contour:

1. c₁x = -12
c₂x = -2 * x = -12 → x = 6
2. c₁x = -14
c₂x = -2 * x = -14 → x = 7
3. c₁x = -16
c₂x = -2 * x = -16 → x = 8

Now, we will draw these contours on the graph:

- For c₁x = -12, we have a contour at x = 6.
- For c₁x = -14, we have a contour at x = 7.
- For c₁x = -16, we have a contour at x = 8.

The solution will be the point within the feasible region where the contours of c₁x = -12, c₁x = -14, and c₁x = -16 intersect. From the vertices we found earlier (0, 0), (5, 0), and (4.5, 4.5), we can see that the solution will be at the intersection of the contours.

Now, let's draw the feasible region and the contours on a graph:



               c₁x = -16
               |   
               |
   (5,0)       |
     +---+---- + +
     |   |      |   
     |   |    c₁x = -14
     |   |   /   
     |   |  /
     |   | /   
     |   |/
     +---+
    /      \
   /        \
(0,0)   (4.5,4.5)



The solution, which minimizes cᵀx, is where the contours of c₁x = -12, c₁x = -14, and c₁x = -16 intersect. This point can be found on the graph where the three contours meet, and it represents the optimal solution to the linear program.

4.	Consider the primal linear program (LP): $\min c^Tx$ s.t. $Ax \geq b, x \geq 0$ and its dual: $\max b^Tu$ s.t. $A^Tu \leq c, u \geq 0$. Prove that for any feasible $(x,u)$ (i.e. $x$ and $u$ satisfying the constraints of the two LPs), $b^Tu \leq c^Tx$. (10 points)

Answer: 

5.	Derive the backpropagation weight updates for hidden-to-output and input-to-hidden weights when the loss function is cross entropy with a weight decay term. Cross entropy is defined as $L(\mathbf{w})=\sum_i y_i\log{(\hat{y}_i)}+(1-y_i)\log{(1-\hat{y}_i)}$ , where $i$ ranges over examples, $y_i$ is true label (assumed 0/1) and $\hat{y}_i$  is the estimated label for the $i^{th}$ example. (10 points)

Answer:
Calculus's chain rule can be used to determine the backpropagation weight updates for hidden-to-output and input-to-hidden weights when the loss function is cross-entropy with a weight decay term. One definition of the loss function is:

L(w) = -(1/n) * Σ[i=1 to n] [y_i * log(ŷ_i) + (1 - y_i) * log(1 - ŷ_i)] + (λ/2) * Σ[j=1 to H] Σ[k=1 to M] w_jk^2

Where:
- n is the no: of examples.
- H is the no: of hidden units.
- M is the no: of output units.
- y_i is the true label (assumed to be 0 or 1).
- ŷ_i is the estimated label for the i-th example.
- w_jk represents the weight connecting the j-th hidden unit to the k-th output unit.
- λ is the weight decay hyperparameter.

The weight updates for input-to-hidden weights (v_ij) and hidden-to-output weights (w_jk) will be derived.

**1. Weight Updates for Hidden-to-Output Weights (w_jk):**

In relation to w_jk, the loss function is defined as follows:

∂L/∂w_jk = -(1/n) * Σ[i=1 to n] [y_i * (1/ŷ_i) * ∂ŷ_i/∂w_jk - (1 - y_i) * (1/(1 - ŷ_i)) * ∂(1 - ŷ_i)/∂w_jk] + λ * w_jk

To compute the derivatives ∂ŷ_i/∂w_jk and ∂(1 - ŷ_i)/∂w_jk, you can use the chain rule:

∂ŷ_i/∂w_jk = ∂(σ(net_k))/∂net_k * ∂(net_k)/∂w_jk = ŷ_i * (1 - ŷ_i) * z_j

∂(1 - ŷ_i)/∂w_jk = -∂ŷ_i/∂w_jk = -ŷ_i * (1 - ŷ_i) * z_j

Where:
- σ(net_k) is the sigmoid activation function applied to net_k.
- net_k is the weighted sum of inputs to the output unit k.
- z_j is the output of the j-th hidden unit.

Now, you can substitute these derivatives into the expression for ∂L/∂w_jk:

∂L/∂w_jk = -(1/n) * Σ[i=1 to n] [y_i * (1/ŷ_i) * ŷ_i * (1 - ŷ_i) * z_j - (1 - y_i) * (1/(1 - ŷ_i)) * (-ŷ_i) * (1 - ŷ_i) * z_j] + λ * w_jk

Simplify further:

∂L/∂w_jk = -(1/n) * Σ[i=1 to n] [y_i * (1 - ŷ_i) * z_j + (1 - y_i) * (-ŷ_i) * z_j] + λ * w_jk

Now, you can update the weight w_jk using a gradient descent update rule:

w_jk = w_jk - η * ∂L/∂w_jk

Where η is the learning rate.

**2. Weight Updates for Input-to-Hidden Weights (v_ij):**

Since the input-to-hidden weights are unaffected by the weight decay term, the update for v_ij is unchanged from the conventional backpropagation update and does not include a weight decay term. The conventional gradient descent rule would be used to update v_ij after calculating the gradient of the loss with respect to v_ij.

In conclusion, the weight updates for hidden-to-output weights contain an extra regularisation term in the gradient when the loss function is cross-entropy with a weight decay term, but the weight updates for input-to-hidden weights do not include the weight decay term.

Let's review the weight updates taking cross-entropy loss with weight decay into consideration for both input-to-hidden weights (v_ij) and hidden-to-output weights (w_jk):

**Weight Updates for Hidden-to-Output Weights (w_jk):**

The cross-entropy loss term and the weight decay term are both included in the update rule for w_jk:

w_jk = w_jk - η * ∂L/∂w_jk

Where:
- The gradient of the cross-entropy loss, which incorporates both the loss term and the weight decay term, is ∂L/∂w_jk with respect to w_jk.

The additional regularisation component (λ * w_jk) in the gradient represents the weight decay term.


**Weight Updates for Input-to-Hidden Weights (v_ij):**

The weight updates for input-to-hidden weights adhere to the conventional backpropagation update rule and do not contain the weight decay term:

v_ij = v_ij - η * ∂L/∂v_ij

Where:
- The gradient of the cross-entropy loss relative to v_ij, devoid of the weight decay term, is expressed as ∂L/∂v_ij.

This indicates that the weight updates for input-to-hidden weights are unaffected by the regularisation term (λ * v_ij), and the updates are carried out using the conventional gradient descent procedure.

In conclusion, weight updates for hidden-to-output weights include the weight decay term as an extra regularisation term in the gradient when using cross-entropy loss with a weight decay term in a neural network, whereas weight updates for input-to-hidden weights do not include the weight decay term and adhere to the conventional backpropagation update rules. This enables you to use weight decay regularisation to prevent overfitting and optimize the model's performance on the specified task at the same time.

6.	Consider a neural network with a single hidden layer with sigmoid activation functions and a single output unit also with a sigmoid activation, and fixed weights. Show that there exists an equivalent network, which computes exactly the same function, where the hidden unit activations are the $\tanh$ function shown in class, and the output unit still has a sigmoid activation. (10 points)

Answer:

7.	Draw an artificial neural network structure which can perfectly classify the examples shown in the table below. Treat attributes as continuous. Show all of the weights on the edges. For this problem, assume that the activation functions are sign functions instead of sigmoids. Propagate each example through your network and show that the classification is indeed correct.
(10 points)
 
|x1	|x2	|Class|
|---|---|-----|
|−4	|−4	|−|
|−1	|−1	|+|
| 1	| 1	|+|
| 4|  4	|−|

Answer:

Here is an artificial neural network structure that can perfectly classify the examples given in the table:

- Input Layer: 2 input neurons (x1 and x2)
- Hidden Layer: 1 hidden neuron with a sign activation function (we'll call it h)
- Output Layer: 1 output neuron with a sign activation function (we'll call it y)

The weights on the edges are as follows:
1. Weight from x1 to h: w1 = 1
2. Weight from x2 to h: w2 = 1
3. Weight from h to y: w3 = 1

The network structure looks like this:

 x1 (w1=1)
  |
 x2 (w2=1)
  |
  h (w3=1)
  |
  y


The sign activation function for the hidden neuron h and output neuron y is as follows:
- Sign Activation Function: The output is +1 in the case of a positive input and -1 in the case of a negative input.

Now, let's propagate the given examples through the network:

1. Example (4, 4, 1):
   - Input: x1 = 4, x2 = 4
   - Hidden Layer (h): sign(1 * 4 + 1 * 4) = sign(8) = 1
   - Output Layer (y): sign(1 * 1) = sign(1) = 1
   - The network classifies this example as Class 1.

2. Example (1, 1, 4):
   - Input: x1 = 1, x2 = 1
   - Hidden Layer (h): sign(1 * 1 + 1 * 1) = sign(2) = 1
   - Output Layer (y): sign(1 * 1) = sign(1) = 1
   - The network classifies this example as Class 4.

The network perfectly classifies both examples as specified in the table, showing that it correctly separates the two classes using the weights and sign activation functions.

8.	Using R/Matlab/Mathematica/python/your favorite software, plot the decision boundary for an ANN with two inputs, two hidden units and one output. All activation functions are sigmoids. Each layer is fully connected to the next. Assume the inputs range between −5 to 5 and fix all activation thresholds to 0. Plot the decision boundaries for  the weights except the thresholds randomly chosen between (i) (−10,10), (ii) (−3,3), (iii) (−0.1,0.1) (one random set for each case is enough). Use your plots to show that weight decay can be used to control overfitting for ANNs. (If you use Matlab, the following commands might be useful: meshgrid and surf). (20 points)

Answer:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Define the input range
x1_range = np.linspace(-5, 5, 400)
x2_range = np.linspace(-5, 5, 400)
X1, X2 = np.meshgrid(x1_range, x2_range)
input_data = np.c_[X1.ravel(), X2.ravel()]

# Define weight ranges
weight_ranges = [(-10, 10), (-3, 3), (-0.1, 0.1)]

# Initialize plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.subtitle('Decision Boundaries with Different Weight Ranges')

for i, weight_range in enumerate(weight_ranges):
    # Generate random weights within the specified range
    weights_hidden = np.random.uniform(weight_range[0], weight_range[1], size=(2, 2))
    weights_output = np.random.uniform(weight_range[0], weight_range[1], size=(2,))
    
    # Create and train a multi-layer perceptron classifier
    clf = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', max_iter=1000)
    clf.coefs_ = [weights_hidden, weights_output]
    clf.intercepts_ = [np.zeros(2), np.zeros(1)]
    clf.fit(input_data, np.zeros(len(input_data)))
    
    # Predict the decision boundary
    decision_boundary = clf.predict(input_data)
    decision_boundary = decision_boundary.reshape(X1.shape)
    
    # Plot the decision boundary
    axes[i].contourf(X1, X2, decision_boundary, levels=[-0.5, 0.5], cmap=plt.cm.RdBu, alpha=0.7)
    axes[i].set_xlim(-5, 5)
    axes[i].set_ylim(-5, 5)
    axes[i].set_xlabel('Input X1')
    axes[i].set_ylabel('Input X2')
    axes[i].set_title(f'Weight Range: {weight_range}')

plt.show()

9.	When learning the weights for the perceptron, we dropped the *sign* activation function to make the objective smooth. Show that the same strategy does not work for an arbitrary ANN. (Hint: consider the shape of the decision boundary if we did this.)  (10 points)

Answer:

A smooth activation function (such as the logistic sigmoid or hyperbolic tangent) is a better option when learning the weights for a perceptron than the sign activation function. Unfortunately, for several reasons, this method cannot be applied directly to any artificial neural network (ANN):

1. **Complex Decision Boundaries:** Complex and non-linear decision boundaries can be modeled by ANNs with several layers and non-linear activation functions. The activation functions and weights have a complicated function that transforms the decision boundary. A simple, smooth function may be used in place of non-linear activation functions to produce a decision boundary that may not be able to accurately depict the intended function.

2. **Loss Function:** Generally, gradient-based optimization techniques like backpropagation are used to train ANNs. The computation of gradients that drive weight updates during training depends on how smoothly the activation functions are fitted. It would be difficult to compute gradients—which are required for the convergence of optimization methods like gradient descent—if non-smooth activation functions, like the sign function, were used.

3. **Function Approximation:** In order to effectively capture and represent complicated processes, artificial neural networks (ANNs) frequently rely on the non-linear characteristics of activation functions. The network’s capacity to approximate intricate mappings may be restricted if these non-linearities are eliminated.

4. **Expressiveness:** The capacity of ANNs to recognize and learn from both linear and non-linear correlations in data is well recognized. The expressiveness of the network is greatly restricted when non-linear activation functions are swapped out for linear ones, which also lessens the network's capacity to recognize intricate patterns and correlations in data.

In conclusion, while a smooth activation function can effectively replace the sign activation function in a basic perceptron model, this approach is inappropriate for arbitrary artificial neural networks (ANNs). To capture complex patterns and relationships in data, artificial neural networks (ANNs) use numerous layers and non-linear activation functions. Modifying the activation functions can produce a fundamentally different and less expressive model.

