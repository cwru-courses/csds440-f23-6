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

8.	Using R/Matlab/Mathematica/python/your favorite software, plot the decision boundary for an ANN with two inputs, two hidden units and one output. All activation functions are sigmoids. Each layer is fully connected to the next. Assume the inputs range between −5 to 5 and fix all activation thresholds to 0. Plot the decision boundaries for  the weights except the thresholds randomly chosen between (i) (−10,10), (ii) (−3,3), (iii) (−0.1,0.1) (one random set for each case is enough). Use your plots to show that weight decay can be used to control overfitting for ANNs. (If you use Matlab, the following commands might be useful: meshgrid and surf). (20 points)

Answer:

9.	When learning the weights for the perceptron, we dropped the *sign* activation function to make the objective smooth. Show that the same strategy does not work for an arbitrary ANN. (Hint: consider the shape of the decision boundary if we did this.)  (10 points)

Answer:

A smooth activation function (such as the logistic sigmoid or hyperbolic tangent) is a better option when learning the weights for a perceptron than the sign activation function. Unfortunately, for several reasons, this method cannot be applied directly to any artificial neural network (ANN):

1. **Complex Decision Boundaries:** Complex and non-linear decision boundaries can be modeled by ANNs with several layers and non-linear activation functions. The activation functions and weights have a complicated function that transforms the decision boundary. A simple, smooth function may be used in place of non-linear activation functions to produce a decision boundary that may not be able to accurately depict the intended function.

2. **Loss Function:** Generally, gradient-based optimization techniques like backpropagation are used to train ANNs. The computation of gradients that drive weight updates during training depends on how smoothly the activation functions are fitted. It would be difficult to compute gradients—which are required for the convergence of optimization methods like gradient descent—if non-smooth activation functions, like the sign function, were used.

3. **Function Approximation:** In order to effectively capture and represent complicated processes, artificial neural networks (ANNs) frequently rely on the non-linear characteristics of activation functions. The network’s capacity to approximate intricate mappings may be restricted if these non-linearities are eliminated.

4. **Expressiveness:** The capacity of ANNs to recognize and learn from both linear and non-linear correlations in data is well recognized. The expressiveness of the network is greatly restricted when non-linear activation functions are swapped out for linear ones, which also lessens the network's capacity to recognize intricate patterns and correlations in data.

In conclusion, while a smooth activation function can effectively replace the sign activation function in a basic perceptron model, this approach is inappropriate for arbitrary artificial neural networks (ANNs). To capture complex patterns and relationships in data, artificial neural networks (ANNs) use numerous layers and non-linear activation functions. Modifying the activation functions can produce a fundamentally different and less expressive model.

