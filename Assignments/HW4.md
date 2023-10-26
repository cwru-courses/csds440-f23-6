# CSDS440 Written Homework 3
**Instructions:** Each question is worth 10 points unless otherwise stated. Write your answers below the question. Each answer should be formatted so it renders properly on github. **Answers that do not render properly may not be graded.** Please comment the last commit with "FINAL COMMIT" and **enter the final commit ID in canvas by the due date.** 

When working as a group, only one answer to each question is needed unless otherwise specified. Each person in each group must commit and push their own work. **You will not get credit for work committed/pushed by someone else even if done by you.** Commits should be clearly associated with your name or CWRU ID (abc123). Each person is expected to do an approximately equal share of the work, as shown by the git logs. **If we do not see evidence of equal contribution from the logs for someone, their individual grade will be reduced.** 

Names and github IDs (if your github ID is not your name or Case ID):

1.	Show that the set $C=$ \{ $x|Ax\geq b$ \}, $A \in R^{m\times n}$, $x \in R^n$, $b \in R^m$, is a convex set. Note that this describes the constraint set of a linear program. (10 points)

Answer: 
The set C is convex. This means that for any two points x1 and x2 in C and for any λ where 0<=λ<=1 ( λ is a scalar that lies in the interval [0,1] ):

A(λx1+(1−λ)x2)>=b
provided that Ax1>=b and Ax2>=b.

Proof:

Given that x1 and x2 are in C:

1. Ax1>=b
2. Ax2>=b

Now consider the linear combination of these inequalities with coefficients λ and 1-λ respectively:

λAx1+(1−λ)Ax2>=λb+(1−λ)b

Using the distributive property, this becomes:
A(λx1+(1−λ)x2)≥λb+(1−λ)b

Here we can simplify the right-hand side to:
λb+(1-λ)b = b(λ+(1-λ)) = b(1)
= b

Now, because λb+(1-λ)b is just b, this inequality becomes:

A(λx1+(1−λ)x2)>=b

This proves our initial statement. For any two points x1 and x2 in C and for any λ in the interval [0,1], the convex combination λx1+(1-λ)x2 is also in C. So this confirms that C is convex.

2.	A function $f$ is said to have a global minimum at $x$ if for all $y$, $f(y) \geq f(x)$. It is said to have a local minimum at $x$ if there exists a neighborhood $H$ around $x$ so that for all $y$ in $H$, $f(y)\geq f(x)$. Show that, if $f$ is convex, every local minimum is a global minimum. [Hint: Prove by contradiction using Jensen’s inequality.] (10 points)

Answer: 

Given: $f$ is convex and has a local minimum at $x$.
We want to prove that $f$ has a global minimum at $x$.

Proof:
Assume for the sake of contradiction that  $f$ has a local minimum at $x$ bit, not a global minimum. This implies:
1. There exists a neighbourhood $H$ around $x$ such that for all $y$ in $H$, 

3.	Consider the LP: $\min c^Tx$ s.t. $Ax \geq b, x \geq 0$, where $T$ is the transpose, $A$ is the 4x2 matrix: \[ 0 −1; −1 −1; −1 2; 1 −1\], $b$ is a 4x1 vector \[−5; −9;0; −3\] and $c$ is a 2x1 vector \[−1; −2\]. (a) Draw the feasible region in $R^2$. (b) Draw the contours of $c^Tx =−12$, $c^Tx =−14$ and $c^Tx =−16$ and determine the solution graphically. (10 points)

Answer: 

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
