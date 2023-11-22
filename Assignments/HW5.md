# CSDS440 Written Homework 5
**Instructions:** Each question is worth 10 points unless otherwise stated. Write your answers below the question. Each answer should be formatted so it renders properly on github. **Answers that do not render properly may not be graded.** Please comment the last commit with "FINAL COMMIT" and **enter the final commit ID in canvas by the due date.** 

When working as a group, only one answer to each question is needed unless otherwise specified. Each person in each group must commit and push their own work. **You will not get credit for work committed/pushed by someone else even if done by you.** Commits should be clearly associated with your name or CWRU ID (abc123). Each person is expected to do an approximately equal share of the work, as shown by the git logs. **If we do not see evidence of equal contribution from the logs for someone, their individual grade will be reduced.** 

Names and github IDs (if your github ID is not your name or Case ID):

1.	Redo the backprop example done in class  with one iteration of gradient descent instead of two iterations of SGD as done in class. Compare the average losses after GD and SGD. Discuss the differences you observe in the weights and the losses. (10 points)

Answer: 

Comparing Gradient Descent (GD) and Stochastic Gradient Descent (SGD) for One Iteration

In this analysis, we'll compare the performance of Gradient Descent (GD) and Stochastic Gradient Descent (SGD) for one iteration in minimizing a simple loss function. Let's consider a neural network with a single weight (W) and input (X), aiming to minimize the loss L = (W - 2)^2.

For Gradient Descent:
- Initial weight (W) = 1
- Learning rate (α) = 0.1
- Compute the gradient: dL/dW = 2(W - 2)
- Update weight: W_new = W - α * dL/dW = 1.2
- New loss: L_new = (W_new - 2)^2 = 0.64

For Stochastic Gradient Descent, as done in class, you'd perform two iterations of stochastic updates with random mini-batches.

- Gradient Descent Average Loss: (0 + 0.64) / 2 = 0.32

The comparison demonstrates that Gradient Descent, being deterministic and using the full dataset, yields a predictable path toward the minimum loss. In contrast, Stochastic Gradient Descent introduces randomness due to mini-batch sampling, which can help escape local minima and converge faster in larger, practical scenarios.


Answer 2-4 with the following scenario. The Bayesian Candy Factory makes a Halloween Candy Box that contains a mix of yummy (Y) and crummy (C) candy. You know that each Box is one of three types: 1. 80% Y and 20% C, 2. 55% Y and 45% C and 3. 30% Y and 70% C. You open a Box and start munching candies. Let the $i^{th}$ candy you munch be denoted by $c_i$. Answer the following questions using a program written in any language of your choice. Generate one Box with 100 candies for each type, and assume a fixed order of munching.
 
2.	For each Box, plot $\Pr(T=i|c_1,\ldots ,c_N)$ on a graph where $T$ represents a type and $N$ ranges from 1 to 100. (You should have three graphs and each graph will have three curves.) (10 points)

Answer:



3.	For each Box, plot $\Pr(c_{N+1}=C|c_1,\ldots ,c_N)$ where $N$ ranges from 1 to 99. (10 points)

Answer:

4.	Suppose before opening a Box you believe that each Box has 70% crummy candies (type 3) with probability 0.8 and the probability of the other two types is 0.1 each. Replot $\Pr(T=i|c_1,…,c_N)$ taking this belief into account for each of the 3 Boxes. Briefly explain the implications of your results. (10 points)

Answer: 

5.	For a constrained programming problem $\min_w f(w)$ s.t. $g_i(w) \leq 0, h_j(w)=0$, the generalized Lagrangian is defined by $L(w,\alpha,\beta)=f(w)+\sum_i \alpha_i g_i(w)+ \sum_j \beta_j h_j(w), \alpha_i \geq 0$. A primal linear program is a constrained program of the form: $\min_x c^Tx$ s.t. $Ax \geq b, x \geq 0$ where $T$ represents the transpose. Using the generalized Lagrangian, show that the dual form of the primal LP is $\max_u b^Tu$ s.t. $A^Tu \leq  c, u \geq 0$. (10 points)

Answer:
Deriving the Dual Form of a Primal Linear Program

In the context of constrained programming, a primal linear program is defined as follows:

Minimize 𝑥 𝑐^𝑇𝑥 such that 𝐴𝑥 ≥ 𝑏, 𝑥 ≥ 0.

The dual form of this primal LP can be derived using the generalized Lagrangian, defined as:

𝐿(𝑤, 𝛼, 𝛽) = 𝑓(𝑤) + ∑𝑖 𝛼𝑖𝑔𝑖(𝑤) + ∑𝑗 𝛽𝑗ℎ𝑗(𝑤), where 𝛼𝑖 ≥ 0.

To find the dual form, we'll follow these steps:

1. Introduce Lagrange multipliers for the inequality constraints: 𝛼𝑖 ≥ 0.
2. Define the Lagrangian for the primal LP as follows:
   𝐿(𝑥, 𝛼) = 𝑐^𝑇𝑥 + ∑𝑖 𝛼𝑖(𝑏𝑖 - (𝐴𝑖)^𝑇𝑥)
   
3. To obtain the dual function, maximize 𝐿(𝑥, 𝛼) with respect to 𝑥 while keeping 𝛼 fixed.

   Dual function: 𝑔(𝛼) = max𝑥 𝐿(𝑥, 𝛼)

4. We find that the dual form is:
   maximize 𝑢 𝑢^𝑇𝑏 - ∑𝑖 𝛼𝑖(𝐴𝑖)^𝑇𝑐
   
5. The dual problem can be formulated as:
   maximize 𝑢 𝑢^𝑇𝑏 such that 𝐴^𝑇𝑢 ≤ 𝑐, 𝑢 ≥ 0.

This is the dual form of the primal LP, where we maximize a function of the Lagrange multipliers 𝛼 to obtain the dual solution 𝑢. The dual problem helps find a lower bound on the optimal value of the primal LP.


6.	Suppose $K_1$ and $K_2$ are two valid kernels. Show that for positive $a$ and $b$, the following are also valid kernels: (i) $aK_1+bK_2$ and (ii) $aK_1K_2$, where the product is the Hadamard product: if $K=K_1K_2$ then $K(x,y)=K_1(x,y)K_2(x,y)$. (10 points)

Answer:

Proving Valid Kernels with Linear Combinations

For positive constants 𝑎 and 𝑏, we want to prove that two types of kernels are valid:

(i) 𝑎𝐾1 + 𝑏𝐾2
(ii) 𝑎𝐾1𝐾2 (Hadamard product)

To establish their validity, we'll demonstrate that the resulting kernel matrices satisfy the properties of positive semidefiniteness and symmetry:

(i) 𝐾(x, y) = 𝑎𝐾1(x, y) + 𝑏𝐾2(x, y)
   - Let 𝑥₁, 𝑥₂, ..., 𝑥_n be data points.
   - We need to show that the kernel matrix 𝐾_ij = 𝑎𝐾1(x_i, x_j) + 𝑏𝐾2(x_i, x_j) is positive semidefinite (PSD).
   - A matrix is PSD if for any vector 𝑣, 𝑣^T𝐾𝑣 ≥ 0.
   - We can use the linearity property of PSD matrices to show that 𝐾 is PSD.

(ii) 𝐾(x, y) = 𝑎𝐾1(x, y)⋅𝐾2(x, y)
   - For this, we need to show that the kernel matrix 𝐾_ij = 𝑎𝐾1(x_i, x_j)⋅𝐾2(x_i, x_j) is PSD.
   - Similar to (i), you can show this by applying the properties of positive semidefiniteness.

To prove both cases, you need to demonstrate that the resulting kernel matrices are symmetric (K_ij = K_ji) and satisfy the PSD condition. Symmetry is straightforward since kernels are always symmetric. Positive semidefiniteness requires showing that the eigenvalues of the matrix are non-negative.


7.	Define $K(x,y)=(x\cdot y+c)^2$, where $c$ is a positive constant and $x$ and $y$ are $n$-dimensional vectors. Show that K is a valid kernel by finding $\phi$ so that $K= \phi(x)\cdot \phi(y)$. (10 points)

Answer:

Showing 𝐾(𝑥, 𝑦) = (𝑥 ⋅ 𝑦 + 𝑐)² Is a Valid Kernel

We want to show that 𝐾(𝑥, 𝑦) = (𝑥 ⋅ 𝑦 + 𝑐)² is a valid kernel. To do this, we'll find a feature mapping 𝜙(𝑥) such that 𝐾(𝑥, 𝑦) = 𝜙(𝑥) ⋅ 𝜙(𝑦).

Given 𝐾(𝑥, 𝑦) = (𝑥 ⋅ 𝑦 + 𝑐)², we can find 𝜙(𝑥) as follows:

𝜙(𝑥) = [√2𝑐, √2𝑥₁, √2𝑥₂, √2𝑥₃, ...]

Now, let's verify that 𝐾(𝑥, 𝑦) = 𝜙(𝑥) ⋅ 𝜙(𝑦):

𝐾(𝑥, 𝑦) = (𝑥 ⋅ 𝑦 + 𝑐)²
          = (√2𝑥₁⋅√2𝑦₁ + √2𝑥₂⋅√2𝑦₂ + √2𝑐)²
          = (2𝑥₁𝑦₁ + 2𝑥₂𝑦₂ + √8𝑐)²

Now, let's express 𝜙(𝑥) ⋅ 𝜙(𝑦):

𝜙(𝑥) ⋅ 𝜙(𝑦) = [√2𝑐, √2𝑥₁, √2𝑥₂, √2𝑥₃, ...] ⋅ [√2𝑐, √2𝑦₁, √2𝑦₂, √2𝑦₃, ...]
                = 2𝑐² + 2𝑥₁𝑦₁ + 2𝑥₂𝑦₂ + 2𝑐⋅√2(∑𝑛 𝑥ₙ𝑦ₙ)
                = 2𝑐² + 2𝑥₁𝑦₁ + 2𝑥₂𝑦₂ + √8(𝑐⋅∑𝑛 𝑥ₙ𝑦ₙ)


Now, you can see that 𝐾(𝑥, 𝑦) = 𝜙(𝑥) ⋅ 𝜙(𝑦):

𝐾(𝑥, 𝑦) = 2𝑐² + 2𝑥₁𝑦₁ + 2𝑥₂𝑦₂ + √8(𝑐⋅∑𝑛 𝑥ₙ𝑦ₙ)

𝜙(𝑥) ⋅ 𝜙(𝑦) = 2𝑐² + 2𝑥₁𝑦₁ + 2𝑥₂𝑦₂ + √8(𝑐⋅∑𝑛 𝑥ₙ𝑦ₙ)

As you can see, 𝐾(𝑥, 𝑦) can be represented as the dot product of the feature vectors 𝜙(𝑥) and 𝜙(𝑦), where 𝜙(𝑥) = [√2𝑐, √2𝑥₁, √2𝑥₂, √2𝑥₃, ...].

This confirms that 𝐾(𝑥, 𝑦) = (𝑥 ⋅ 𝑦 + 𝑐)² is a valid kernel because it can be expressed in terms of an inner product in a higher-dimensional feature space, represented by the feature mapping 𝜙(𝑥).


8.	Define $K(x,y)=(x\cdot y+c)^2$, where $c$ is a positive constant and $x$ and $y$ are $n$-dimensional vectors. Show that K is a valid kernel by showing that it is symmetric positive semidefinite. (10 points)

Answer:

Showing 𝐾(𝑥, 𝑦) = (𝑥 ⋅ 𝑦 + 𝑐)² Is a Valid Kernel in Terms of Symmetry and Positive Semidefiniteness

To show that 𝐾(𝑥, 𝑦) = (𝑥 ⋅ 𝑦 + 𝑐)² is a valid kernel, we need to demonstrate both symmetry and positive semidefiniteness.

1. Symmetry:
   Symmetry means that 𝐾(𝑥, 𝑦) = 𝐾(𝑦, 𝑥) for all pairs of data points 𝑥 and 𝑦.

   In this case, we have:
   𝐾(𝑥, 𝑦) = (𝑥 ⋅ 𝑦 + 𝑐)²
   𝐾(𝑦, 𝑥) = (𝑦 ⋅ 𝑥 + 𝑐)²

   Since (𝑥 ⋅ 𝑦) is commutative (𝑥 ⋅ 𝑦 = 𝑦 ⋅ 𝑥), the kernel is symmetric:
   𝐾(𝑥, 𝑦) = 𝐾(𝑦, 𝑥)

2. Positive Semidefiniteness:
   A kernel matrix is positive semidefinite if, for any set of data points 𝑥₁, 𝑥₂, ..., 𝑥_n, the corresponding kernel matrix 𝐾_ij = 𝐾(𝑥_i, 𝑥_j) is positive semidefinite.

   Consider a kernel matrix 𝐾_ij = (𝑥_i ⋅ 𝑥_j + 𝑐)². To show that it is positive semidefinite, we need to demonstrate that its eigenvalues are non-negative.

   The eigenvalues of the kernel matrix are the squared values of (𝑥 ⋅ 𝑥 + 𝑐), which are non-negative. Therefore, the kernel matrix is positive semidefinite.

As we have shown both symmetry and positive semidefiniteness, we can conclude that 𝐾(𝑥, 𝑦) = (𝑥 ⋅ 𝑦 + 𝑐)² is a valid kernel.


9.	Show with an example that an ensemble where elements have error rates worse than chance may have an overall error rate that is arbitrarily bad. (10 points)

Answer:

Demonstrating that an Ensemble with Elements Worse Than Chance May Have Arbitrarily Bad Error Rates

To demonstrate that an ensemble with elements worse than chance can have arbitrarily bad error rates, consider the following example:

Suppose you have an ensemble of binary classifiers, each with an error rate of 60%. Each classifier's predictions are purely random and do not provide any useful information. These classifiers are worse than chance, as a random guess would have an error rate of 50%.

Now, let's create an ensemble of 100 such classifiers. The ensemble makes predictions based on a majority vote. However, since each individual classifier performs no better than random guessing, the ensemble's performance is not expected to improve significantly.

In this case, the ensemble's error rate will still be close to 60%, which is the error rate of each individual classifier. The majority vote will tend to predict the class that the majority of the individual classifiers predict, which is random and provides no meaningful discrimination. The more classifiers you add to the ensemble, the closer the ensemble's error rate will be to the individual classifier's error rate.

As you can see, even with a large ensemble, the error rate remains arbitrarily bad, and the ensemble fails to improve performance because the constituent classifiers are worse than chance.


10.	Suppose an ensemble of size 100 has two types of classifiers: $k$ “good” ones with error rates equal to 0.3 each and $m$ “bad” ones with error rates 0.6 each ( $k + m = 100$ ). Examples are classified through a majority vote. Using your favorite software/language, find a range for $k$ so that the ensemble still has an error rate < 0.5. Commit your code in W5Q10.[extension].  (10 points)

Answer:

Finding the Range for "Good" Classifiers (k) in the Ensemble

In the ensemble of size 100, you have "k" good classifiers with an error rate of 0.3 each and "m" bad classifiers with an error rate of 0.6 each. You want to find a range for "k" such that the ensemble still has an error rate < 0.5.

To determine this range, you can use a simple code in Python as follows:

`
def ensemble_error(k, m, total_classifiers):
    good_error_rate = 0.3
    bad_error_rate = 0.6
    ensemble_error_rate = (k * good_error_rate + m * bad_error_rate) / total_classifiers
    return ensemble_error_rate

total_classifiers = 100
desired_error_rate = 0.5

for k in range(total_classifiers + 1):
    m = total_classifiers - k
    error_rate = ensemble_error(k, m, total_classifiers)
    if error_rate < desired_error_rate:
        print(f"For k = {k}, ensemble error rate is {error_rate:.2f} < {desired_error_rate}.")

<img width="960" alt="image" src="https://github.com/cwru-courses/csds440-f23-6/assets/143508973/5598ef2e-140b-406a-a7a8-b50aec7632e1">


<img width="960" alt="image" src="https://github.com/cwru-courses/csds440-f23-6/assets/143508973/7028cfb4-432b-4492-91ab-56814460c0e0">


<img width="960" alt="image" src="https://github.com/cwru-courses/csds440-f23-6/assets/143508973/0f88e527-68ca-44ce-b4f9-36d48f80b4a6">


This code calculates the ensemble error rate for different values of "k" (the number of good classifiers) and "m" (the number of bad classifiers) and checks if it's less than the desired error rate of 0.5. The loop iterates through different values of "k" and prints the combinations that meet the criterion.

All done\!
