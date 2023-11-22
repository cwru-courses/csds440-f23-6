# CSDS440 Written Homework 5
**Instructions:** Each question is worth 10 points unless otherwise stated. Write your answers below the question. Each answer should be formatted so it renders properly on github. **Answers that do not render properly may not be graded.** Please comment the last commit with "FINAL COMMIT" and **enter the final commit ID in canvas by the due date.** 

When working as a group, only one answer to each question is needed unless otherwise specified. Each person in each group must commit and push their own work. **You will not get credit for work committed/pushed by someone else even if done by you.** Commits should be clearly associated with your name or CWRU ID (abc123). Each person is expected to do an approximately equal share of the work, as shown by the git logs. **If we do not see evidence of equal contribution from the logs for someone, their individual grade will be reduced.** 

Names and github IDs (if your github ID is not your name or Case ID):

1.	Redo the backprop example done in class  with one iteration of gradient descent instead of two iterations of SGD as done in class. Compare the average losses after GD and SGD. Discuss the differences you observe in the weights and the losses. (10 points)

Answer: 

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

7.	Define $K(x,y)=(x\cdot y+c)^2$, where $c$ is a positive constant and $x$ and $y$ are $n$-dimensional vectors. Show that K is a valid kernel by finding $\phi$ so that $K= \phi(x)\cdot \phi(y)$. (10 points)

Answer:

8.	Define $K(x,y)=(x\cdot y+c)^2$, where $c$ is a positive constant and $x$ and $y$ are $n$-dimensional vectors. Show that K is a valid kernel by showing that it is symmetric positive semidefinite. (10 points)

Answer:


9.	Show with an example that an ensemble where elements have error rates worse than chance may have an overall error rate that is arbitrarily bad. (10 points)

Answer:

10.	Suppose an ensemble of size 100 has two types of classifiers: $k$ “good” ones with error rates equal to 0.3 each and $m$ “bad” ones with error rates 0.6 each ( $k + m = 100$ ). Examples are classified through a majority vote. Using your favorite software/language, find a range for $k$ so that the ensemble still has an error rate < 0.5. Commit your code in W5Q10.[extension].  (10 points)

Answer:


All done\!
