# CSDS440 Written Homework 2
**Instructions:** Each question is worth 10 points unless otherwise stated. Write your answers below the question. Each answer should be formatted so it renders properly on github. **Answers that do not render properly may not be graded.** Please comment the last commit with "FINAL COMMIT" and **enter the final commit ID in canvas by the due date.** 

When working as a group, only one answer to each question is needed unless otherwise specified. Each person in each group must commit and push their own work. **You will not get credit for work committed/pushed by someone else even if done by you.** Commits should be clearly associated with your name or CWRU ID (abc123). Each person is expected to do an approximately equal share of the work, as shown by the git logs. **If we do not see evidence of equal contribution from the logs for someone, their individual grade will be reduced.** 


1.	(i) Give an example of a nontrivial (nonconstant) Boolean function over $3$ Boolean attributes where $IG(X)$ would return zero for *all* attributes at the root. (ii) Explain the significance of this observation. 

Answer:
(i) Let us consider a Boolean function where,
     F(A,B,C) = (A AND B) OR (NOT A AND NOT B) OR (NOT C)
     Here in this function OR computes three terms, one where A and B are true, while in the next one both are not true and in the last C is not true. In this function we have deliberately designed it so that the information gain for attributes B and C will be 0 when used as a root attribute.
     1. Information Gain for A:
          First, we are getting an original dataset (D) and since each attribute can take 2 values(T or F), there rae 2^3 = 8 combinations. Also, the dataset is divided into two, D1 and D2 where A is true and false respectively.
     Entropy of the original dataset H(D),
     H(D) = -P(0/D)log2(P(0/D))-P(1/D)log2(P(1/D))
          = -4/8 log2(4/8) - 4/8 log2(4/8)
          = -0.5(-1) -0.5(-1)
          = 0.5+0.5 = 1
     So, D1 contains (1,1,0),(1,0,0),(1,1,1),(1,0,1).
     P(0/D1) = 1/4
     P(1/D1) = 3/4
     Entropy for D1: 
          H(D1) = -1/4 log2 (1/4) - 3/4 log2 (3/4) ≈ 0.8113
     now, D2 contains (0,1,0),(0,0,0),(0,1,1),(0,0,1).
     P(0/D2) = 1/4
     P(1/D2) = 3/4
     Entropy for D2:
          H(D2) = -1/4 log2(1/4) - 3/4log2(3/4) ≈ 0.8113
     Now, information gain for A is,
     IG(A) = H(D) - (4/8*0.8113 + 4/8*0.8113)
           = 1 - 0.8113 ≈ 0
     2. Information Gain for B:
        Here the entropy for D1 and D2 are the same as when calculating IG(A).
        So, H(D1) ≈ 0.8113
            H(D2) ≈ 0.8113
        Therefore, IG(B) = H(D) - (4/8*0.8113 + 4/8*0.8113) ≈ 0
     3. Information Gain for C:
        Here, D1 contains (0, 0, 1), (1, 1, 1), (0, 1, 1), (1, 0, 1). 
        P(0/D1) = 1/2
        P(1/D1) = 1/2
        Entropy for D1: 
          H(D1) = -1/2 log2 (1/2) - 1/2 log2 (1/2) = 1
        now, D2 contains (1, 1, 0), (0, 0, 0), (1, 0, 0), (0, 1, 0).
        Entropy for D2:
          P(0/D2) = 1/2
          P(1/D2) = 1/2
          Entropy for D2: 
            H(D2) = -1/2 log2 (1/2) - 1/2 log2 (1/2) = 1
          Now, information gain for C is,
            IG(C) = H(D) - (4/8*1 + 4/8*1)
                  = 1 - (0.5+0.5) = 0
                  
(ii)
     The significance of the observation is that for the given Boolean function and dataset, attributes A,B and C have IG(X) = 0 when used as root attributes for a decision tree. So in this specific context, none of these attributes provide any advantage in reducing uncertainty of separating data. The balanced distribution of the outcomes in the dataset H(D) =1 makes all attributes equally ineffective as root attributes for decision tree split.
          
     

2. Estimate how many functions satisfying Q1 (i) could exist over $n$ attributes, as a function of $n$. 

Answer:
 
3.	Show that for a continuous attribute $X$, the only split values we need to check to determine a split with max $IG(X)$ lie between points with different labels. (Hint: consider the following setting for $X$: there is a candidate split point $S$ in the middle of $N$ examples with the same label. To the left of $S$ are $n$ such examples. To the left of $N$, there are $L_0$ examples with label negative and the $L_1$ positive, and likewise $(M_0, M_1)$ to the right. Express the information gain of $S$ as a function of $n$. Then show that this function is maximized either when $n=0$ or $n=N$ with all else constant.) (20 points)

Answer:

4.	Write a program to sample a set of $N$ points from $(−1,1)^2$. Label the points using the classifier $y=sign(0.5x_1+0.5x_2)$. Generate datasets from your program and use your ID3 code from Programming 1 to learn trees on this data (there is no need to do cross validation or hold out a test set). Plot a graph where the $x$-axis is the value of $N$, over $N={50, 100, 500, 1000, 5000}$, and the $y$-axis is the depth of the tree learned by ID3. Explain your observations. (20 points)

Answer: 

5.	Show the decision boundaries learned by ID3 in Q4 for $N=50$ and $N=5000$ by generating an independent test set of size 100,000, plotting all the points and coloring them according to the predicted label from the $N=50$ and $N=5000$ trees. Explain what you see relative to the true decision boundary. What does this tell you about the suitability of trees for such datasets? (20 points)

Answer:

6.	Under what circumstances might it be beneficial to overfit? 

Answer:
     Since it results in poor generalization, overfitting is typically viewed as a concern in statistical modeling and machine learning. Overfitting, however, may be advantageous or at the very least acceptable when you want the model to be susceptible to rare and unusual patterns or it helps with anomaly detection.
     Secondly, when we have a very small dataset, it will be challenging to train a model that generalizes well, so in those cases, some level of overfitting may help the model capture the limited information available.
     Third, when your data contains a high level of noise or measurement errors then a more complex model with some level of overfitting may capture both the signal and noise. Though this might not help with generalization, it might provide better results with noisy data.
     
     
