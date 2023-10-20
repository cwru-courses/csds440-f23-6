# CSDS440 Written Homework 3
**Instructions:** Each question is worth 10 points unless otherwise stated. Write your answers below the question. Each answer should be formatted so it renders properly on github. **Answers that do not render properly may not be graded.** Please comment the last commit with "FINAL COMMIT" and **enter the final commit ID in canvas by the due date.** 

When working as a group, only one answer to each question is needed unless otherwise specified. Each person in each group must commit and push their own work. **You will not get credit for work committed/pushed by someone else even if done by you.** Commits should be clearly associated with your name or CWRU ID (abc123). Each person is expected to do an approximately equal share of the work, as shown by the git logs. **If we do not see evidence of equal contribution from the logs for someone, their individual grade will be reduced.** 


Names and github IDs (if your github ID is not your name or Case ID):


1.	Consider the following table of examples over Boolean attributes, annotated with the target concept's label. Ignore the "Weight" column and use information gain to find the first split in a decision tree (remember that ID3 stops if there is no information gain). You can use your code/a numerical package like Matlab to do this, and just report the final result.(10 points)

|A1|	A2|	A3|	A4|	Label|	Weight|
|---|---|---|---|---|---|
|F|	F|	F|	F|	0	|1/256|
|F	|F	|F	|T|	0	|3/256|
|F	|F	|T	|F	|1	|3/256|
|F	|F	|T	|T	|1	|9/256|
|F	|T	|F	|F	|1	|3/256|
|F	|T	|F	|T	|1	|9/256|
|F	|T	|T	|F	|0	|9/256|
|F	|T	|T	|T	|0	|27/256|
|T	|F	|F	|F	|1	|3/256|
|T	|F	|F	|T	|1	|9/256|
|T	|F	|T	|F	|0	|9/256|
|T	|F	|T	|T	|0	|27/256|
|T	|T	|F	|F	|0	|9/256|
|T	|T	|F	|T	|0	|27/256|
|T	|T	|T	|F	|1	|27/256|
|T	|T	|T	|T	|1	|81/256|

Answer:

2.	Now from the same table, find another split using "weighted" information gain. In this case, instead of counting the examples for each label in the information gain calculation, add the numbers in the Weight column for each example. You can use your code/a numerical package like Matlab to do this, and just report the final result. (10 points)

Answer:

3.	There is a difference between the splits for Q1 and Q2. Can you explain what is happening? (10 points)

Answer:

4.	Restriction biases of learning algorithms prevent overfitting by restricting the hypothesis space, while preference biases prevent overfitting by preferring simpler concepts but not necessarily restricting the hypothesis space. Discuss the pros and cons of preference vs restriction biases. (10 points)

Answer:
  Learning algorithms use preference and restriction biases as two tactics to prevent overfitting, although they work differently and have advantages and disadvantages of their own. 

5.	Person X wishes to evaluate the performance of a learning algorithm on a set of $n$ examples ( $n$ large). X employs the following strategy:  Divide the $n$ examples randomly into two equal-sized disjoint sets, A and B. Then train the algorithm on A and evaluate it on B. Repeat the previous two steps for $N$ iterations ( $N$ large), then average the $N$ performance measures obtained. Is this sound empirical methodology? Explain why or why not. (10 points)

Answer: 

6.	Two classifiers A and B are evaluated on a sample with P positive examples and N negative examples and their ROC graphs are plotted. It is found that the ROC of A dominates that of B, i.e. for every FP rate, TP rate(A) $\geq$ TP rate(B). What is the relationship between the precision-recall graphs of A and B on the same sample? (10 points)

Answer: 

7.	Prove that an ROC graph must be monotonically increasing. (10 points)

Answer: 
An ROC curve is constructed using two metrics: True Positive Rate $(TPR$) and False Positive Rate $(FPR)$

Y-Axis: $TPR$, 
X-Axis: $FPR$

Each new point along the curve is $(TPR_{n}, FPR_{n})$.

We go from $(TPR_{n}, FPR_{n}) \rightarrow (TPR_{n+1}, FPR_{n+1})$ by decreasing the confidence threshold between examples $n$ and $n+1$ .

As the threshold decreases along the examples, instances that were previously classified as negative may now be classified as positive. 

Some of these new positives may be true: increasing $TPR$

Some of these new positives may be false: increasing $FPR$

Either case, causes the curve to move up, to the right, or both. 

The only case in which the curve moves downwards, is if $TPR$ decreased while $FPR$ increased. Given that $TPR$ only ever increases with a decreasing confidence threshold, this case is impossible. 

Therefore, the ROC curve is monotonically increasing, as it is impossible for a decrease in the confidence threshold to cause a decrease in $TPR$.



8.	Prove that the ROC graph of a random classifier that ignores attributes and guesses each class with equal probability is a diagonal line. (10 points)

Answer: 

Outputs of a random binary classifier: 
$$
\hat{Y} = \begin{bmatrix} \hat{y}_1 \\ \hat{y}_2 \\ \vdots \\ \hat{y}_n \end{bmatrix} 
$$
$$P(\hat{Y}=\hat{y}) = 
\begin{cases} 
0.5 & \text{if } \ \hat{y} = 1, \\
0.5 & \text{if } \ \hat{y} = 0.
\end{cases}
$$

It is a coin flip whether $\hat{y}$, the label assigned to some example will be positive or negative. 

The two metrics that we use to construct an ROC curve are: 

True Positive Rate $(TPR$) = $\frac{TP}{TP+FN}$
False Positive Rate $(FPR)$ = $\frac{FP}{FP + TN}$

If the distribution of labels for a given set of $n$ examples is perfectly random. Both the metrics $(TPR, FPR) \rightarrow (0.5, 0.5)$

|Examples|	True Label|	Confidence|	$TPR$|	$FPR$|
|---|---|---|---|---|
|$x_{1}$|-/+|$c_{1}$|0.5|0.5|
|$x_{2}$|-/+|$c_{2} = c_{1}-\epsilon$|0.5|0.5|	
|$\vdots$|-/+|$c_{i} = c_{i-1}-\epsilon$|0.5|0.5|
|$x_{n}$|-/+|$c_{n} = c_{n-1}-\epsilon$|0.5|0.5|


$TPR$ Metric Analysis: Probability of guessing a label correctly is 0.5 therefore all true positive instances $TP = \frac{1}{2}(AllPos)$ and the rest of the positive labels will be missed, $FN = \frac{1}{2}(AllPos)$ meaning $TPR = \frac{0.5}{0.5+0.5} = 0.5$ in all possible examples

$FPR$ Metric Analysis: Probability of guessing a label correctly is 0.5 therefore all true negative instances $TN = \frac{1}{2}(AllNeg)$ and the rest of the negative labels will be missed, $FP = \frac{1}{2}(AllPos)$ meaning $FPR = \frac{0.5}{0.5+0.5} = 0.5$ in all possible examples

This gives our ROC curve a constant slope of $\frac{\frac{1}{2}}{\frac{1}{2}}= 1$
A constant slope of equal $x$ and $y$ axis growth gives us a diagonal line. 
