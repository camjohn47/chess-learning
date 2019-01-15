# Chess-Project
Immense curiosity in AI and chess inspired me to start a chess project three years ago. Originally, I coded the project in
Java, after which I authored another rough draft in Swift. Due to Python's tremendously convenient and elegant chess library, I've now mostly switched over to Python. Using Python's chess engine has dramatically saved time
and reduced code length and complexity. Also, Python's numpy library offers far more ML tools than found in Java, making the math used in the learning/optimization
algorithms much faster and easier/cleaner to code. Newest and best components of the project are mostly in Python, with select pieces, such 
as the game animation, feature extraction, and the alpha-beta pruning used for move optimization still working better in Java. The latter two 
appear to stem from the fact that Java is a faster programming language than Python. For the time being, only the Python code has been uploaded. 

At the core of the project is cost function selection, learning algorithms, and the AI's rapid calculation of optimal moves. Cost function selection 
refers to the following question: What form should a function take if it takes data from a chess board as input, and outputs a probability
of white winning the game? I suspect that a neural network could be a very powerful answer to this question, but for now, a logistic regression 
model is chosen. More specifically, let the result of the game be our output y, where y=0 is a defeat for white, y=0.5 is a draw, and y=1 is a victory
for white. Logistic regression, in the context of chess, predicts game outcome y using N inputs [x_1,...,x_N] = X extracted from the board, and N parameters 
[p_1,...,p_N] = P as follows: 

p(y|X,P) = 1 / (1 + exp(- sum_{i=1}^{N} ( x_i*p_i) ) ).

Note that a perfect logistic regression model almost surely doesn't exist for chess, because it's far too complicated. Hence why neural networks are likely needed to 
exceed performance past a certain threshold. Choosing which features to include, as well as how many (how big should N be), not only effects how well 
the above probability can be approximated, but run time as well. As the size and complexity of features increase, the maximum depth of the game tree search decreases.

In the above model, another question is how to choose the parameter P. Entropy is a deep mathematical concept, but in short, it encapsulates 
the uncertainty of a probability distribution. For example, if you have two coins such that one is fair and the other is very heavily weighted on one side, 
the former coin has relatively higher entropy. This is because it's more difficult to predict the outcome of the fair coin than the outcome of the weighted coin. 
Cross entropy is similar, except that it measures the uncertainty between two distributions (kind of). Specifically, cross entropy measures 
the number of bits needed (on average) to describe an event from one distribution using optimal code for another. It follows that cross entropy of 
the chess game results and our logistic model gives a notion of how similar they are, and thus how well our logistic model captures information 
about the results we're trying to predict. 

So we want minimal cross entropy between outputs y and our logistic model p(y|X,P). Through minimizing the cross entropy, we can determine 
optimal parameters for our logistic regression, thereby giving our optimal predictive model that can be used in the chess engine for playing games. 
Two separate optimization algorithms can be used for cross entropy optimization, as outlined below with the other parts of the project.

Here's what's included as of 1/15/19: 

1. Learn_Grad.py: A program which analyzes thousands of chess games and extracts certain key features from each one. Using this data, 
                  a stochastic gradient descent algorithm is executed on this data for optimizing the logistic regression. Stochastic gradient
                  descent works by randomly permuting the training data at each iteration, and applying gradient increments in the parameters
                  associated with each random training sample. Step size dictates how far each step is scaled. When convergence has been reached or the maximum iteration is reached, the optimal 
                  parameter is written to a chosen file that is used in the game program Game.py. 

2. Learn_Newton.py: Identical to Learn_Grad.py, except that newton's method is used instead. Newton's method typically requires far fewer iterations than
                  any gradient descent; optimal step size is known in closed form; and its use of higher derivatives often results in smoother convergence.
                  However, the inverse of the hessian matrix needs to be calculated at each step, which is an enormous matrix in our context. 

3. Game.py:       Script which executes the game in terminal. The game is currently NOT READY for solid play, as my old computer is unable to execute the learning algorithms 
                  properly on the training data. Moreover, transitions from alpha beta pruning are still being made. Game with animations should be up soon. 

4. Parameters.txt: Text file containing the optimized parameters. Note that if you use Learn_Grad or Learn_Newton with new training data 
                  to learn improved parameters, ensure that they are saved to this parameters.txt file. **NOTE: I cannot currently run either learning algorithm on my comuter with very much training data, so I have set the initial parameters to a naive configuration. 
       
If you have questions, please email me at camjohn@g.ucla.edu. Enjoy! 
