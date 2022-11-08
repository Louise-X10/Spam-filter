# Spam-filter

The `build_naive_bayes_classifier` function takes training data and returns a `nbc` matrix of conditional probabilities for each feature. Each row corresponds to a feature, which are all binary in this example. The 1st column is the probability of feature = positive given label = positive. The 2nd column is the probability of feature = negative given label = positive. The 3rd and 4th column are defined similarly except for the condition being label = negative. It uses Laplace smoothing with parameter 1 to fix the issue of zero counts. It also returns the proportion of positive labeled observations.

The `classify` function takes testing data, the `nbc` matrix, and `pos_prob` the proportion of positive datapoints. It calculates the probability of each test datapoint belonging to a positive or negative label, and classifies by taking the argmax.  It uses sum of log probabilities instead of product of probabilities to avoid numeric underflow. These predictions are stored and compared to the actual labels to compute the precision and recall values. 

The first partition size is 80 training data and 20 testing data. The remaining partition sizes are sequential multiples of 100, each splitting into training and testing with 80% and 20% split. The last partition size uses the most of the available data. Finally, an ROC plot showing the precision and recall values for each partition size is displayed.

If an animated plot is desired, comment the code under the section `create static plot` and uncomment the code under `create animated plot`.
