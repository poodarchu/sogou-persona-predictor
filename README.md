# Sogou Persona
the CCF Data Foundation Competition, predict the Sogou Search Engine's user profile.

A brief working flow:

0. split train set into Train.csv and Validation.csv(for example, 4:1), leave test.csv untouched. 
1. read data, cut tokens then remove stop tokens.
2. applying LDA to each user's query list, which is seen as a document.
3. after applying LDA, calculate each user's interest topics by wordNum/TopicNum, 
    which is saved as a 100-dimensional vector.
4. we can see the above 100 dimensional vector as features, Age, Gender and 
    Certificate is labels. then the task is use the 100 features to predict.
5. in the last step, I choose two ways:
    1. Support Vector Machine.
    2. Gradient Boosting Decision/Regression Tree.

------------------------------------------------------------------------

Another try is to use LSTMs to predict user labels.

1. read data, cut tokens then remove stop tokens.
2. build the word-index trans table. 
3. specify the vocabulary_size, for example, 10000, and all the words not 
    in the vocabulary is replaced with UNKNOWN_TOKEN.
4. use One-Hot encoding to represent each token of a user's query list. so 
    the query list of each user become a matrix of len(query_list) by vocabulary_size.
5. Build a 5 layers LSTM network, and input the above query matrix of each user 
    and the corresponding labels vector into it to produce the predictions.

------------------------------------------------------------------------

The above process will produce a qualified baseline, which means we can tune the details to gain improvements:

1. clean more irrelevant tokens, for example, the most frequent tokens.
2. add user dict to make more precision and meaningful cut result.
3. to be continued ...


Another way to try:
 
 **Word2Vec**



