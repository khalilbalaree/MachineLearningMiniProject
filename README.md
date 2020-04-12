# MachineLearningMiniProject
Cmput466 mini-project

## Main task:  
Apply three classification algorithms (logistic regression, neural network and soft-margin SVM) on a machine learning problem of predicting spam emails. The goal of this task is to make accuracy of the prediction as high as possible, the loss of each algorithm as low as possible.

## Dataset background: 
1. Url: http://archive.ics.uci.edu/ml/datasets/spambase
2. Name: UCI Spambase Data Set
3. Abstract: Classifying Email as Spam or non-Spam
4. Brief introduction: This is a pre-processed dataset and the raw data are already featurized. There are 4600 samples,1813 Spam (39.4%) and 2788 non-spam (60.6%), each has 57 features and 1 nominal class label. The detail of the 57 features is in the document of this data set (spambase/spambase.Documentation).

## Data preparation:
Load the data set from spambase/spambase.data line by line and split each line to 57 features adding a trivial feature x0 = 1 as input X and 1 label as y. The raw label is 1(spam) or 0(non-spam). Shuffle X and y and split them into 2800 train samples, 900 validation samples and 900 test samples(nearly 3:1:1).
