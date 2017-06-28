__author__ = 'vikram'

# In this exercise, we'll use the Titanic dataset as before, train two classifiers and
# look at their confusion matrices. Your job is to create a train/test split in the data
# and report the results in the dictionary at the bottom.

import numpy as np
import pandas as pd

# Load the dataset
from sklearn import datasets

X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
from sklearn.metrics import f1_score

# TODO: split the data into training and testing sets,
# using the default settings for train_test_split (or test_size = 0.25 if specified).
# Then, train and test the classifiers with your newly split data instead of X and y.
Xtrain, Xtest, ytrain, ytest = cross_validation.train_test_split(X, y, test_size=0.25)
clf1 = DecisionTreeClassifier()
clf1.fit(Xtrain,ytrain)
print "Decision tree score is", clf1.score(Xtest, ytest)
print "Confusion matrix for this Decision Tree:\n", confusion_matrix(ytest, clf1.predict(Xtest))
print "Decision Tree recall: {:.2f} and \
precision: {:.2f}".format(recall(ytest, clf1.predict(Xtest)), precision(ytest, clf1.predict(Xtest)))

clf2 = GaussianNB()
clf2.fit(Xtrain, ytrain)
print "GNB score is", clf2.score(Xtest, ytest)
print "GaussianNB confusion matrix:\n", confusion_matrix(ytest, clf2.predict(Xtest))
print "GaussianNB recall: {:.2f} and \
precision: {:.2f}".format(recall(ytest, clf2.predict(Xtest)), precision(ytest, clf2.predict(Xtest)))

print "Decision Tree F1 score: {:.2f}".format(f1_score(ytest, clf1.predict(Xtest)))
print "NB F1 score: {:.2f}".format(f1_score(ytest, clf2.predict(Xtest)))

#TODO: store the confusion matrices on the test sets below

confusions = {
 "Naive Bayes": confusion_matrix(ytest, clf2.predict(Xtest)),
 "Decision Tree": confusion_matrix(ytest, clf1.predict(Xtest))
}

results = {
  "Naive Bayes Recall": recall(ytest, clf2.predict(Xtest)),
  "Naive Bayes Precision": precision(ytest, clf2.predict(Xtest)),
  "Decision Tree Recall": recall(ytest, clf1.predict(Xtest)),
  "Decision Tree Precision": precision(ytest, clf1.predict(Xtest))
}

F1_scores = {
 "Naive Bayes": f1_score(ytest, clf2.predict(Xtest)),
 "Decision Tree": f1_score(ytest, clf1.predict(Xtest))
}
