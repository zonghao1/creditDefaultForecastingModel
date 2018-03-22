# Fit logistic regression
from sklearn import datasets, linear_model

regr = linear_model.LogisticRegression()

regr.fit(traning.iloc[:,:-1],traning.iloc[:,-1])

errorRate = 1 - regr.score(test.iloc[:,:-1],test.iloc[:,-1])
errorRate
