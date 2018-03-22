
# Fit decision tree classifier and set tuning parameter with values provided 
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=8,min_samples_leaf=20,min_impurity_split=0.01)
clf.fit(traning.iloc[:,:-1],traning.iloc[:,-1])


errorRate2 = 1 - clf.score(test.iloc[:,:-1],test.iloc[:,-1])
errorRate2