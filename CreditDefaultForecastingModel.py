import pandas as pd
import numpy as np


#load the data
# Please change path to the location of CreditModelData1.csv in your computer to preceed further steps!
path = 'CreditModelData1.csv'
data1 = pd.read_csv(path,engine='python')
data1.shape

#change dependent variable into 0/1
default = pd.Series(data1['S&P Entity Credit Rating Action [3/3/2008-3/5/2018]'])

default1 = np.where(default=='-', 0, 1)

del data1['S&P Entity Credit Rating Action [3/3/2008-3/5/2018]']


# The original dataset use '-' and 'NM' for missing data, change them into np.NaN 
isUnderLine = data1 == str('-')
isNM = data1 ==str('NM')

data1 = data1.mask(isUnderLine,np.NaN)
data1 = data1.mask(isNM,np.NaN)

data1['default'] = default1

data1 = data1.drop('Company Type',axis=1)



# Delete all columns that contain all missing value
for name in data1.columns[data1.isnull().all()]:
    del data1[name]


# Delete all rows that contain all missing value
numbers = data1.iloc[:,3:-2]
emptyRow = []
for row in range(len(data1)):
    if(numbers.iloc[row].isnull().all()):
        emptyRow.append(row)


data1 = data1.drop(emptyRow,axis=0)

data1.index = np.arange(data1.shape[0])



# Change the data type from string into float in order to do mathematical operation
data1.iloc[:,3:75] = data1.iloc[:,3:75].astype(np.float)


np.sum(data1.isnull()==False, axis=0)


# We found out that all columns for free operating cash flow/debt and current ratio were deleted except two that only containg 438 and 448 values. Since we could not predict missing values for these categories using past year/previous year's record, and we want to keep these columns. We fill these columns with column mean. 


data1['Free Operating Cash Flow/Debt (%), Adj. - Credit Stats Direct [LTM]'] = data1['Free Operating Cash Flow/Debt (%), Adj. - Credit Stats Direct [LTM]'].fillna(data1['Free Operating Cash Flow/Debt (%), Adj. - Credit Stats Direct [LTM]'].mean())

data1['Free Operating Cash Flow/Debt (%), Adj. - Credit Stats Direct [LTM]'] = data1['Current Ratio (%), Adj. - Credit Stats Direct [LTM]'].fillna(data1['Current Ratio (%), Adj. - Credit Stats Direct [LTM]'].mean())



# Divide remaining columns into 9 categories and delete rows that doesn't have any value in any of these 9 categories
columnNumber = [np.arange(3,13),np.arange(13,23),np.arange(23,33),np.arange(33,43),np.arange(43,53),
               np.arange(55,65),np.arange(65,75)]

allNullOneCategory = data1.iloc[:,np.arange(3,13)].isnull().all(axis=1)
for category in columnNumber:
    allNullOneCategory = (allNullOneCategory | data1.iloc[:,category].isnull().all(axis=1))

data = data1[allNullOneCategory!=True]


data.shape


# So there are 2509 companies left with at least one value in each financial category.



companyName = data['Company Name']



# For each category, we use the mean of avaiable values of each company to fill in missing values
frames = []
for category in columnNumber:
    newData = data.iloc[:,category].T.fillna(np.mean(data.iloc[:,category].T)).T
    frames.append(newData)

frames.append(data.iloc[:,76:])

finalData = pd.concat(frames,axis=1)

finalData.index = companyName


finalData.head(5)

finalData.shape


# Randomly splitting data into training and test
np.random.seed(1)
sampleIndex = np.random.randint(0,2,size=finalData.shape[0])

sampleIndex = np.array(sampleIndex,dtype=np.bool)

traning = finalData[~sampleIndex]
test    = finalData[sampleIndex]

# Fit logistic regression
from sklearn import datasets, linear_model

regr = linear_model.LogisticRegression()

regr.fit(traning.iloc[:,:-1],traning.iloc[:,-1])

errorRate = 1 - regr.score(test.iloc[:,:-1],test.iloc[:,-1])
errorRate

# Fit decision tree classifier and set tuning parameter with values provided 
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=8,min_samples_leaf=20,min_impurity_split=0.01)
clf.fit(traning.iloc[:,:-1],traning.iloc[:,-1])


errorRate2 = 1 - clf.score(test.iloc[:,:-1],test.iloc[:,-1])
errorRate2

