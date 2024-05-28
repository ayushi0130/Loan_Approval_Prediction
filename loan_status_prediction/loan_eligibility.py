import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('train.csv')
print(dataset.head())

print(dataset.shape)

print(dataset.info())

print(dataset.describe())

print(pd.crosstab(dataset['Credit_History'],dataset['Loan_Status'],margins=True))

dataset.boxplot(column='ApplicantIncome')
plt.show()

dataset['ApplicantIncome'].hist(bins=20)
plt.show()

dataset['CoapplicantIncome'].hist(bins=20)
plt.show()

dataset.boxplot(column='ApplicantIncome',by='Education')
plt.show()

dataset.boxplot(column='LoanAmount')
plt.show()

dataset['LoanAmount'].hist(bins=20)
plt.show()

dataset['LoanAmount_log']= np.log(dataset['LoanAmount'])
dataset['LoanAmount_log'].hist(bins=20)
plt.show()

print(dataset.isnull().sum())
print(list(dataset))

dataset['Gender'] = dataset['Gender'].fillna(dataset['Gender'].mode()[0])
dataset['Married'] = dataset['Married'].fillna(dataset['Married'].mode()[0])
dataset['Dependents'] = dataset['Dependents'].fillna(dataset['Dependents'].mode()[0])
dataset['Self_Employed'] = dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0])
dataset['Loan_Amount_Term'] = dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0])
dataset['Credit_History'] = dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0])

dataset.LoanAmount = dataset.LoanAmount.fillna(dataset.LoanAmount.mean())
dataset.LoanAmount_log = dataset.LoanAmount_log.fillna(dataset.LoanAmount_log.mean())

print(dataset.isnull().sum())

dataset['TotalIncome']=dataset['ApplicantIncome']+dataset['CoapplicantIncome']
dataset['TotalIncome'] = np.log(dataset['TotalIncome'])
dataset['TotalIncome'].hist(bins=20)
plt.show()

print(dataset.head())

X = dataset.iloc[:,np.r_[1:5,9:11,13:15]].values
y = dataset.iloc[:,12].values
print(X)
print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train)

from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()

for i in range(0,5):
    X_train[:,i] = labelencoder_x.fit_transform(X_train[:,i])

X_train[:,7] = labelencoder_x.fit_transform(X_train[:,7])
print(X_train)

labelencoder_y = LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
print(y_train)

for i in range(0,5):
    X_test[:,i] = labelencoder_x.fit_transform(X_test[:,i])

X_test[:,7] = labelencoder_x.fit_transform(X_test[:,7])
print(X_test)

y_test = labelencoder_y.fit_transform(y_test)
print(y_test)

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

from sklearn.tree import DecisionTreeClassifier
DTClassifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
DTClassifier.fit(X_train,y_train)
y_pred = DTClassifier.predict(X_test)
print(y_pred)

from sklearn import metrics
print('The accuracy of decision tree is: ', metrics.accuracy_score(y_pred,y_test))

from sklearn.naive_bayes import GaussianNB

NBClassifier = GaussianNB()
NBClassifier.fit(X_train,y_train)

y_pred= NBClassifier.predict(X_test)
print(y_pred)
print('The accuracy of naive tree is: ', metrics.accuracy_score(y_pred,y_test))

testdata = pd.read_csv('test.csv')
print(testdata.head())

print(testdata.info())

print(testdata.isnull().sum())

testdata.boxplot(column='LoanAmount')
plt.show()

testdata['Gender']=testdata['Gender'].fillna(testdata['Gender'].mode()[0])
testdata['Dependents']=testdata['Dependents'].fillna(testdata['Dependents'].mode()[0])
testdata['Self_Employed']=testdata['Self_Employed'].fillna(testdata['Self_Employed'].mode()[0])
testdata['Credit_History']=testdata['Credit_History'].fillna(testdata['Credit_History'].mode()[0])
testdata['LoanAmount']=testdata['LoanAmount'].fillna(testdata['LoanAmount'].mean())
testdata['Loan_Amount_Term']=testdata['Loan_Amount_Term'].fillna(testdata['Loan_Amount_Term'].mode()[0])

print(testdata.isnull().sum())

testdata.boxplot(column='ApplicantIncome')
plt.show()

testdata['LoanAmount_log'] = np.log(testdata['LoanAmount'])
testdata.boxplot(column='LoanAmount')
plt.show()

testdata['TotalIncome'] = testdata['ApplicantIncome']+testdata['CoapplicantIncome']
testdata['TotalIncome_log']=np.log(testdata['TotalIncome'])

print(testdata.head())

test = testdata.iloc[:,np.r_[1:5,9:11,13:15]].values

for i in range(0,5):
    test[:,i]= labelencoder_x.fit_transform(test[:,i])

test[:,7]= labelencoder_x.fit_transform(test[:,7])

print(test)

test = ss.fit_transform(test)

pred = NBClassifier.predict(test)

print(pred)