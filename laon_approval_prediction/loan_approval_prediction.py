import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as svm

df = pd.read_csv('loan.csv')
print(df.head())

print(df.info())
print(df.isnull().sum())

df['LoanAmount_log']= np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)
plt.show()

print(df.isnull().sum())

df['TotalIncome']= df['ApplicantIncome']+df['CoapplicantIncome']
df['TotalIncome_log']=np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)
plt.show()

df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
df['Married']=df['Married'].fillna(df['Married'].mode()[0])
df['Dependents']=df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df.LoanAmount=df.LoanAmount.fillna(df.LoanAmount.mean())
df.LoanAmount_log=df.LoanAmount_log.fillna(df.LoanAmount_log.mean())
print(df.isnull().sum())

x = df.iloc[:,np.r_[1:5,9:11,13:15]].values
y= df.iloc[:,12].values

print(x)
print(y)

print('percentage of missing gender is %2f%%'%((df['Gender'].isnull().sum()/df.shape[0])*100))

print('Number of people who takes loan as group by gender: ')
print(df['Gender'].value_counts())
sns.countplot(x ='Gender',data=df,hue='Gender',palette='Set1')
plt.show()

print('Number of people who takes loan as group by marital status: ')
print(df['Married'].value_counts())
sns.countplot(x='Married',data=df,hue='Married',palette='Set1',legend=False)
plt.show()

print('Number of people who takes loan as group by Dependents: ')
print(df['Dependents'].value_counts())
sns.countplot(x='Dependents',data=df,hue='Dependents',palette='Set1',legend=False)
plt.show()

print('Number of people who takes loan as group by LoanAmount: ')
print(df['LoanAmount'].value_counts())
sns.countplot(x='LoanAmount',data=df,hue='LoanAmount',palette='Set1',legend=False)
plt.show()

print('Number of people who takes loan as group by Credit_History: ')
print(df['Credit_History'].value_counts())
sns.countplot(x='Credit_History',data=df,hue='Credit_History',palette='Set1',legend=False)
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

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

from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier()
rf_clf.fit(X_train,y_train)

from sklearn import metrics
y_pred = rf_clf.predict(X_test)

print('acc of random forest clf is', metrics.accuracy_score(y_pred,y_test))
print(y_pred)

from sklearn.naive_bayes import GaussianNB

nb_clf = GaussianNB()
nb_clf.fit(X_train,y_train)

y_pred = nb_clf.predict(X_test)

print('accuracy of naive bayes is %', metrics.accuracy_score(y_pred,y_test))

print(y_pred)

from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train,y_train)

y_pred = dt_clf.predict(X_test)

print('accuracy of DecisionTree is %',metrics.accuracy_score(y_pred,y_test))

print(y_pred)

from sklearn.neighbors import KNeighborsClassifier
kn_clf = KNeighborsClassifier()
kn_clf.fit(X_train,y_train)

y_pred = kn_clf.predict(X_test)

print('accuracy of kneighbors is %',metrics.accuracy_score(y_pred,y_test))
print(y_pred)