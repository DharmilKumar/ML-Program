# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 23:57:02 2023

@author: 91814
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/91814/Desktop/Sem 9th/ML/PRACTICAL CIE 1  08-07-22/heart.csv')
print(df.shape)

print(df.info())

print(df.describe)

plt.hist(df['target'])
df.plot.hist()


fig = plt.figure(figsize = (10, 5))
plt.bar(df['target'].unique(), df['target'].value_counts(), color = ['red', 'blue'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')


#Print the min, max and avg age for the given dataset.

print("count", df['age'].count())
print("mean", df['age'].mean())
print("minimum", df['age'].min())
print("maximum", df['age'].max())
print(min(df.age))


#Find the correlation matrix of features and shows the color bar for the matrix.
import numpy as np
plt.matshow(df.corr()) 
plt.yticks(np.arange(df.shape[1]), df.columns)
plt.xticks(np.arange(df.shape[1]), df.columns) 
plt.colorbar()


Young = df[(df.age>=29)&(df.age<40)]
Middle = df[(df.age >=40) & (df.age<55)]
Elder = df[(df.age >55)]

import seaborn as sns 
sns.set_context('notebook',font_scale = 1)
sns.countplot(df['sex'])

#Sex is a categorical variable. Break each categorical column into dummy columns with 1s and 0s.
dfs = pd.get_dummies(df, columns = ['sex'] )
dfs.head(6)


#Perform feature scaling on the given dataset.
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
df[columns_to_scale] = standardScaler.fit_transform(df[columns_to_scale])
print(df[columns_to_scale])
print(df['age'])

#Split the dataset into 67% training data and 33% testing data
from sklearn.model_selection import train_test_split
y = df['target'] 
X = df.drop(['target'], axis = 1) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
print(X)
print(y)


#Perform K Neighbors Classifier for K= 8. Print score.

from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors = 8)
knn_classifier.fit(X_train, y_train)
print(knn_classifier.score(X_test, y_test))


#Display confusion matrix for the above experiment.


# prediction

y_pred = knn_classifier.predict(X_test)

# confusion matrix



from sklearn.metrics import confusion_matrix
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
