# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 23:35:12 2023

@author: 91814
"""

import pandas as pd

df = pd.read_csv("C:/Users/91814/Desktop/Sem 9th/ML/PRACTICAL CIE 1  08-07-22/Salary_Data.csv")
print(df)
a = df[pd.isnull(df.Salary)]
print(a)

df['Salary'].fillna(df['Salary'].mean(),inplace=True)
print(df)
y = df['Salary'] 
# print(y)
X = df.drop(['Salary'], axis = 1) 
# print(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_predict = regressor.predict(X_test)
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.title('Year of experience Vs. Salary')
plt.xlabel('Year of experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test, regressor.predict(X_test), color = 'blue')