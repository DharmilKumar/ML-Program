# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 10:16:12 2023

@author: 91814
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('Desktop/simpleLinear.csv')

# Split the data into features (X) and the target variable (y)
X = data[['study_hours']]
y = data['exam_score']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the Linear Regression model
regression = LinearRegression()

# Fit the model to the training data
regression.fit(X_train, y_train)

# Predict the exam scores for the test data
y_pred = regression.predict(X_test)

# Print the coefficients and intercept of the model
print("Coefficient:", regression.coef_[0])
print("Intercept:", regression.intercept_)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
