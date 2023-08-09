# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 09:37:49 2023

@author: 91814
"""

import pandas as pd
import numpy as np

data  = pd.read_csv("Desktop/knn.csv")

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset


# Split the data into features and target variable
X = data[['study_hours', 'exam_score']]
y = data['pass']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

# Create an instance of the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the model to the training data
knn.fit(X_train, y_train)

# Predict the pass status for the test data
y_pred = knn.predict(X_test)
print(y_pred)
predicted_data = pd.DataFrame({'study_hours': X_test['study_hours'], 'exam_score': X_test['exam_score'], 'predicted_pass': y_pred})
print(predicted_data)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
