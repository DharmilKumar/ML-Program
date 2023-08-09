# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 04:00:21 2023

@author: 91814
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read the dataset into a DataFrame
url = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
df = pd.read_csv(url)

# Select features (excluding species)
X = df.drop("species", axis=1)

# Initialize KMeans with the desired number of clusters
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

# Fit the KMeans model to the data
kmeans.fit(X)

# Add the cluster labels to the DataFrame
df["cluster"] = kmeans.labels_

# Plotting the clusters
plt.scatter(X["sepal_length"], X["sepal_width"], c=df["cluster"], cmap="viridis")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering")
plt.show()
