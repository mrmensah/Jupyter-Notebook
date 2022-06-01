# -*- coding: utf-8 -*-
"""
Created on Mon May 30 11:47:10 2022

@author: mrmensah
"""

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math

# Making NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# Importing the dataset
dataset = pd.read_csv('data.csv')

# Removing the year
dataset = dataset.drop('Year', axis='columns')

""" Checking the dataset for distribution of the data using the filter method (Pearson correlation)
Only 
"""
plt.figure(figsize=(12,10))
cor = dataset.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# Correlation with output variable
cor_target = abs(cor["Suppressed demand"])

# Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features

new_data = dataset.drop('losses %', axis='columns')

improved_data = new_data.drop('price ($)', axis='columns')

""" Building the Neural Netowork with Sklearn"""
# Importing the library
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Splitting ghe data into independent and dependent variables
X = improved_data.iloc[:, :-1].values
y = improved_data.iloc[:, 7].values

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Scaling the dataset
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Building the Regressor
regr = MLPRegressor(hidden_layer_sizes=(100,), random_state=1, max_iter=100000, activation='relu').fit(X_train, y_train)
y_pred = regr.predict(X_test)
y_pred

# Scoring the Model
score = regr.score(X_test, y_test)





