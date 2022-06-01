# -*- coding: utf-8 -*-
"""
Created on Mon May 30 11:55:15 2022

@author: mrmensah
"""
# Importing the libraries
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Making NumPy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# Importing the dataset
dataset = pd.read_csv('data.csv')

# Removing the year
dataset = dataset.drop('Year', axis='columns')

""" 
Checking the dataset for distribution of the
 data using the filter method (Pearson correlation) Only 
"""
plt.figure(figsize=(12, 10))
cor = dataset.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# Correlation with output variable
cor_target = abs(cor["Suppressed demand"])

# Inspecting the data
sns.pairplot(dataset[['Purchases (GWh)', 'GDP per Capita', 'Customers',
                      'Informal sector (%)', 'price (cedi)', 'Installed Capacity (MW)',
                      'reserve margin (MW)', 'Suppressed demand']], diag_kind='kde')

# Selecting highly correlated features
relevant_features = cor_target[cor_target > 0.5]
relevant_features

new_data = dataset.drop('losses %', axis='columns')

improved_data = new_data.drop('price ($)', axis='columns')

# Importing tensorflow libraries


# Splitting the data into training and test sets
train_dataset = improved_data.sample(frac=0.8, random_state=0)
test_dataset = improved_data.drop(train_dataset.index)


# Selecting the training and testing labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Suppressed demand')
test_labels = test_features.pop('Suppressed demand')


# Normalizing the data
normalizer = tf.keras.layers.Normalization(axis=-1)
# Fitting the state of the preprocessing layer to the data
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())
print('Normalized:', normalizer(train_features).numpy())

""" The Model """
# The model


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


dnn_model = build_and_compile_model(normalizer)
summary = dnn_model.summary()

test_predictions = dnn_model.predict(test_features).flatten()
