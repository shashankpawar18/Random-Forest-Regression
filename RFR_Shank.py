#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 22:56:05 2018

@author: shashankpawar
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling not needed

# Fitting the Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
r1 = RandomForestRegressor(n_estimators=10, random_state = 0)
r2 = RandomForestRegressor(n_estimators=20, random_state = 0)
r3 = RandomForestRegressor(n_estimators=50, random_state = 0)
r4 = RandomForestRegressor(n_estimators=100, random_state = 0)
r5 = RandomForestRegressor(n_estimators=1000, random_state = 0)

r1.fit(X,y)
r2.fit(X,y)
r3.fit(X,y)
r4.fit(X,y)
r5.fit(X,y)

# Predicting a new result
y1 = r1.predict(6.5)
y2 = r2.predict(6.5)
y3 = r3.predict(6.5)
y4 = r4.predict(6.5)
y5 = r5.predict(6.5)

# Visualising the Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, r1.predict(X_grid), color = 'blue')
plt.plot(X_grid, r1.predict(X_grid), color = 'black')
plt.plot(X_grid, r1.predict(X_grid), color = 'aqua')
plt.plot(X_grid, r1.predict(X_grid), color = 'crimson')
plt.plot(X_grid, r1.predict(X_grid), color = 'yellow')
plt.title('Truth or Bluff (Random Forest Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()