# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 00:45:29 2020

@author: Emmanuel Hammond
"""

## Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Importing dataset and defining variables
dataset = pd.read_csv(
    "C:/Users/User/Desktop/HashAnalytic Internship/Data Science 1/6. Regression Polynomial Regression/3.1 Gaming_data.csv.csv")
X = dataset.iloc[:, 0:1].values
Y = dataset.iloc[:, 1].values

## Fitting Linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y)
Y_pred = lin_reg.predict(X)

## Visualising Linear model
plt.plot(X,Y,X,Y_pred)

### Fitting Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 10)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, Y)
Y_pred2 = lin_reg2.predict(X_poly)

### Visualising polynomial regression
plt.scatter(X,Y)
plt.plot(X,Y_pred2, 'r')

### Single prediction with both models
lin_reg.predict([[3.5]])
lin_reg2.predict(poly_reg.fit_transform([[11]]))





