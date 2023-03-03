#!/usr/bin/env python
# coding: utf-8
# In[18]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn import metrics
from io import StringIO
import os
import warnings
warnings.filterwarnings('ignore')

#Importing data from csv and read Training Dataset
input_dir = 'C:/Users/MMohan/Videos/Scikit-Learn_ML/Models_MM/ST-RF_ILs/'
Dataset = pd.read_csv(input_dir + 'SurfaceTension-ILs.csv')
var_columns = [c for c in Dataset.columns if c not in('IonicLiuid','ST-exp')]
X = Dataset.loc[:, var_columns]
y = Dataset.loc[:, 'ST-exp']

# Splitting the data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=15)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# creating a regression model with Training and Testing sets
MLR_model = LinearRegression()

# fitting the model
MLR_model.fit(X_train, y_train)

# Make prediction for Training
pred_train = MLR_model.predict(X_train)

# Mean absolute error (MAE)
mae_train = mean_absolute_error(y_train.values.ravel(), pred_train)
# Mean squared error (MSE)
mse_train = mean_squared_error(y_train.values.ravel(), pred_train)
rmse_train = (mse_train**0.5)
# R-squared scores
r2_train = r2_score(y_train.values.ravel(), pred_train)

# Print metrics
print("")
print('MAE_Training:', round(mae_train, 3))
print('MSE_Training:', round(mse_train, 3))
print('RMSE_Training:', round(rmse_train, 3))
print('R2_Training:', round(r2_train, 3))

# Make prediction for Testing
pred_test = MLR_model.predict(X_test)

# Mean absolute error (MAE)
mae_test = mean_absolute_error(y_test.values.ravel(), pred_test)
# Mean squared error (MSE)
mse_test = mean_squared_error(y_test.values.ravel(), pred_test)
rmse_test = (mse_test**0.5)
# R-squared scores
r2_test = r2_score(y_test.values.ravel(), pred_test)

# Print metrics
print("")
print('MAE_Testing:', round(mae_test, 3))
print('MSE_Testing:', round(mse_test, 4))
print('RMSE_Testing:', round(rmse_test, 4))
print('R2_Testing:', round(r2_test, 3))

# creating a regression model
MLR_model = LinearRegression()

# fitting the model
MLR_model.fit(X_train, y_train)

# with statsmodels Training
import statsmodels.api as sm
X_train = sm.add_constant(X_train) # adding a constant
 
model_train = sm.OLS(y_train, X_train).fit()
predictions_train = model_train.predict(X_train) 

print_MLRmodel_train = model_train.summary()
print("")
print("************************** Training **************************")
print("")
print(print_MLRmodel_train)




