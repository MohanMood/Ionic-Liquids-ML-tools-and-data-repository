#!/usr/bin/env python
# coding: utf-8
# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from pandas.plotting import scatter_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, make_scorer, r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error as MSE
import warnings
warnings.filterwarnings('ignore')

#Importing data from csv and read Training Dataset
input_dir = 'C:/Users/MMohan/Videos/Scikit-Learn_ML/Models_MM/ST-RF_ILs/'
Dataset = pd.read_csv(input_dir + 'SurfaceTension-ILs.csv')
var_columns = [c for c in Dataset.columns if c not in('IonicLiuid','ST-exp')]
X = Dataset.loc[:, var_columns]
y = Dataset.loc[:, 'ST-exp']

# Splitting the data into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=35)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# Performing the Booster Tree (GradientBoostingRegressor)
import time
start = time.time()

GBTModel = GradientBoostingRegressor(loss='squared_error', learning_rate=0.03, n_estimators=100, 
                          subsample=0.6, criterion='friedman_mse', min_samples_split=3, 
                          min_samples_leaf=1, min_weight_fraction_leaf=0.003, max_depth=None, 
                          min_impurity_decrease=0.0, init=None, random_state=None, 
                          max_features=None, alpha=0.4, verbose=2, max_leaf_nodes=None, 
                          warm_start=False, validation_fraction=0.1, n_iter_no_change=None, 
                          tol=0.0001, ccp_alpha=0.0)

GBTModel.fit(X_train, y_train)

# Make prediction for Training
pred_train = GBTModel.predict(X_train)

# Mean absolute error (MAE)
mae_train = mean_absolute_error(y_train.values.ravel(), pred_train)
# Mean squared error (MSE)
mse_train = mean_squared_error(y_train.values.ravel(), pred_train)
rmse_train = (mse_train**0.5)
# mean absolute percentage error (MAPE)
mape_train = mean_absolute_percentage_error(y_train.values.ravel(), pred_train)
# R-squared scores
r2_train = r2_score(y_train.values.ravel(), pred_train)

# Print metrics
print('R2_Training:', round(r2_train, 3))
print('MAPE_Training:', "{:.2%}".format(mape_train))
print('MAE_Training:', round(mae_train, 2))
print('RMSE_Training:', round(rmse_train, 2))

# Make prediction for Testing
pred_test = GBTModel.predict(X_test)

# Mean absolute error (MAE)
mae_test = mean_absolute_error(y_test.values.ravel(), pred_test)
# Mean squared error (MSE)
mse_test = mean_squared_error(y_test.values.ravel(), pred_test)
rmse_test = (mse_test**0.5)
# mean absolute percentage error (MAPE)
mape_test = mean_absolute_percentage_error(y_test.values.ravel(), pred_test)
# R-squared scores
r2_test = r2_score(y_test.values.ravel(), pred_test)

# Print metrics
print("")
print('R2_Testing:', round(r2_test, 3))
print('MAPE_Testing:', "{:.2%}".format(mape_test))
print('MAE_Testing:', round(mae_test, 2))
print('RMSE_Testing:', round(rmse_test, 2))

end = time.time()
diff = end - start
print("")
print('Execution_time:', diff)




