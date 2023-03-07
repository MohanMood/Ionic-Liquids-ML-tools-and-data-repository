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
input_dir = 'file_path'
Dataset = pd.read_csv(input_dir + 'SpeedOfSound_ILs.csv')
var_columns = [c for c in Dataset.columns if c not in('IonicLiuid','SS-exp')]
X = Dataset.loc[:, var_columns]
y = Dataset.loc[:, 'SS-exp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=205)

# creating a regression model with Training and Testing sets
MLR_model = LinearRegression()
# fitting the model
MLR_model.fit(X_train, y_train)

# Make prediction for Testing
pred_test = MLR_model.predict(X_test)

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
