import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import pickle
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
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')

#Importing data from csv and read Training Dataset
input_dir = 'file_path'
Dataset = pd.read_csv(input_dir + 'SurfaceTension-ILs.csv')
var_columns = [c for c in Dataset.columns if c not in('IonicLiuid','ST-exp')]
X = Dataset.loc[:, var_columns]
y = Dataset.loc[:, 'ST-exp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45, random_state=15)

#Hyperparameter tuning using RandomSearch CV
n_estimators = [100, 200, 300, 400, 500]
subsample = [0.1, 0.2, 0.4, 0.6, 0.8]
max_depth = [None, 2, 4, 6]
learning_rate = [0.01, 0.02, 0.03, 0.04]
min_samples_split = [2, 3, 4]
min_samples_leaf = [1, 2, 3]
min_weight_fraction_leaf = [0, 0.003, 0.03, 0.1]

random_grid = {'n_estimators': n_estimators, 'subsample': subsample,
               'max_depth': max_depth, 'min_samples_split': min_samples_split, 
               'min_samples_leaf': min_samples_leaf, 'learning_rate': learning_rate, 
               'min_weight_fraction_leaf': min_weight_fraction_leaf}

## Importing Random Forest Classifier from the sklearn.ensemble
GBTModel = GradientBoostingRegressor()
GBTModel_random = RandomizedSearchCV(estimator = GBTModel, param_distributions = random_grid, 
                              n_iter = 100, cv = 5, verbose=0, random_state=None, n_jobs = -1)

GBTModel_random.fit(X_train, y_train)

df_cv_results = pd.DataFrame(GBTModel_random.cv_results_)
df_cv_results = df_cv_results[['rank_test_score']]
df_cv_results.sort_values('rank_test_score', inplace=True)
#df_cv_results
print("Final Estimator Parameters")
print("---------------------------")

GBTModel_random.best_params_




