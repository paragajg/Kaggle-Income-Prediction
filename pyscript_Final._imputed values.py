##############################################

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.metrics import accuracy_score
import time
import sys	

# Load train csv

df = pd.read_csv('xgb/train_imputed.csv',header = 0)
df_test = pd.read_csv('xgb/test_imputed.csv',header = 0)

train_set = df.iloc[:,1:]
test_set = df_test.iloc[:,1:]

train_set.info()
test_set.info()

#train_nomissing = train_set.replace('?', np.nan).dropna()
#test_nomissing = test_set.replace(' ?', np.nan)

##1 Applying Ordinal Encodings to categorical features
for feature in train_set.columns: # Loop through all columns in the dataframe
    if train_set[feature].dtype == 'object': # Only apply for columns with categorical strings
        train_set[feature] = pd.Categorical(train_set[feature]).codes # Replace strings with an integer

for feature in test_set.columns: # Loop through all columns in the dataframe
    if test_set[feature].dtype == 'object': # Only apply for columns with categorical strings
        test_set[feature] = pd.Categorical(test_set[feature]).codes # Replace strings with an integer


train_set.info()
test_set.info()

## 2 Setting up initial model
y_train = train_set.pop('income')


#3 Fitting the xgb model and tuning the parameters

cv_params = {'learning_rate':[0.1,0.05,0.01],'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
ind_params = {'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic'}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params),cv_params,scoring = 'accuracy', cv = 5, n_jobs = -1)

start_time = time.time()
optimized_GBM.fit(train_set, y_train)
print("--- %s seconds ---" % (time.time() - start_time))
optimized_GBM.grid_scores_


## 4 Using Early Stoping using the DMatrix object of xgb

xgdmat = xgb.DMatrix(train_set, y_train)
our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'max_depth':3, 'min_child_weight':1} 
# Grid Search CV optimized settings

cv_xgb = xgb.cv(params = our_params, dtrain = xgdmat, num_boost_round = 3000, nfold = 5,
                metrics = ['error'], early_stopping_rounds = 100)

final_gb = xgb.train(our_params, xgdmat, num_boost_round = 440)

importances = final_gb.get_fscore()
importances

#5 Analyzing performance on Test Data

testdmat = xgb.DMatrix(test_set)
y_pred = final_gb.predict(testdmat) # Predict using our testdmat
y_pred = pd.Series(y_pred)
df_test["Predictions"] =  y_pred.values

# 6 Saving predicted values to a file
df_test.to_csv('xgb/Predictions_Imputation.csv',header=0)
