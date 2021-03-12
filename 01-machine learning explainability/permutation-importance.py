# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 20:42:34 2021

@author: acer
"""

####################################################################################################################################
####################################################################################################################################
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import eli5
from eli5.sklearn import PermutationImportance

####################################################################################################################################
####################################################################################################################################

print(os.getcwd())  # Prints the current working directory
path="C:/Users/acer/OneDrive/Documents/GitHub/machine-learning/01-machine learning explainability"
os.chdir(path)

###############################################################################
DATA_DIR = "data" # indicate magical constansts (maybe rather put it on the top of the script)
fifa_filename = "FIFA 2018 Statistics.csv"# fix gruesome var names
fifa_df = pd.read_csv(os.path.join(DATA_DIR, fifa_filename))
data = fifa_df

###############################################################################

y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

###############################################################################
my_model = RandomForestClassifier(n_estimators=100,random_state=0).fit(train_X, train_y)

###############################################################################

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
explanation = eli5.formatters.as_dataframe.explain_weights_df(perm, feature_names = val_X.columns.tolist())

#explanation_pred = eli5.explain_prediction_df(estimator=my_model, doc=X_test[0])

####################################################################################################################################
####################################################################################################################################
