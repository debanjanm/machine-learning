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
from sklearn.tree import DecisionTreeClassifier

import eli5
from eli5.sklearn import PermutationImportance

from sklearn import tree
import graphviz

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

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

tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)

###############################################################################

os.environ["PATH"] += os.pathsep + 'C:/Users/acer/Graphviz/bin/'

tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=feature_names)
graphviz.Source(tree_graph)

###############################################################################

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=feature_names, feature='Goal Scored')

# plot it
pdp.pdp_plot(pdp_goals, 'Goal Scored')
plt.show()

###############################################################################

feature_to_plot = 'Distance Covered (Kms)'
pdp_dist = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=feature_names, feature=feature_to_plot)

pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()

###############################################################################

# Build Random Forest model
rf_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

pdp_dist = pdp.pdp_isolate(model=rf_model, dataset=val_X, model_features=feature_names, feature=feature_to_plot)

pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()

###############################################################################

# Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate and pdp_interact_plot instead of pdp_isolate_plot
features_to_plot = ['Goal Scored', 'Distance Covered (Kms)']
inter1  =  pdp.pdp_interact(model=tree_model, dataset=val_X, model_features=feature_names, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()

####################################################################################################################################
####################################################################################################################################