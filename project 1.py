# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 01:42:08 2019

@author: yash1
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

# Import supplementary visualizations code visuals.py
import visuals as vs


# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
    
# Success
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))
minimum_price = np.min(prices)
maximum_price = np.max(prices)
mean_price = np.mean(prices)
median_price = np.median(prices)
std_price = np.std(prices)

print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${}".format(minimum_price)) 
print("Maximum price: ${}".format(maximum_price))
print("Mean price: ${}".format(mean_price))
print("Median price ${}".format(median_price))
print("Standard deviation of prices: ${}".format(std_price))

def performance_metric(y_true,y_predict):
    from sklearn.metrics import r2_score
    score = r2_score(y_true, y_predict) 
    return score


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features,prices,test_size =0.2, random_state=42)
print("Training and testing split was successful.")



def fit_model(X,y):
    # Create cross-validation sets from the training data
    from sklearn.model_selection import GridSearchCV
    cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    from sklearn.tree import DecisionTreeRegressor
    regressor = DecisionTreeRegressor(random_state=0)
    params = {'max_depth': list(range(1,11))}
    from sklearn.metrics import make_scorer
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(regressor,params,scoring_fnc,cv=cv_sets)
    grid = grid.fit(X, y)
    return grid.best_estimator_


reg = fit_model(X_train, y_train)
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

client_data = [[5, 17, 15], # Client 1
               [4, 32, 22], # Client 2
               [8, 3, 12]]  # Client 3

# Show predictions
for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home: ${:,.2f}".format(i+1, price))
    













