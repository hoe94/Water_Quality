import pandas as pd
import numpy as np
import hydra
import mlflow
import os

from standardization import standardization
from hyperparameter_tuning import Parameter
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle

@hydra.main(config_name= '../config.yaml')

def train_model(config):
    train_x_scaled, test_x_scaled, train_y, test_y = standardization(config)
    #check the model exist in saved_models path

    if len(os.listdir(config.model_path)) != 0:
        if (config.algorithm == "random_forest"):
            rf = RandomForestClassifier( n_estimators = config.parameters.random_forest.n_estimators,
                                         criterion = config.parameters.random_forest.criterion,
                                         max_depth = config.parameters.random_forest.max_depth,
                                         max_leaf_nodes = config.parameters.random_forest.max_leaf_nodes
                                        )
            rf.fit(train_x_scaled, train_y)
            rf_y_pred = rf.predict(test_x_scaled)
            print(accuracy_score(test_y, rf_y_pred))
    

        elif (config.algorithm == "gradient_boosting"):
            gb = GradientBoostingClassifier( learning_rate = config.parameters.gradient_boosting.learning_rate,
                                             n_estimators = config.parameters.gradient_boosting.n_estimators,
                                             criterion = config.parameters.gradient_boosting.criterion,
                                             max_depth = config.parameters.gradient_boosting.max_depth,
                                             max_leaf_nodes = config.parameters.gradient_boosting.max_leaf_nodes)

            gb.fit(train_x_scaled, train_y)
            gb_y_pred = gb.predict(test_x_scaled)
            print(accuracy_score(test_y, gb_y_pred))

    #Hyperparameter Tuning for selected algorithm
    else: 
        if (config.algorithm == "random_forest"):
            n_estimators, criterion, max_depth, max_leaf_nodes = Parameter(config)
            rf = RandomForestClassifier( n_estimators = n_estimators,
                                         criterion = criterion,
                                         max_depth = max_depth,
                                         max_leaf_nodes = max_leaf_nodes
                                        )
            rf.fit(train_x_scaled, train_y)
            rf_y_pred = rf.predict(test_x_scaled)
            print(accuracy_score(test_y, rf_y_pred))
            with open( os.path.join(config.model_path, 'model.pkl'), 'wb')as file:
                pickle.dump(rf, file)

        elif (config.algorithm == "gradient_boosting"):
            learning_rate, n_estimators, criterion, max_depth, max_leaf_nodes = Parameter(config)
            gb = GradientBoostingClassifier(
                                            learning_rate = learning_rate,
                                            n_estimators = n_estimators,
                                            criterion = criterion,
                                            max_depth = max_depth,
                                            max_leaf_nodes = max_leaf_nodes
                                        )
            gb.fit(train_x_scaled, train_y)
            gb_y_pred = gb.predict(test_x_scaled)
            print(accuracy_score(test_y, gb_y_pred))
            with open( os.path.join(config.model_path, 'model.pkl'), 'wb')as file:
                pickle.dump(gb, file)

if __name__ == "__main__":
    train_model()
