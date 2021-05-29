import pandas as pd
import numpy as np
import hydra
import mlflow
from mlflow.tracking import MlflowClient
import os
import json
import pickle
import time

from standardization import standardization
from hyperparameter_tuning import Parameter
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def eval_metric(y_test, y_pred):
    log_time = time.strftime("%Y%m%d-%H%M%S")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    c_matrix = confusion_matrix(y_test, y_pred)
    true_positve = c_matrix[0][0]
    true_negative = c_matrix[1][1]
    false_positive = c_matrix[0][1]
    false_negative = c_matrix[1][0]

    #json_object = {
    #               "log_time": log_time,
    #               "accuracy_score": float(np.round(accuracy, 4)),
    #               "precision": float(np.round(precision, 4)),
    #               "recall": float(np.round(recall, 4)),
    #               "true_positve" : int(true_positve),
    #               "true_negative" : int(true_negative),
    #               "false_positive" : int(false_positive),
    #               "false_negative" : int(false_negative),
    #              }
    #json_file = json.dumps(json_object, indent = 4)
    #return json_file
    return accuracy, precision, recall, true_positve, true_negative, false_positive, false_negative
    

@hydra.main(config_name= '../config.yaml')

#26.5.2021
#//1. new function for eval metric (accuracy, precision, recall, f1 ratio)
#//1. create new folder, store the json result

#29.5.2021
#1. implement mlflow
#2 write json result
#3. retrain the base model with export the result & params



def train_model(config):
    train_x_scaled, test_x_scaled, train_y, test_y = standardization(config)
    log_time = time.strftime("%Y%m%d-%H%M%S")
    
    mlflow.set_tracking_uri(config.mlflow_config.remote_server_uri)
    mlflow.set_experiment(config.mlflow_config.experiment_name)


    with mlflow.start_run(run_name = config.mlflow_config.run_name) as run:
        #check the model exist in saved_models path
        if (os.path.exists(config.base_model.random_forest) & os.path.exists(config.base_model.gradient_boosting)):
            if (config.algorithm == "random_forest"):
                rf = RandomForestClassifier( n_estimators = config.parameters.random_forest.n_estimators,
                                             criterion = config.parameters.random_forest.criterion,
                                             max_depth = config.parameters.random_forest.max_depth,
                                             max_leaf_nodes = config.parameters.random_forest.max_leaf_nodes
                                            )
                rf.fit(train_x_scaled, train_y)
                rf_y_pred = rf.predict(test_x_scaled)
                #json_file = eval_metric(test_y, rf_y_pred)
                accuracy, precision, recall, true_positve, true_negative, false_positive, false_negative = eval_metric(test_y, rf_y_pred)
                mlflow.log_metric('accuracy_score', float(np.round(accuracy, 4))),
                mlflow.log_metric('precision', np.round(precision, 4)),
                mlflow.log_metric('recall', np.round(recall, 4)),
                mlflow.log_metric('true_positive', int(true_positve)),
                mlflow.log_metric('true_negative', int(true_negative)),
                mlflow.log_metric('false_positive', int(false_positive)),
                mlflow.log_metric('false_negative', int(false_negative)), 
                
                mlflow.log_param('n_estimators', rf.n_estimators),
                mlflow.log_param('criterion', rf.criterion),
                mlflow.log_param('max_depth', rf.max_depth),
                mlflow.log_param('max_leaf_nodes', rf.max_leaf_nodes),

                mlflow.sklearn.log_model(rf,"RandomForest")
                #with open( os.path.join(config.results_path, f"random_forest_{log_time}"), 'w')as file:
                #    file.write(json_file)
            
            elif(config.algorithm == "gradient_boosting"):
                gb = GradientBoostingClassifier( learning_rate = config.parameters.gradient_boosting.learning_rate,
                                                 n_estimators = config.parameters.gradient_boosting.n_estimators,
                                                 criterion = config.parameters.gradient_boosting.criterion,
                                                 max_depth = config.parameters.gradient_boosting.max_depth,
                                                 max_leaf_nodes = config.parameters.gradient_boosting.max_leaf_nodes)

                gb.fit(train_x_scaled, train_y)
                gb_y_pred = gb.predict(test_x_scaled)
                #print(accuracy_score(test_y, gb_y_pred))
                json_file = eval_metric(test_y, gb_y_pred)
                with open( os.path.join(config.results_path, f"gradient_boosting_{log_time}"), 'w')as file:
                    file.write(json_file)

            #Hyperparameter Tuning for selected algorithm if the base model doesn't exist
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
                with open( os.path.join(config.model_path, 'random_forest_base_model.pkl'), 'wb')as file:
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
                with open( os.path.join(config.model_path, 'gradient_boosting_base_model.pkl'), 'wb')as file:
                    pickle.dump(gb, file)

if __name__ == "__main__":
    train_model()
