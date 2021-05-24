import pandas as pd
import numpy as np
import hydra
import time
import datetime

from standardization import standardization
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

@hydra.main(config_name= '../config.yaml')
def Parameter(config):

    if (config.algorithm == "random_forest"):
        start_time = time.time()
        rf = RandomForestClassifier()
        parameters = {'n_estimators': config.hyperparameter_tuning.random_forest.n_estimators,
                      'criterion': config.hyperparameter_tuning.random_forest.criterion,
                      'max_depth': config.hyperparameter_tuning.random_forest.max_depth,
                      #'min_samples_split': config.hyperparameter_tuning.min_samples_split,
                      'max_leaf_nodes': config.hyperparameter_tuning.random_forest.max_leaf_nodes
                      #'random_state': config.hyperparameter_tuning.random_state,
                      #'verbose': config.hyperparameter_tuning.verbose
                    }
        grid_search = GridSearchCV(rf, 
                                   param_grid = parameters, 
                                   scoring = "accuracy", 
                                   cv = config.hyperparameter_tuning.random_forest.cv,
                                   verbose = config.hyperparameter_tuning.random_forest.verbose)
        train_x_scaled, test_x_scaled, train_y, test_y = standardization(config)

        grid_search.fit(train_x_scaled, train_y)
        #print(f"Best Parameters Setting: {grid_search.best_params_}")
        #print(f"Best Score: {grid_search.best_score_}")
        #print(f"running time ： {str(datetime.timedelta(seconds = (time.time() - start_time)))}")

        return grid_search.best_params_["n_estimators"], grid_search.best_params_["criterion"], grid_search.best_params_["max_depth"], grid_search.best_params_["max_leaf_nodes"]
    
    elif (config.algorithm == "gradient_boosting"):
        pass

if __name__ == "__main__":
    Parameter()