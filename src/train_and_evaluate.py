import pandas as pd
import numpy as np
import hydra
from standardization import standardization
from hyperparameter_tuning import Parameter
from sklearn.ensemble import RandomForestClassifier

@hydra.main(config_name= '../config.yaml')

def train_model(config):
    train_x_scaled, test_x_scaled, train_y, test_y = standardization(config)
    if (config.algorithm == "random_forest"):
        n_estimators, criterion, max_depth, max_leaf_nodes = Parameter(config)
        pass
    elif (config.algorithm == "gradient_boosting"):
        pass

if __name__ == "__main__":
    train_model()
