import hydra
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

@hydra.main(config_name = '../config.yaml')
def standardization(config):
    
    train = pd.read_csv(config.split_data.train_path)
    test = pd.read_csv(config.split_data.test_path)
    min_max = MinMaxScaler()

    train_x = train.drop("Potability", axis = 1)
    train_y = train["Potability"]

    test_x = test.drop("Potability", axis = 1)
    test_y = test["Potability"]

    train_x_scaled = min_max.fit_transform(train_x)
    test_x_scaled = min_max.transform(test_x)

    return train_x_scaled, test_x_scaled, train_y, test_y

if __name__ == "__main__":
    standardization()