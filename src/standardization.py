import hydra
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import Counter
from imblearn.under_sampling import NearMiss

# Using standardization formula (x-u / standard deviation) to standardize all features value
@hydra.main(config_name = '../config.yaml')
def standardization(config):
    
    train = pd.read_csv(config.split_data.train_path)
    test = pd.read_csv(config.split_data.test_path)
    sc = StandardScaler()

    train_x = train.drop("Potability", axis = 1)
    train_y = train["Potability"]

    test_x = test.drop("Potability", axis = 1)
    test_y = test["Potability"]

    train_x_scaled = sc.fit_transform(train_x)
    test_x_scaled = sc.transform(test_x)

    return train_x_scaled, test_x_scaled, train_y, test_y

@hydra.main(config_name = '../config.yaml')
def undersampling(config):

    train_x_scaled, test_x_scaled, train_y, test_y = standardization(config)
    ns = NearMiss(sampling_strategy = config.undersampling.ratio)
    x_train_ns, y_train_ns = ns.fit_resample(train_x_scaled, train_y)
    #print('The number of classes before under sampling: {}'.format(Counter(y_train)))
    #print('The number of classes after under sampling: {}'.format(Counter(y_train_ns)))
    return x_train_ns, test_x_scaled, y_train_ns, test_y


if __name__ == "__main__":
    undersampling()