import hydra
import pandas as pd
from load_data import load_data
from sklearn.model_selection import train_test_split

#Handle the missing data by imputing Mean Value
@hydra.main(config_name = '../config.yaml')
def handle_missing_data(config):
    df = load_data(config)
    for feature in df.columns.values:
        df.loc[df[feature].isnull(), feature] = df[feature].mean()
    return df

#Split the dataset into train, test set
@hydra.main(config_name = '../config.yaml')
def split_data(config):
    train_path = config.split_data.train_path
    test_path = config.split_data.test_path
    test_size = config.split_data.test_size
    random_state = config.split_data.random_state

    df = handle_missing_data(config)
    train, test = train_test_split(df, test_size = test_size, random_state = random_state)
    train.to_csv(train_path, index = False)
    test.to_csv(test_path, index = False)

if __name__ == "__main__":
    split_data()