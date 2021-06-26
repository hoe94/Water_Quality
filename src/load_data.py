import hydra
import pandas as pd

@hydra.main(config_name = '../config.yaml')
#hydra.initialize(config_path = '../config.yaml')
def load_data(config):
    # load the dataset from data/raw
    df = pd.read_csv(config.dataset.raw, sep = ',')
    return df

if __name__ == "__main__":
    load_data()