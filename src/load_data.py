import hydra
import pandas as pd

@hydra.main(config_name = '../config.yaml')
def load_data(config):
    df = pd.read_csv(config.dataset.raw, sep = '')
    return df

if __name__ == "__main__":
    load_data()