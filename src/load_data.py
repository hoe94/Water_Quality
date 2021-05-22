import hydra
import pandas as pd

@hydra.main(config_name='config.yaml')
def main(config):
    df = pd.read_csv(config.dataset.raw)
    print(df.head())

if __name__ == "__main__":
    main()