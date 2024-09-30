# preprocess_data.py

import pandas as pd

def preprocess():
    df = pd.read_csv('../datasets/raw_dataset.csv')

    # Add your preprocessing steps here
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df.to_csv('../datasets/custom_dataset.csv', index=False)

if __name__ == '__main__':
    preprocess()
