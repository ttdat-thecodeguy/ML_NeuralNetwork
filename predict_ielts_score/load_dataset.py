import pandas as pd

def load_dataset():
    df = pd.read_csv('./data/ielts_study_dataset.csv')
    return df
