import pandas as pd
import numpy as np


def process_data(file_path):
    data = pd.read_csv(file_path)


    key_words_to_select = ['mean', 'std', 'skew']
    column_to_select = data.columns[data.columns.str.contains('|'.join(key_words_to_select), regex=True)]
    data = data[column_to_select]

    return data









def main():
    file_path = 'C:/Users/user/Desktop/NTU_hw/final/html-2024-fall-final-project-stage-1/train_data.csv'

    data = process_data(file_path)

    data.to_csv('C:/Users/user/Desktop/NTU_myHW/final/data_with_mean_std_skew.csv', index=False)
    print("Data process completed")