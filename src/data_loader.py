import pandas as pd


def get_user_input():
    print('Enter iris names: ')
    names = []
    for i in range(2):
        names.append(input('Name: '))
    return names


class DataLoader:
    def __init__(self, file):
        self.file = file

    def load_data(self):
        df = pd.read_csv(self.file, index_col=0)
        return df
