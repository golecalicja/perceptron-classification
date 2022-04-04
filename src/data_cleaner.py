class DataCleaner:
    def __init__(self, df_train, df_test, input_names):
        self.df_train = df_train
        self.df_test = df_test
        self.input_names = input_names

    def cleaned(self):
        self.remove_unselected_iris()
        self.map_to_number()
        return self.to_numpy()

    def remove_unselected_iris(self):
        irises = ['setosa', 'versicolor', 'virginica']
        for iris in irises:
            if iris not in self.input_names:
                self.df_train.drop(self.df_train.index[self.df_train['Species'] == iris], inplace=True)
                self.df_test.drop(self.df_test.index[self.df_test['Species'] == iris], inplace=True)

    def map_to_number(self):
        for i in range(len(self.input_names)):
            names_map = {self.input_names[i]: i}
            self.df_train['Species'] = self.df_train['Species'].replace(names_map)
            self.df_test['Species'] = self.df_test['Species'].replace(names_map)
            i += 1

    def to_numpy(self):
        train = self.df_train.to_numpy()
        test = self.df_test.to_numpy()
        return train, test
