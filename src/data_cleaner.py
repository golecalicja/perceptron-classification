class DataCleaner:
    def __init__(self, df, input_names):
        self.df = df
        self.input_names = input_names

    def cleaned(self):
        self.remove_unselected_iris()
        self.map_to_number()
        return self.to_numpy()

    def remove_unselected_iris(self):
        irises = ['setosa', 'versicolor', 'virginica']
        for iris in irises:
            if iris not in self.input_names:
                self.df.drop(self.df.index[self.df['Species'] == iris], inplace=True)

    def map_to_number(self):
        for i in range(len(self.input_names)):
            names_map = {self.input_names[i]: i}
            self.df['Species'] = self.df['Species'].replace(names_map)
            i += 1

    def to_numpy(self):
        df = self.df.to_numpy()
        return df
