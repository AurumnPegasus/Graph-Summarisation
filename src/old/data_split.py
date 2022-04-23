import pandas as pd


class Data:
    def __init__(self, path):
        self.data_path = path
        self.data = self.read_data()

    def read_data(self):
        # Function to read jsonl file and return pandas dataframe
        df = pd.read_json(path_or_buf=self.data_path, lines=True)
        return df

    def get_df(self):
        # Function to return the dataframe for the given path
        return self.data
