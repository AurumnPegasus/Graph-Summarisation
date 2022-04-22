from constants import TRAIN_PATH, TEST_PATH, VALIDATION_PATH

from icecream import ic
import pandas as pd


class Data:
    def __init__(
        self,
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        validation_path=VALIDATION_PATH,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.validation_path = validation_path

    def readData(self, path) -> object:
        # Function to read jsonl files and return pandas dataframe

        df = pd.read_json(path_or_buf=path, lines=True)
        return df

    def getDFs(self) -> tuple:
        # Function to return (train, test, val) dataframes

        return self.train, self.test, self.val

    def handle(self) -> None:
        # Function to handle basic logistics :p

        self.train = self.readData(self.train_path)
        self.test = self.readData(self.test_path)
        self.val = self.readData(self.validation_path)


if __name__ == "__main__":
    d = Data()
    d.handle()
    train, test, val = d.getDFs()
    ic(test.head())
