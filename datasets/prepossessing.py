# This is for automatically merge datasets and split it into training and validation datasets
# author WANG Hongru

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("./")

class Prepossessing:
    def __init__(self, task, seed):
        self.task = task
        self.seed = seed
        if self.task == 1:
            self.preA("./datasets/TrainingData/subtaskA_data_all.csv", "./datasets/TrainingData/subtaskA_answers_all.csv")
        elif self.task == 2:
            self.preB("./datasets/TrainingData/subtaskB_data_all.csv", "./datasets/TrainingData/subtaskB_answers_all.csv")
        elif self.task == 3:
            self.preC("./datasets/TrainingData/subtaskC_data_all.csv", "./datasets/TrainingData/subtaskC_answers_all.csv")

    def preA(self, input_file_path1, input_file_path2):
        if os.path.exists("./datasets/TrainingData/subtaskA.csv"):
            self.splitA("./datasets/TrainingData/subtaskA.csv", self.seed)
            return
        features = pd.read_csv(input_file_path1)
        answer = pd.read_csv(input_file_path2, header=None, names=["id", "label"])
        merged = features.merge(answer, on="id", how="outer")
        merged.to_csv("./datasets/TrainingData/subtaskA.csv", index=False)
        self.splitA("./datasets/TrainingData/subtaskA.csv", self.seed)

    def preB(self, input_file_path1, input_file_path2):
        if os.path.exists("./datasets/TrainingData/subtaskB.csv"):
            self.splitB("./datasets/TrainingData/subtaskB.csv", self.seed)
            return
        features = pd.read_csv(input_file_path1)
        answer = pd.read_csv(input_file_path2, header=None, names=["id", "label"])
        merged = features.merge(answer, on="id", how="outer")
        merged.to_csv("./datasets/TrainingData/subtaskB.csv", index=False)
        self.splitB("./datasets/TrainingData/subtaskB.csv", self.seed)

    def preC(self, input_file_path1, input_file_path2):
        if os.path.exists("./datasets/TrainingData/subtaskC.csv"):
            self.splitA("./datasets/TrainingData/subtaskC.csv", self.seed)
            return
        features = pd.read_csv(input_file_path1)
        answer = pd.read_csv(input_file_path2, header=None, names=["id", "label"])
        merged = features.merge(answer, on="id", how="outer")
        merged.to_csv("./datasets/TrainingData/subtaskC.csv", index=False)
        self.splitC("./datasets/TrainingData/subtaskC.csv", self.seed)

    def splitA(self, file_path, seed):
        df = pd.read_csv(file_path)
        train, test = train_test_split(df, test_size=seed)
        train.to_csv("datasets/TrainingData/subtaskA_training.csv", index = False)
        test.to_csv("datasets/TrainingData/subtaskA_test.csv", index = False)

    def splitB(self, file_path, seed):
        df = pd.read_csv(file_path)
        train, test = train_test_split(df, test_size=seed)
        train.to_csv("datasets/TrainingData/subtaskB_training.csv", index=False)
        test.to_csv("datasets/TrainingData/subtaskB_test.csv", index=False)

    def splitC(self, file_path, seed):
        df = pd.read_csv(file_path)
        train, test = train_test_split(df, test_size=seed)
        train.to_csv("datasets/TrainingData/subtaskC_training.csv", index=False)
        test.to_csv("datasets/TrainingData/subtaskC_test.csv", index=False)


if __name__ == "__main__":
    task = int(input("Please input datasets of which task you want to prepossessing:"))
    seed = float(input("Please input the random seed:"))

    reader = Prepossessing(task, seed)
