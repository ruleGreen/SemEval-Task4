# This is for automatically merge datasets and split it into training and validation datasets
# author WANG Hongru

import os
import sys
import random
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("./")

class Prepossessing:
    def __init__(self, task, seed, test_size):
        self.task = task
        self.seed = seed
        self.test_size = test_size
        if self.task == 1:
            self.preA("./datasets/TrainingData/subtaskA_data_all.csv", "./datasets/TrainingData/subtaskA_answers_all.csv")
        elif self.task == 2:
            self.preB("./datasets/TrainingData/subtaskB_data_all.csv", "./datasets/TrainingData/subtaskB_answers_all.csv")
        elif self.task == 3:
            self.preC("./datasets/TrainingData/subtaskC_data_all.csv", "./datasets/TrainingData/subtaskC_answers_all.csv")

    def preA(self, input_file_path1, input_file_path2):
        features = pd.read_csv(input_file_path1)
        answer = pd.read_csv(input_file_path2, header=None, names=["id", "label"])
        merged = features.merge(answer, on="id", how="outer")
        merged.to_csv("./datasets/TrainingData/subtaskA.csv", index=False)
        self.shuffle("./datasets/TrainingData/subtaskA.csv")
        self.splitA("./datasets/TrainingData/subtaskA.csv", self.test_size)

    def preB(self, input_file_path1, input_file_path2):
        features = pd.read_csv(input_file_path1)
        answer = pd.read_csv(input_file_path2, header=None, names=["id", "label"])
        merged = features.merge(answer, on="id", how="outer")
        merged.to_csv("./datasets/TrainingData/subtaskB.csv", index=False)
        self.shuffle("./datasets/TrainingData/subtaskB.csv")
        self.splitB("./datasets/TrainingData/subtaskB.csv", self.test_size)

    def preC(self, input_file_path1, input_file_path2):
        features = pd.read_csv(input_file_path1)
        answer = pd.read_csv(input_file_path2, header=None, names=["id", "ref1", "ref2", "ref3"])
        merged = features.merge(answer, on="id", how="outer")
        merged.to_csv("./datasets/TrainingData/subtaskC.csv", index=False)
        self.shuffle("./datasets/TrainingData/subtaskC.csv")
        # the split of task c is different from task a and b
        self.splitC("./datasets/TrainingData/subtaskC.csv")

    def shuffle(self, path):
        # shuffle the datasets with random seed
        df = pd.read_csv(path)
        df = df.sample(n=len(df), random_state = self.seed)
        print(df.head(5))
        df.to_csv(path, index = False)

    def splitA(self, file_path, test_size):
        df = pd.read_csv(file_path)
        train, test = train_test_split(df, test_size=test_size)
        train.to_csv("datasets/TrainingData/subtaskA_training.csv", index = False)
        test.to_csv("datasets/TrainingData/subtaskA_test.csv", index = False)

    def splitB(self, file_path, test_size):
        df = pd.read_csv(file_path)
        train, test = train_test_split(df, test_size=test_size)
        train.to_csv("datasets/TrainingData/subtaskB_training.csv", index=False)
        test.to_csv("datasets/TrainingData/subtaskB_test.csv", index=False)

    def splitC(self, file_path):
        df = pd.read_csv(file_path)
        # to do dataframe for task c
        ref1 = df.loc[:, ['id', 'FalseSent', 'ref1']]
        ref2 = df.loc[:, ['id', 'FalseSent', 'ref2']]
        ref3 = df.loc[:, ['id', 'FalseSent', 'ref3']]
        ref1.columns = ['id', 'falsesent', 'ref']
        ref2.columns = ['id', 'falsesent', 'ref']
        ref3.columns = ['id', 'falsesent', 'ref']
        res = pd.concat([ref1, ref2, ref3], axis = 0)
        res.to_csv("datasets/TrainingData/subtaskC.csv", index=False)
        


if __name__ == "__main__":
    task = int(input("Please input datasets of which task you want to prepossessing:"))
    shuffle = int(input("Please input the random seed you want to shuffle data:"))
    seed = float(input("Please input the test_size(percent) you want to split train and test data:"))
    
    reader = Prepossessing(task, shuffle, seed)
