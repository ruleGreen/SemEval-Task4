# 24 October 2019
# author: Carrey WANG

import os
import json
import numpy as np
import pandas as pd
from sklearn import datasets,svm
from sklearn.model_selection import KFold,cross_val_score



def read(task):
    df = pd.read_csv("datasets/TrainingData/subtask" + task + ".csv")
    return df


def kfold(df, k, random):
    kf = KFold(n_splits = k, shuffle = True, random_state = random)
    # build datasets
    current = 0
    for train_index, test_index in kf.split(df):
        if current < k:
            current += 1
        path = "datasets/k_fold_validation/" + task
        if not os.path.exists(path):
            os.makedirs(path)
        train = df.iloc[train_index]
        test = df.iloc[test_index]
        # print(train.head(5), test.head(5))
        train.to_csv(path + "/task_" + task + str(current) + "_train.csv", index = False)
        test.to_csv(path +"/task_" + task + str(current) + "_test.csv", index = False)


def readjson(model, task, k):
    if model == 1:
        path = "experiments/semeval4" + task + "_classifier.json"
        with open(path, 'r') as f:
            config = json.load(f)
        for current in range(1, k + 1):
            train_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_train.csv")
            validation_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_train.csv") 

            # change data path into config json
            config['train_data_path'] = train_data_path
            config['validation_data_path'] = validation_data_path

            with open(path, 'w') as f:
                json.dump(config, f)

            # run the model
            run(model, task, current)
    elif model == 2:
        path = "experiments/semeval4" + task + "_single_elmo.json"
        with open(path, 'r') as f:
            config = json.load(f)
        for current in range(1, k + 1):
            train_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_train.csv")
            validation_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_train.csv") 

            # change data path into config json
            config['train_data_path'] = train_data_path
            config['validation_data_path'] = validation_data_path

            with open(path, 'w') as f:
                json.dump(config, f)

            # run the model
            run(model, task, current)
    elif model == 3:
        path = "experiments/semeval4" + task + "_elmo.json"
        with open(path, 'r') as f:
            config = json.load(f)
        for current in range(1, k + 1):
            train_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_train.csv")
            validation_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_train.csv") 

            # change data path into config json
            config['train_data_path'] = train_data_path
            config['validation_data_path'] = validation_data_path

            with open(path, 'w') as f:
                json.dump(config, f)

            # run the model
            run(model, task, current)


def run(model, task, current):
    if model == 1:
        command = "allennlp train experiments/" + "semeval4" + task + "_classifier.json -s " + " ./tmp/semeval4" + task + "_classifier_output_dir/" + str(current) + "/" \
                        + " --include-package my_project"
        os.system(command)
    elif model == 2:
        command = "allennlp train experiments/" + "semeval4" + task + "_single_elmo.json -s " + " ./tmp/semeval4" + task + "_single_elmo_output_dir/" + str(current) + "/" \
                        + " --include-package my_project"
        os.system(command)
    elif model == 3:
        command = "allennlp train experiments/" + "semeval4" + task + "_elmo.json -s " + " ./tmp/semeval4" + task + "_elmo_output_dir/" + str(current) + "/" \
                        + " --include-package my_project"
        os.system(command)


def result(model, task, k):
    res = {}
    if model == 1:
        acc_train, acc_test = [], []
        for current in range(1, k+1):
            path = "tmp/semeval4" + task + "_classifier_output_dir/" + str(current) + "/" + "metrics.json"
            with open(path, r) as f:
                per = json.load(f)
            acc_train.append(per['training_accuracy'])
            acc_test.append(per['validation_accuracy'])
        avg_train = sum(acc_train) / len(acc_train)
        avg_test = sum(acc_test) / len(acc_test)

        res['train_accuracy'] = avg_train
        res['test_accuracy'] = avg_test
        with open("tmp/semeval4" + task + "_classifier_output_dir/result.json", 'w') as f:
            json.dump(res, f)
    elif model == 2:
        acc_train, acc_test = [], []
        for current in range(1, k+1):
            path = "tmp/semeval4" + task + "_single_elmo_output_dir/" + str(current) + "/" + "metrics.json"
            with open(path, r) as f:
                per = json.load(f)
            acc_train.append(per['training_accuracy'])
            acc_test.append(per['validation_accuracy'])
        avg_train = sum(acc_train) / len(acc_train)
        avg_test = sum(acc_test) / len(acc_test)

        res['train_accuracy'] = avg_train
        res['test_accuracy'] = avg_test
        with open("tmp/semeval4" + task + "_single_elmo_output_dir/result.json", 'w') as f:
            json.dump(res, f)
    elif model == 3:
        acc_train, acc_test = [], []
        for current in range(1, k+1):
            path = "tmp/semeval4" + task + "_elmo_output_dir/" + str(current) + "/" + "metrics.json"
            with open(path, r) as f:
                per = json.load(f)
            acc_train.append(per['training_accuracy'])
            acc_test.append(per['validation_accuracy'])
        avg_train = sum(acc_train) / len(acc_train)
        avg_test = sum(acc_test) / len(acc_test)

        res['train_accuracy'] = avg_train
        res['test_accuracy'] = avg_test
        with open("tmp/semeval4" + task + "_elmo_output_dir/result.json", 'w') as f:
            json.dump(res, f)


if __name__ == "__main__":
    #added some parameters
    task = input("please input the task(A,B,C): ")
    k = int(input("please input the k value of k fold cross validation: "))
    random = int(input("please input the random seed(integer): "))

    # read data
    df = read(task)
    kfold(df, k, random)

    model = int(input("please input the model you want to use: 1 -> single glove 2 -> single elmo 3-> ensemble: "))
    readjson(model, task, k)

    result(model, task, k)



# process command terminal
"""
mycmd = "ls -a"
os.system(mycmd)
"""