# 24 October 2019
# author: Carrey WANG

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from sklearn import datasets, svm
from sklearn.model_selection import KFold,cross_val_score



def read(task, data):
    if data == '1':
        df = pd.read_csv("datasets/TrialData/task" + task + ".csv")
    elif data == '2':
        df = pd.read_csv("datasets/TrainingData/subtask" + task + ".csv")
    return df


def kfold(df, k, rand, task):
    kf = KFold(n_splits = k, shuffle = True, random_state = rand)
    # build datasets
    current = 0
    # if there is datasets, then just return
    # and you need to shuffle first before save to csv
    path = "datasets/k_fold_validation/" + task
    if os.path.exists(path):
        return
    for train_index, test_index in kf.split(df):
        if current < k:
            current += 1
        if not os.path.exists(path):
            os.makedirs(path)
        train = df.iloc[train_index]
        test = df.iloc[test_index]
        # get validation datasets, and others as test Datasets
        # validation, test = test.iloc[:len(test) // 2], test.iloc[len(test) // 2:]
        # shuffle data
        train = train.sample(n = len(train), random_state = rand)
        test = test.sample(n = len(test), random_state = rand)
        # validation = validation.sample(n = len(validation), random_state = rand)
        # print(train.head(5), test.head(5))
        if task == 'A' or task == 'B':
            train.to_csv(path + "/task_" + task + str(current) + "_train.csv", index = False)
            # validation.to_csv(path + "/task_" + task + str(current) + "_validation.csv", index = False)
            test.to_csv(path +"/task_" + task + str(current) + "_test.csv", index = False)

def k_fold_for_c(data, k, rand):
    # read data
    if data == '1':
        df = pd.read_csv("datasets/TrialData/task" + task + "_merge.csv")
    elif data == '2':
        df = pd.read_csv("datasets/TrainingData/subtask" + task + "_merge.csv")
    
    path = "datasets/k_fold_validation/" + task
    if not os.path.exists(path):
            os.makedirs(path)

    for e in range(1, k + 1):
        # generate random numer between 0 - 9999
        number = [i for i in range(10000)]
        slice = random.sample(number, 10000 // k )
        for i in range(len(slice)):
            number.remove(slice[i])
        train = number
        validation = random.sample(slice, len(slice) // 2)
        for j in range(len(validation)):
            slice.remove(validation[j])
        test = slice

        # shuffle data
        train_data, test_data, validation_data = df[df['id'].isin(train)], df[df['id'].isin(test)], df[df['id'].isin(validation)]
        train_data = train_data.sample(n = len(train_data), random_state = rand)
        test_data = test_data.sample(n = len(test_data), random_state = rand)
        validation_data = validation_data.sample(n = len(validation_data), random_state = rand)

        # drop id column
        train_data = train_data.drop(['id'], axis = 1)
        test_data = test_data.drop(['id'], axis = 1)
        validation_data = validation_data.drop(['id'], axis = 1)


        train_data.to_csv(path + "/task_" + task + str(e) + "_train.csv", header=False, index = False)
        validation_data.to_csv(path + "/task_" + task + str(e) + "_validation.csv", header=False, index = False)
        test_data.to_csv(path +"/task_" + task + str(e) + "_test.csv", header=False, index = False)
    
    

def readjson(model, task, k):
    if model == 1:
        path = "experiments/semeval4" + task + "_glove.json"
        with open(path, 'r') as f:
            config = json.load(f)
        for current in range(1, k + 1):
            train_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_train.csv")
            test_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_test.csv")
            if task == 'A':
                validation_path = ("datasets/DevData/subtaskA_dev.csv")
            elif task == 'B':
                validation_path = ("datasets/DevData/subtaskB_dev.csv")
            elif task == 'C':
                validation_path = ("datasets/DevData/subtaskC_dev.csv")

            # change data path into config json
            config['train_data_path'] = train_data_path
            config['test_data_path'] = test_data_path
            config['validation_data_path'] = validation_path

            with open(path, 'w') as f:
                json.dump(config, f, indent=4)

            # run the model
            run(model, task, current)
    elif model == 2:
        path = "experiments/semeval4" + task + "_single_elmo.json"
        with open(path, 'r') as f:
            config = json.load(f)
        for current in range(1, k + 1):
            train_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_train.csv")
            test_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_test.csv") 
            if task == 'A':
                validation_path = ("datasets/DevData/subtaskA_dev.csv")
            elif task == 'B':
                validation_path = ("datasets/DevData/subtaskB_dev.csv")
            elif task == 'C':
                validation_path = ("datasets/DevData/subtaskC_dev.csv")

            # change data path into config json
            config['train_data_path'] = train_data_path
            config['test_data_path'] = test_data_path
            config['validation_data_path'] = validation_path

            with open(path, 'w') as f:
                json.dump(config, f, indent=4)

            # run the model
            run(model, task, current)
    elif model == 3:
        path = "experiments/semeval4" + task + "_ensemble.json"
        with open(path, 'r') as f:
            config = json.load(f)
        for current in range(1, k + 1):
            train_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_train.csv")
            test_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_test.csv") 
            validation_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_validation.csv") 

            # change data path into config json
            config['train_data_path'] = train_data_path
            config['test_data_path'] =  test_data_path
            config['validation_data_path'] = validation_path

            with open(path, 'w') as f:
                json.dump(config, f, indent=4)

            # run the model
            run(model, task, current)
    elif model == 6:
        path = "experiments/semeval4" + task + "_single_bert.json"
        with open(path, 'r') as f:
            config = json.load(f)
        for current in range(1, k + 1):
            train_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_train.csv")
            test_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_test.csv") 
            if task == 'A':
                validation_path = ("datasets/DevData/subtaskA_dev.csv")
            elif task == 'B':
                validation_path = ("datasets/DevData/subtaskB_dev.csv")
            elif task == 'C':
                validation_path = ("datasets/DevData/subtaskC_dev.csv")

            # change data path into config json
            config['train_data_path'] = train_data_path
            config['test_data_path'] =  test_data_path
            config['validation_data_path'] = validation_path

            with open(path, 'w') as f:
                json.dump(config, f, indent=4)

            # run the model
            run(model, task, current)
    # this is for task C
    elif model == 4:
        path = "experiments/semeval4" + task + "_generation.json"
        with open(path, 'r') as f:
            config = json.load(f)
        for current in range(1, k + 1):
            train_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_train.csv")
            test_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_test.csv") 
            if task == 'A':
                validation_path = ("datasets/DevData/subtaskA_dev.csv")
            elif task == 'B':
                validation_path = ("datasets/DevData/subtaskB_dev.csv")
            elif task == 'C':
                validation_path = ("datasets/DevData/subtaskC_dev.csv")

            # change data path into config json
            config['train_data_path'] = train_data_path
            config['test_data_path'] =  test_data_path
            config['validation_data_path'] = validation_path

            with open(path, 'w') as f:
                json.dump(config, f, indent=4)

            # run the model
            run(model, task, current)
    elif model == 5:
        path = "experiments/semeval4" + task + "_glove.json"
        with open(path, 'r') as f:
            config = json.load(f)
        for current in range(1, k + 1):
            train_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_train.csv")
            test_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_test.csv") 
            if task == 'A':
                validation_path = ("datasets/DevData/subtaskA_dev.csv")
            elif task == 'B':
                validation_path = ("datasets/DevData/subtaskB_dev.csv")
            elif task == 'C':
                validation_path = ("datasets/DevData/subtaskC_dev.csv")

            # change data path into config json
            config['train_data_path'] = train_data_path
            config['test_data_path'] =  test_data_path
            config['validation_data_path'] = validation_path

            with open(path, 'w') as f:
                json.dump(config, f, indent=4)

            # run the model
            run(model, task, current)
    elif model == 9:   # this is for testing
        path = "experiments/semeval4" + task + "_model3.json"
        with open(path, 'r') as f:
            config = json.load(f)
        for current in range(1, k + 1):
            train_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_train.csv")
            test_data_path = ("datasets/k_fold_validation/" + task + "/task_" + task + str(current) + "_test.csv") 
            if task == 'A':
                validation_path = ("datasets/DevData/subtaskA_dev.csv")
            elif task == 'B':
                validation_path = ("datasets/DevData/subtaskB_dev.csv")
            elif task == 'C':
                validation_path = ("datasets/DevData/subtaskC_dev.csv") 

            # change data path into config json
            config['train_data_path'] = train_data_path
            config['test_data_path'] =  test_data_path
            config['validation_data_path'] = validation_path

            with open(path, 'w') as f:
                json.dump(config, f, indent=4)

            # run the model
            run(model, task, current)


def run(model, task, current):
    if model == 1:
        command = "allennlp train experiments/" + "semeval4" + task + "_glove.json -s " + " ./tmp/semeval4" + task + "_glove_output_dir/" + str(current) + "/" \
                        + " --include-package my_project"
        os.system(command)
    elif model == 2:
        command = "allennlp train experiments/" + "semeval4" + task + "_single_elmo.json -s " + " ./tmp/semeval4" + task + "_single_elmo_output_dir/" + str(current) + "/" \
                        + " --include-package my_project"
        os.system(command)
    elif model == 3:
        command = "allennlp train experiments/" + "semeval4" + task + "_ensemble.json -s " + " ./tmp/semeval4" + task + "_ensemble_output_dir/" + str(current) + "/" \
                        + " --include-package my_project"
        os.system(command)
    elif model == 6:
        command = "allennlp train experiments/" + "semeval4" + task + "_single_bert.json -s " + " ./tmp/semeval4" + task + "_single_bert_output_dir/" + str(current) + "/" \
                        + " --include-package my_project"
        os.system(command)
    elif model == 4:
        command = "allennlp train experiments/" + "semeval4" + task + "_generation.json -s " + " ./tmp/semeval4" + task + "_generation_output_dir/" + str(current) + "/" \
                        + " --include-package my_project"
        os.system(command)
    elif model == 5:
        command = "allennlp train experiments/" + "semeval4" + task + "_glove.json -s " + " ./tmp/semeval4" + task + "_glove_output_dir/" + str(current) + "/" \
                        + " --include-package my_project"
        os.system(command)
    
    elif model == 9:  # this is for testing
        command = "allennlp train experiments/" + "semeval4" + task + "_model3.json -s " + " ./tmp/semeval4" + task + "_test_output/" + str(current) + "/" \
                        + " --include-package my_project"
        os.system(command)


def result(model, task, k):
    res = {}
    if model == 1:
        acc_train, acc_test = [], []
        for current in range(1, k+1): 
            path = "tmp/semeval4" + task + "_glove_output_dir/" + str(current) + "/" + "metrics.json"
            with open(path, 'r') as f:
                per = json.load(f)
            acc_train.append(per['training_accuracy'])
            acc_test.append(per['test_accuracy'])
        avg_train = sum(acc_train) / len(acc_train)
        avg_test = sum(acc_test) / len(acc_test)
        max_train = max(acc_train)
        max_test = max(acc_test)
        min_train = min(acc_train)
        min_test = min(acc_test)

        res['avg_train_accuracy'] = avg_train
        res['avg_test_accuracy'] = avg_test
        res['max_train_accuracy'] = max_train
        res['max_test_accuracy'] = max_test
        res['min_train_accuracy'] = min_train
        res['min_test_accuracy'] = min_test
        with open("tmp/semeval4" + task + "_glove_output_dir/result.json", 'w') as f:
            json.dump(res, f)
    elif model == 2:
        acc_train, acc_test = [], []
        for current in range(1, k+1):
            path = "tmp/semeval4" + task + "_single_elmo_output_dir/" + str(current) + "/" + "metrics.json"
            with open(path, 'r') as f:
                per = json.load(f)
            acc_train.append(per['training_accuracy'])
            acc_test.append(per['test_accuracy'])
        avg_train = sum(acc_train) / len(acc_train)
        avg_test = sum(acc_test) / len(acc_test)
        max_train = max(acc_train)
        max_test = max(acc_test)
        min_train = min(acc_train)
        min_test = min(acc_test)

        res['avg_train_accuracy'] = avg_train
        res['avg_test_accuracy'] = avg_test
        res['max_train_accuracy'] = max_train
        res['max_test_accuracy'] = max_test
        res['min_train_accuracy'] = min_train
        res['min_test_accuracy'] = min_test
        with open("tmp/semeval4" + task + "_single_elmo_output_dir/result.json", 'w') as f:
            json.dump(res, f)
    elif model == 3:
        acc_train, acc_test = [], []
        for current in range(1, k+1):
            path = "tmp/semeval4" + task + "_ensemble_output_dir/" + str(current) + "/" + "metrics.json"
            with open(path, 'r') as f:
                per = json.load(f)
            acc_train.append(per['training_accuracy'])
            acc_test.append(per['test_accuracy'])
        avg_train = sum(acc_train) / len(acc_train)
        avg_test = sum(acc_test) / len(acc_test)
        max_train = max(acc_train)
        max_test = max(acc_test)
        min_train = min(acc_train)
        min_test = min(acc_test)

        res['avg_train_accuracy'] = avg_train
        res['avg_test_accuracy'] = avg_test
        res['max_train_accuracy'] = max_train
        res['max_test_accuracy'] = max_test
        res['min_train_accuracy'] = min_train
        res['min_test_accuracy'] = min_test
        with open("tmp/semeval4" + task + "_ensemble_output_dir/result.json", 'w') as f:
            json.dump(res, f)
    elif model == 4:
        loss_train, loss_test, bleu_test = [], [], []
        for current in range(1, k+1):
            path = "tmp/semeval4" + task + "_generation_output_dir/" + str(current) + "/" + "metrics.json"
            with open(path, 'r') as f:
                per = json.load(f)
            loss_train.append(per['training_loss'])
            loss_test.append(per['test_loss'])
            bleu_test.append(per['test_BLEU'])
        avg_train = sum(loss_train) / len(loss_train)
        avg_test = sum(loss_test) / len(loss_test)
        avg_bleu = sum(bleu_test) / len(bleu_test)
        max_train = max(loss_train)
        max_test = max(loss_test)
        max_bleu = max(bleu_test)
        min_train = min(loss_train)
        min_test = min(loss_test)
        min_bleu = min(bleu_test)

        res['avg_train_loss'] = avg_train
        res['avg_test_BLEU'] = avg_test
        res['avg_test_loss'] = avg_bleu
        res['max_train_loss'] = max_train
        res['max_test_loss'] = max_test
        res['max_test_BLEU'] = max_bleu
        res['min_train_loss'] = min_train
        res['min_test_loss'] = min_test
        res['min_test_BLEU'] = min_bleu
        with open("tmp/semeval4" + task + "_generation_output_dir/result.json", 'w') as f:
            json.dump(res, f)
    elif model == 5:
        loss_train, loss_test, bleu_test = [], [], []
        for current in range(1, k+1):
            path = "tmp/semeval4" + task + "_glove_output_dir/" + str(current) + "/" + "metrics.json"
            with open(path, 'r') as f:
                per = json.load(f)
            loss_train.append(per['training_loss'])
            loss_test.append(per['test_loss'])
            bleu_test.append(per['test_BLEU'])
        avg_train = sum(loss_train) / len(loss_train)
        avg_test = sum(loss_test) / len(loss_test)
        avg_bleu = sum(bleu_test) / len(bleu_test)
        max_train = max(loss_train)
        max_test = max(loss_test)
        max_bleu = max(bleu_test)
        min_train = min(loss_train)
        min_test = min(loss_test)
        min_bleu = min(bleu_test)

        res['avg_train_loss'] = avg_train
        res['avg_test_BLEU'] = avg_test
        res['avg_test_loss'] = avg_bleu
        res['max_train_loss'] = max_train
        res['max_test_loss'] = max_test
        res['max_test_BLEU'] = max_bleu
        res['min_train_loss'] = min_train
        res['min_test_loss'] = min_test
        res['min_test_BLEU'] = min_bleu
        with open("tmp/semeval4" + task + "_glove_output_dir/result.json", 'w') as f:
            json.dump(res, f)
    elif model == 6:
        acc_train, acc_test = [], []
        for current in range(1, k+1):
            path = "tmp/semeval4" + task + "_single_bert_output_dir/" + str(current) + "/" + "metrics.json"
            with open(path, 'r') as f:
                per = json.load(f)
            acc_train.append(per['training_accuracy'])
            acc_test.append(per['test_accuracy'])
        avg_train = sum(acc_train) / len(acc_train)
        avg_test = sum(acc_test) / len(acc_test)
        max_train = max(acc_train)
        max_test = max(acc_test)
        min_train = min(acc_train)
        min_test = min(acc_test)

        res['avg_train_accuracy'] = avg_train
        res['avg_test_accuracy'] = avg_test
        res['max_train_accuracy'] = max_train
        res['max_test_accuracy'] = max_test
        res['min_train_accuracy'] = min_train
        res['min_test_accuracy'] = min_test
        with open("tmp/semeval4" + task + "_single_bert_output_dir/result.json", 'w') as f:
            json.dump(res, f)

    elif model == 9:  # this is for testing
        acc_train, acc_test = [], []
        for current in range(1, k+1):
            path = "tmp/semeval4" + task + "_test_output/" + str(current) + "/" + "metrics.json"
            with open(path, 'r') as f:
                per = json.load(f)
            acc_train.append(per['training_accuracy'])
            acc_test.append(per['test_accuracy'])
        avg_train = sum(acc_train) / len(acc_train)
        avg_test = sum(acc_test) / len(acc_test)
        max_train = max(acc_train)
        max_test = max(acc_test)
        min_train = min(acc_train)
        min_test = min(acc_test)

        res['avg_train_accuracy'] = avg_train
        res['avg_test_accuracy'] = avg_test
        res['max_train_accuracy'] = max_train
        res['max_test_accuracy'] = max_test
        res['min_train_accuracy'] = min_train
        res['min_test_accuracy'] = min_test
        with open("tmp/semeval4" + task + "_test_output/result.json", 'w') as f:
            json.dump(res, f)


if __name__ == "__main__":
    #added some parameters
    print("number of argv", len(sys.argv))
    print("content of argv", sys.argv)
    print("+++++++++++++++++++++++++++++++")

    data, task, k, rand, model = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])


    if task == 'A' or task == 'B':
        # read data
        df = read(task, data)
        kfold(df, k, rand, task)

        readjson(model, task, k)
        result(model, task, k)
    elif task == 'C':
        k_fold_for_c(data, k, rand)

        readjson(model, task, k)
        result(model, task, k)





# process command terminal
"""
mycmd = "ls -a"
os.system(mycmd)
"""