# preprocess data
import pandas as pd

train = pd.read_csv("../data/subtaskA.csv")
test = pd.read_csv("../data/subtaskA_test_data.csv")
dev = pd.read_csv("../data/subtaskA_dev.csv")

train = train.drop(['id'], axis = 1)
dev = dev.drop(['id'], axis = 1)

train.columns = ['text_a', 'text_b', 'labels']
dev.columns = ['text_a', 'text_b', 'labels']


from simpletransformers.classification import ClassificationModel
import sklearn


train_args={
    'reprocess_input_data': True,
    'overwrite_output_dir': True,
    'num_train_epochs': 100,
    'evaluate_during_training': True,
    'max_seq_length': 512,
    'evaluate_during_training_steps': 50,
}

# Create a ClassificationModel
model = ClassificationModel('albert', 'albert-base-v1', num_labels=2, use_cuda=True, args=train_args)

# train the model
model.train_model(train, eval_df=dev)

# evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(dev, acc=sklearn.metrics.accuracy_score)

# predict the model
evaluation = []
for i in range(len(test)):
    sample = []
    sample.append(test.iloc[i, 1])
    sample.append(test.iloc[i, 2])
    evaluation.append(sample)

predictions, raw_outputs = model.predict(evaluation)
np.savetxt("prediction.txt", prediction, delimiter=',', fmt="%d")