import json
import pandas as pd


if __name__ == "__main__":
    with open("a_res.jsonl", "r") as f:
        lines = f.readlines()

        labels = []
        for line in lines:
            line = json.loads(line)
            label = line["label"]
            labels.append(label)

    test = pd.read_csv("subtaskA_test_data.csv")
    label_series = pd.Series(labels)

    result = pd.DataFrame(columns=["id", "label"])
    for i in range(len(test)):
        add_data = pd.Series({'id': test.iloc[i, 0], 'label': labels[i]})
        result = result.append(add_data, ignore_index=True)

    result.to_csv("subtaskA_test_answer.csv", index = False)
