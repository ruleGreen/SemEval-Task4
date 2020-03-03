import json
import pandas as pd


if __name__ == "__main__":
    with open("c_res.jsonl", "r") as f:
        lines = f.readlines()

        targets = []
        for line in lines:
            line = json.loads(line)
            target = line["predicted_tokens"]
            target = " ".join(target[i] for i in range(len(target)))
            targets.append(target)
    
    test = pd.read_csv("subtaskC_test_data.csv")
    label_series = pd.Series(targets)

    result = pd.DataFrame(columns=["id", "label"])
    for i in range(len(test)):
        add_data = pd.Series({'id': test.iloc[i, 0], 'label': targets[i]})
        result = result.append(add_data, ignore_index=True)

    result.to_csv("subtaskC_test_answer.csv", header=False, index = False)
