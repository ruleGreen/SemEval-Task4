# !/usr/bin
# this is a script for SemEval Task 4
# author WANG Hongru
# The Chinese University of Hong Kong

# DATA PREPROCESSING
if [ -e datasets/prepossessing.py ]; then
  python3 datasets/prepossessing.py
else
  echo "Sorry, you do not have prepossessing.py file at the datasets directory."
  exit 1
fi

# NOTE
echo "Please make sure that you already change the data path at the json config file."

# TRAINING PREDICTING TESTING THE MODEL
read -p "Please input the model that you want to train and predict: 1 -> ELMo" model
if [ "$model" == "1" ]; then
  read -p "Please input the task number that you want to train and predict:" ans
  if [ "$ans" = "1" ]; then
    allennlp train experiments/semeval4a_elmo.json -s ./tmp/semeval4a_elmo_output_dir/ --include-package my_project
    allennlp predict tmp/semeval4a_output_dir/model.tar.gz datasets/taskA_prediction.jsonl --include-package my_project --predictor a-classifier
    python -m allennlp.service.server_simple --archive-path tmp/semeval4a_elmo_output_dir --predictor a-classifier --include-package my_project --title "Which Sentence Does Not Make Sense?" --field-name sent0 --field-name sent1
  fi
  if [ "$ans" = "2" ]; then
    allennlp train experiments/semeval4b_elmo.json -s ./tmp/semeval4b_elmo_output_dir/ --include-package my_project
    # allennlp predict tmp/semeval4b_output_dir/model.tar.gz datasets/taskB_prediction.jsonl --include-package my_project --predictor a-classifier
    python -m allennlp.service.server_simple --archive-path tmp/semeval4b_elmo_output_dir --predictor a-classifier --include-package my_project --title "Which Sentence Does Not Make Sense?" --field-name sent --field-name reason1 \
    --field-name reason2 --field-name reason2
  fi
  if [ "$ans" = "3" ]; then
    allennlp train experiments/semeval4c_elmo.json -s ./tmp/semeval4c_elmo_output_dir/ --include-package my_project
  fi
fi
