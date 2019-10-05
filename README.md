<h1>First of all</h1>

<h2>Please make sure that you already install allennlp and related python packages. After that, u can just run these scripts.</h2>

allennlp train experiments/semeval4a_classifier.json -s ./tmp/semeval4a_output_dir --include-package my_project

allennlp predict tmp/semeval4a_output_dir/model.tar.gz datasets/taskA_prediction.jsonl --include-package my_project --predictor a-classifier

python -m allennlp.service.server_simple --archive-path tmp/semeval4a_output_dir/model.tar.gz --predictor a-classifier --include-package my_project --title "Which Sentence Does Not Make Sense?" --field-name sent0 --field-name sent1
