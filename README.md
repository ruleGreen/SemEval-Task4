<h1>First of all</h1>

<h2>Please make sure that you already install allennlp and related python packages. After that, u can just run these scripts.</h2>

allennlp train experiments/semeval4a_classifier.json -s ./tmp/semeval4a_output_dir --include-package my_project

allennlp predict tmp/semeval4a_output_dir/model.tar.gz datasets/taskA_prediction.jsonl --include-package my_project --predictor a-classifier

python -m allennlp.service.server_simple --archive-path tmp/semeval4a_output_dir/model.tar.gz --predictor a-classifier --include-package my_project --title "Which Sentence Does Not Make Sense?" --field-name sent0 --field-name sent1

Pls ref our paper if u use our code.

@article{wang2020cuhk,
  title={Cuhk at semeval-2020 task 4: Commonsense explanation, reasoning and prediction with multi-task learning},
  author={Wang, Hongru and Tang, Xiangru and Lai, Sunny and Leung, Kwong Sak and Zhu, Jia and Fung, Gabriel Pui Cheong and Wong, Kam-Fai},
  journal={arXiv preprint arXiv:2006.09161},
  year={2020}
}
