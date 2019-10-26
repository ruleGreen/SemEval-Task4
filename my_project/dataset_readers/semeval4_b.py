from typing import Dict, Iterator

import os
import logging
import pandas as pd

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)

@DatasetReader.register("semeval4_b_reader")
class SemEvalBDatasetReader(DatasetReader):
    def __init__(self, lazy: bool = False, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def text_to_instance(self, sent: str, reason1: str, reason2: str, reason3: str, label: str = None):
        #pylint: disable=arguments-differ
        tokenized_sent = self._tokenizer.tokenize(sent)
        tokenized_reason1 = self._tokenizer.tokenize(reason1)
        tokenized_reason2 = self._tokenizer.tokenize(reason2)
        tokenized_reason3 = self._tokenizer.tokenize(reason3)
        sent_field = TextField(tokenized_sent, self._token_indexers)
        reason1_field = TextField(tokenized_reason1, self._token_indexers)
        reason2_field = TextField(tokenized_reason2, self._token_indexers)
        reason3_field = TextField(tokenized_reason3, self._token_indexers)
        fields = {"sent": sent_field, "reason1": reason1_field, "reason2": reason2_field, "reason3": reason3_field}
        if label is not None:
            fields["label"] = LabelField(label)
        return Instance(fields)

    @overrides
    def _read(self, file_path) -> Iterator[Instance]:
        logger.info("Reading instance from lines in file at: %s", file_path)
        with open(cached_path(file_path)) as f:
            lines = f.readlines()[1:]  # skip the first line
            for line in lines:
                line = line.strip("\n")
                if not line:
                    continue
                splitted_line = line.split(',')
                if len(splitted_line) == 5:
                    sent, reason1, reason2, reason3 = splitted_line[1:]
                    # print(sent, reason1, reason2, reason3)
                    yield self.text_to_instance(sent, reason1, reason2, reason3)
                elif len(splitted_line) == 6:
                    sent, reason1, reason2, reason3, answer = splitted_line[1:]
                    # print(sent, reason1, reason2, reason3, answer)
                    yield  self.text_to_instance(sent, reason1, reason2, reason3, answer)

    def prepossing(self, file_path1, file_path2, file_path):
        if os.path.exists(file_path):
            return
        features = pd.read_csv(file_path1)
        answer = pd.read_csv(file_path2, header = None, names = ["id", "label"])
        merged = features.merge(answer, on = "id", how = "outer")
        merged.to_csv(file_path, index = False)


"""
if __name__ == "__main__":
    reader = SemEvalBDatasetReader()

    reader.prepossing("../../datasets/TrainingData/subtaskA_data_all.csv", "../../datasets/TrainingData/subtaskA_answers_all.csv", \
                      "../../datasets/TrainingData/subtaskA_training.csv")
    reader.read("../../datasets/TrainingData/subtaskA_training.csv")
"""