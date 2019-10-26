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

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("semeval4_a_reader")
class SemEvalADatasetReader(DatasetReader):
    def __init__(self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy = False)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
    
    @overrides
    def text_to_instance(self, sent0: str, sent1: str, label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        tokenized_sent0 = self._tokenizer.tokenize(sent0)
        tokenized_sent1 = self._tokenizer.tokenize(sent1)
        sent0_field = TextField(tokenized_sent0, self._token_indexers)
        sent1_field = TextField(tokenized_sent1, self._token_indexers)
        fields = {'sent0': sent0_field, 'sent1': sent1_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)
    
    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        logger.info("Reading instances from lines in file at: %s", file_path)
        with open(cached_path(file_path)) as f:
            lines = f.readlines()[1:] # skip the first line
            for line in lines:
                line = line.strip("\n")
                if not line:
                    continue
                splitted_line = line.split(',')
                if len(splitted_line) == 3:
                    sent0, sent1 = splitted_line[1:]
                    yield self.text_to_instance(sent0, sent1)
                elif len(splitted_line) == 4:
                    sent0, sent1, label = splitted_line[1:]
                    yield self.text_to_instance(sent0, sent1, label)
                
    def prepossing(self, file_path1, file_path2, file_path):
        if os.path.exists(file_path):
            return
        features = pd.read_csv(file_path1)
        labels = pd.read_csv(file_path2, header = None, names = ['id', 'label'])
        merged = features.merge(labels, on="id", how="outer")
        merged.to_csv(file_path, index=False)