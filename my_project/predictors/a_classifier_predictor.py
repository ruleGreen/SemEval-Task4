from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

@Predictor.register('a-classifier')
class SemevalClassifierPredictor(Predictor):
    """"Predictor wrapper for the AcademicPaperClassifier"""
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        all_labels = [label_dict[i] for i in range(len(label_dict))]
        output_dict["all_labels"] = all_labels
        return output_dict
    
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sent0 = json_dict['sent0']
        sent1 = json_dict['sent1']
        return self._dataset_reader.text_to_instance(sent0=sent0, sent1=sent1)

    @overrides
    def load_line(self, line: str) -> JsonDict:
        splitted_line = line.split(',')
        res = {}
        if len(splitted_line) == 4:
            sent0, sent1, label = splitted_line[1:]
            # print(sent, reason1, reason2, reason3, answer)
            res['sent0'] = sent0
            res['sent1'] = sent1
            res['answer'] = label
        return res
