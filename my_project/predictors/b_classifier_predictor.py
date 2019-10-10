from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('b-classifier')
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
        sent = json_dict['sent']
        reason1 = json_dict['reason1']
        reason2 = json_dict['reason2']
        reason3 = json_dict['reason3']
        return self._dataset_reader.text_to_instance(sent=sent, reason1=reason1, reason2=reason2, reason3=reason3)