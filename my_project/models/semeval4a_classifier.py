from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("a_classifier")
class SenseClassifier(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 sent0_encoder: Seq2VecEncoder,
                 sent1_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 loss = torch.nn.CrossEntropyLoss(),
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SenseClassifier, self).__init__(vocab, regularizer)
        
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.sent0_encoder = sent0_encoder
        self.sent1_encoder = sent1_encoder
        self.classifier_feedforward = classifier_feedforward
        self.loss = loss
        
        """
        if text_field_embedder.get_output_dim() != sent0_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the title_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            sent0_encoder.get_input_dim()))
        if text_field_embedder.get_output_dim() != sent1_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the abstract_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            sent1_encoder.get_input_dim()))
        """
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }

        initializer(self)
    
    @overrides
    def forward(self, 
                sent0: Dict[str, torch.LongTensor],
                sent1: Dict[str, torch.LongTensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        embedded_sent0 = self.text_field_embedder(sent0)
        sent0_mask = util.get_text_field_mask(sent0)
        encoded_sent0 = self.sent0_encoder(embedded_sent0, sent0_mask)
        
        embedded_sent1 = self.text_field_embedder(sent1)
        sent1_mask = util.get_text_field_mask(sent1)
        encoded_sent1 = self.sent1_encoder(embedded_sent1, sent1_mask)

        logits = self.classifier_feedforward(torch.cat([encoded_sent0, encoded_sent1], dim=-1))
        output_dict = {'logits': logits}
        
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict
    
    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities
        
        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict
    
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}