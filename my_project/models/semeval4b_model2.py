from typing import Dict, Optional

import numpy
import torch
import torch.nn.functional as F

from overrides import overrides
from allennlp.nn import util
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.modules.seq2seq_encoders.bidirectional_language_model_transformer import PositionalEncoding
from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy

@Model.register("b_classifier_2")
class SenseBClassifier(Model):
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 sent_encoder: Seq2VecEncoder,
                 option_a_encoder: Seq2VecEncoder,
                 option_b_encoder: Seq2VecEncoder,
                 option_c_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 loss = torch.nn.CrossEntropyLoss(),
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(SenseBClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        # self.positional_encoding = positional_encoding
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.sent_encoder = sent_encoder
        self.option_a_encoder = option_a_encoder
        self.option_b_encoder = option_b_encoder
        self.option_c_encoder = option_c_encoder
        self.classifier_feedforward = classifier_feedforward
        self.loss = loss

        self.metrics = {
            "accuracy": CategoricalAccuracy()
        }

        initializer(self)

    @overrides
    def forward(self, sent: Dict[str, torch.LongTensor],
                reason1: Dict[str, torch.LongTensor],
                reason2: Dict[str, torch.LongTensor],
                reason3: Dict[str, torch.LongTensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        embedded_sent = self.text_field_embedder(sent)
        sent_mask = util.get_text_field_mask(sent)
        encoded_sent = self.sent_encoder(embedded_sent, sent_mask)

        embedded_option_a = self.text_field_embedder(reason1)
        option_a_mask = util.get_text_field_mask(reason1)
        encoded_option_a = self.option_a_encoder(embedded_option_a, option_a_mask)

        embedded_option_b = self.text_field_embedder(reason2)
        option_b_mask = util.get_text_field_mask(reason2)
        encoded_option_b = self.option_b_encoder(embedded_option_b, option_b_mask)

        embedded_option_c = self.text_field_embedder(reason3)
        option_c_mask = util.get_text_field_mask(reason3)
        encoded_option_c = self.option_c_encoder(embedded_option_c, option_c_mask)

        logits1 = self.classifier_feedforward(torch.cat([encoded_sent, encoded_option_a], dim=-1))
        logits2 = self.classifier_feedforward(torch.cat([encoded_sent, encoded_option_b], dim=-1))
        logits3 = self.classifier_feedforward(torch.cat([encoded_sent, encoded_option_c], dim=-1))

        logits = F.softmax(torch.cat([logits1, logits2, logits3], dim=-1), dim=1)
        output_dict = {'logits': logits}

        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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