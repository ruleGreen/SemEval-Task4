from typing import List, Dict, Iterable, Any, Set
from collections import defaultdict
import os

import logging
import tqdm
import torch

from allennlp.common import Registrable
from allennlp.common.params import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Instance
from allennlp.data.dataset import Batch
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, MetadataField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.optimizers import Optimizer
from allennlp.training.trainer_base import TrainerBase 

logger = logging.getLogger(__name__)

#@title
@DatasetReader.register("multi-task-a")
class ReaderA(DatasetReader):
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
    
@DatasetReader.register("multi-task-b")
class ReaderB(DatasetReader):
  def __init__(self, tokenizer: Tokenizer = None, token_indexers: Dict[str, TokenIndexer] = None) -> None:
    super().__init__(lazy = False)
    self._tokenizer = tokenizer or WordTokenizer()
    self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

  @overrides
  def text_to_instance(self, sent: str, reason1: str, reason2: str, reason3: str, label: str = None):
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
    with open(cache_path(file_path)) as f:
      lines = f.readlines()[1:]
      for line in lines:
        line = line.strip("\n")
        if not line:
          continue
        splitted_line = line.split(',')
        if len(splitted_line) == 5:
          sent, reason1, reason2, reason3 = splitted_line[1:]
          yield self.text_to_instance(sent, reason1, reason2, reason3)
        elif len(splitted_line) == 6:
          sent, reason1, reason2, reason3, answer = splitted_line[1:]
          yield self.text_to_instance(sent, reason1, reason2, reason3, answer)

@DatasetReader.register("multi-task-c")
class ReaderC(DatasetReader):
	def __init__(
		self, 
        tokenizer: Tokenizer = None, 
        token_indexers: Dict[str, TokenIndexer] = None,
        source_add_start_token: bool = True,
        delimiter: str = ",",
        source_max_tokens: Optional[int] = None,
        target_max_tokens: Optional[int] = None
        ) -> None:
    	super().__init__(lazy = False)
    	self._tokenizer = tokenizer or WordTokenizer()
    	self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
    	self._source_add_start_token = source_add_start_token
    	self._delimiter = delimiter
    	self._source_max_tokens = source_max_tokens
    	self._target_max_tokens = target_max_tokens
    	self._source_max_exceeded = 0
    	self._target_max_exceeded = 0

  
  @overrides
  def _read(self, file_path):
    self._source_max_exceeded = 0
    self._target_max_excedded = 0
    with open(cached_path(file_path), 'r') as data_file:
      for line_num, row in enumerate(csv.reader(data_file, delimiter = self._delimiter)):
        if len(row) != 2:
          raise ConfigurationError(
              "Invalid line format: %s (line number %d)" % (row, line_num + 1)
          )
          source_sequence, target_sequence = row
          yield self.text_to_instance(source_sequence, target_sequence)
    if self._source_max_tokens and self._source_max_exceeded:
      logger.info(
          "In %d instances, the source token length exceeded the max limit (%d) and were truncated.",
          self._source_max_exceeded,
          self._source_max_tokens,
      )
    if self._target_max_tokens and self._target_max_exceeded:
      logger.info(
          "In %d instances, the target token length exceeded  the max limit (%d)",
          self._target_max_exceeded,
          self._target_max_tokens,
      )
  
  @overrides
  def text_to_instance(self, source_string: str, target_string: str = None):
    tokenized_source = self._tokenizer.tokenize(source_string)
    if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
      self._source_max_tokens += 1
      tokenized_source = tokenized_source[:self._source_max_tokens]
    if self._source_add_start_token:
      tokenized_source.insert(0, Token(START_SYMBOL))
    tokenized_source.append(Token(END_SYMBOL))
    source_field = TextField(tokenized_source, self._token_indexers)
    if target_string is not None:
      tokenized_target = self._tokenizer.tokenize(target_string)
      if self._target_max_tokens and len(tokenized_target) > self._target_max_tokens:
        self._target_max_exceeded += 1
        tokenized_target = tokenized_target[:self._target_max_tokens]
      tokenized_target.insert(0, Token(START_SYMBOL))
      tokenized_target.append(Token(END_SYMBOL))
      target_field = TextField(toeknized_target, self._target_token_indexers)
      return Instance({"source_tokens": source_field, "target_tokens": target_field})
    else:
      return Instance({"source_tokens": source_field})


@DatasetMingler.register("round-robin")
class RoundRobinMingler(DatasetMingler):
  """
  Cycle through datasets, ``take_at_time`` instances at a time.
  """
  def __init__(self, dataset_name_field: str = "dataset", take_at_time: int = 1) -> None:
    self.dataset_name_field = dataset_name_field
    self.take_at_time = take_at_time

  def mingle(self, datasets: Dict[str, Iterable[Instance]]) -> Iterable[Instance]:
    iterators = {name: iter(dataset) for name, dataset in datasets.items()}
    done: Set[str] = set()

    while iterators.keys() != done:
      for name, iterator in iterator.items():
        if name not in done:
          try:
            for _ in range(self.take_at_a_time):
              instance = next(iterator)
              instance.fields[self.dataset_name_field] = MetadataField(name)
              yield instance
          except StopIteration:
            done.add(name)

@DataIterator.register("homogeneous-batch")
class HomogeneousBatchIterator(DataIterator):
  """
  An iterator that takes instances of various type
  and yields sing-type batches of them. There's a flag
  to allow mixed-type batches, but at that point you might 
  as well just use ``BasicIterator`` ?
  """
  def __init__(
      self,
      type_field_name: str = "dataset",
      allow_mixed_batches: bool = False,
      batch_size: int = 32
  ) -> None:
    super().__init__(batch_size)
    self.type_field_name = type_field_name
    self.allow_mixed_batches = allow_mixed_batches
  
  def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
    """
    This method should return one epoch worth of batches.
    """
    hoppers: Dict[Any, List[Instance]] = defaultdict(list)

    for instance in instances:
      # which hopper do we put this instance in?
      if self.allow_mixed_batches:
        instance_type = ""
      else:
        instance_type = instance.fields[self.type_field_name].medadata  # type ignore

      hoppers[instance_type].append(instance)

      # if the hopper is full, yield up the batch and clear it
      if len(hoppers[instance_type]) >= self._batch_size:
        yield Batch(hoppers[instance_type])
        hoppers[instance_type].clear()

    # Deal with leftovers
    for remaining in hoppers.values():
      if remaining:
        yield Batch(remaining)  


@Model.register("multi-task")
class MyModel(Model):
  """
  This model does nothing interesting, but it's designed to
  operate on heterogeneous instance using shared parameters
  (well, one shared parameter) like you'd have in multi-task learning.
  """
  def __init__(self, vovab: Vocabulary) -> None:
    super().__init__(vocab)
    self.weight = torch.nn.Parameter(torch.randn(()))
  
  def forward(self,
              datasets: List[str],
              field_a: torch.Tensor = None,
              field_b: torch.Tensor = None
            ) -> Dict[str, torch.Tensor]:
    loss = torch.tensor(0.0)
    if dataset[0] == "a":
      loss += field_a["tokens"].sum() * self.weight
    elif dataset[1] == "b":
      loss += field_b["tokens"].sum() * self.weight ** 2
    else:
      raise ValueError(f"unknown dataset: {dataset[0]}")
    
    return {"loss": loss}


@TrainerBase.register("multi-task")
class MultiTaskTrainer(TrainerBase):
  """
  A simple trainer that works in our mulit-task setup.
  Really the main thing that makes this task not fit into our
  existing trainer is the multiple datasets.
  """
  def __init__(
      self,
      model: Model,
      serialization_dir: str,
      iterator: DataIterator,
      mingler: DatasetMingler,
      optimizer: torch.optim.Optimizer,
      datasets: Dict[str, Iterable[Instance]],
      num_epochs: int = 10,
      num_serialized_models_to_keep: int = 10,
  ) -> None:
    super().__init__(serialization_dir)
    self.model = model
    self.iterator = iterator
    self.mingler = mingler
    self.optimizer = optimizer
    self.datasets = datasets
    self.num_epochs = num_epochs
    self.checkpointer = Checkpointer(
        serialization_dir, num_serialized_models_to_keep = num_serialized_models_to_keep
    )

  def save_checkpoint(self, epoch: int) -> None:
    training_state = {"epoch": epoch, "optimizer": self.optimizer.state_dict()}
    self.checkpointer.save_checkpoint(epoch, self.model.state_dict(), training_state, True)

  def restore_checkpoint(self) -> int:
    model_state, trainer_state = self.checkpoint.restore_checkpoint()
    if not model_state and not trainer_state:
      return 0
    else:
      self.model.load_state_dict(model_state)
      self.optimizer.load_state_dict(trainer_state["optimizer"])
      return trainer_state["epoch"] + 1
  
  def train(self) -> Dict:
    start_epoch = self.restore_checkpoint()

    self.model.train()
    for epoch in range(start_epoch, self.num_epochs):
      total_loss = 0.0
      batches = tqdm.tqdm(self.iterator(self.mingler.mingle(self.datasets), num_epochs = 1))
      for i, batch in enumerate(batches):
        self.optimizer.zero_grad()
        loss = self.model.forward(**batch)["loss"]   # type: ignore
        loss.backward()
        total_loss += loss.item()
        self.optimizer.step()
        batches.set_description(f"epoch: {epoch} loss: {total_loss / (i + 1)}")

      # Save checkpoint
      self.save_checkpoint(epoch)
  
    return {}
  
  @classmethod
  def from_params(   # type: ignore
      cls,
      params: Params,
      serialization_dir: str,
      recover: bool = False,
      cache_directory: str = None,
      cache_prefix: str = None,
  ) -> "MultiTaskTrainer":
    readers = {
        name: DatasetReader.from_params(reader_params)
        for name, reader_params in params.pop("train_dataset_readers").items()
    }
    train_file_paths = params.pop("train_file_paths").as_dict()

    datasets = {name: reader.read(train_file_paths[name]) for name, reader in readers.items()}

    instances = (instance for dataset in datasets.values() for instance in dataset)
    vocab = Vocabulary.from_params(Params({}), instances)
    model = Model.from_params(Params({}), instances)
    iterator = DataIterator.from_params(params.pop("iterator"))
    iterator.index_with(vocab)
    mingler = DatasetMingler.from_params(params.pop("mingler"))

    parameters = [[n,p] for n, p in model.named_parameters() if p.requires_grad]
    optimizer = Optimizer.from_params(parameters, params.pop("optimizer"))

    num_epochs = params.pop_int("num_epochs", 10)
    
    _ = params.pop("trainer", Params({}))

    params.assert_empty(__name__)

    return MultiTaskTrainer(
        model, serialization_dir, iterator, mingler, optimizer, datasets, num_epochs
    )

class MultiTask(AllenNlpTestCase):
  def setUp(self):
    super().setUp()

    params = Params(
      {
        "model": {"type": "multi-task"},
        "iterator": {"type": "homogeneous-batch"},
        "mingler": {"type": "round-robin"},
        "optimizer": {"type": "sgd", "lr": 0.01},
        "train_dataset_reader": {
            
        },
        "train_file_path": {
            
        },
        "trainer": {"type": "multi-task"}
      }
    )

    self.trainer = TrainerBase.from_params(params, self.TEST_DIR)
  
  def test_training(self):
    self.trainer.train()

    assert os.path.exists(os.path.join(self.TEST_DIR, "best.th"))

if __name__ == "__main__":
	MultiTask = MultiTaskTest()
	MultiTask.setUp()
	MultiTask.test_training()
	