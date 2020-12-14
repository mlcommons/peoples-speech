import os

from lingvo import model_registry
from lingvo.core import base_model_params
from lingvo.core import datasource
from lingvo.core import program
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.core import tokenizers
from lingvo.tasks.asr import input_generator
from lingvo.tasks.asr import ctc_model

@model_registry.RegisterSingleTaskModel
class Peoplesspeech100Base(base_model_params.SingleTaskModelParams):
  """Base parameters for Peoplesspeech 100k hour task."""

  def _CommonInputParams(self, is_eval):
    """Input generator params for Peoplesspeech."""
    p = input_generator.AsrInput.Params()

    # Insert path to the base directory where the data are stored here.
    # Generated using scripts in lingvo/tasks/asr/tools.
    p.file_datasource = datasource.PrefixedDataSource.Params()
    p.file_datasource.file_type = 'tfrecord'
    p.file_datasource.file_pattern_prefix = 'gs://the-peoples-speech-west-europe/PeoplesSpeech/v0.7.1/'

    p.frame_size = 80
    # Interesting. First I've heard of this.
    p.append_eos_frame = False

    p.pad_to_max_seq_length = True
    p.file_random_seed = 0
    p.file_buffer_size = 10000
    # N1 standard 2 has only 2 vCPUs, so we may want a larger machine.
    # https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types
    p.file_parallelism = 16

    if is_eval:
      p.source_max_length = 3600
      p.bucket_upper_bound = [639, 1062, 1275, 1377, 1449, 1506, 1563, 3600]
    else:
      # So it looks like
      p.source_max_length = 1710
      p.bucket_upper_bound = [639, 1062, 1275, 1377, 1449, 1506, 1563, 1710]

    p.bucket_batch_limit = [48] * 8

    return p

  def SetBucketSizes(self, params, bucket_upper_bound, bucket_batch_limit):
    """Sets bucket sizes for batches in params."""
    params.bucket_upper_bound = bucket_upper_bound
    params.bucket_batch_limit = bucket_batch_limit
    return params

  def Train(self):
    p = self._CommonInputParams(is_eval=False)
    p.file_datasource.file_pattern = 'train/train.tfrecords-*'
    p.num_samples = 2292260
    return p

  def Dev(self):
    p = self._CommonInputParams(is_eval=True)
    p.file_datasource.file_pattern = (
        'devtest/dev.tfrecords-00000-of-00001')
    p.num_samples = 3000
    return p

  def Test(self):
    p = self._CommonInputParams(is_eval=True)
    p.file_datasource.file_pattern = (
        'devtest/test.tfrecords-00000-of-00001')
    p.num_samples = 3000
    return p

  def Task(self):
    p = ctc_model.CTCModel.Params()
    p.name = 'peoplesspeech'

    # No default encoder params in this class.

    tp = p.train
    tp.learning_rate = 1e-4
    tp.lr_schedule = schedule.ContinuousSchedule.Params().Set(
        start_step=25_000, half_life_steps=5_000, min=1e-6)
    tp.scale_gradients = False
    tp.l2_regularizer_weight = None

    # Setting p.eval.samples_per_summary to a large value ensures that dev,
    # devother, test, testother are evaluated completely (since num_samples for
    # each of these sets is less than 5000), while train summaries will be
    # computed on 5000 examples.
    p.eval.samples_per_summary = 2700
    p.eval.decoder_samples_per_summary = 2700

    return p

  def ProgramSchedule(self):
    return program.SimpleProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=500,
        eval_dataset_names=['Dev', 'Train'],
        eval_steps_per_loop=50,
        decode_steps_per_loop=0)


@model_registry.RegisterSingleTaskModel
class Peoplesspeech100Grapheme(Peoplesspeech100Base):

  GRAPHEME_TARGET_SEQUENCE_LENGTH = 620
  GRAPHEME_VOCAB_SIZE = 76
  BLANK_IDX = 73

  def InitializeTokenizer(self, params):
    """Initializes a grapheme tokenizer."""
    params.tokenizer = tokenizers.AsciiTokenizer.Params()
    tokp = params.tokenizer
    tokp.vocab_size = self.GRAPHEME_VOCAB_SIZE
    tokp.append_eos = False
    tokp.target_unk_id = 0
    tokp.target_sos_id = 1
    tokp.target_eos_id = 2

    params.target_max_length = self.GRAPHEME_TARGET_SEQUENCE_LENGTH
    return params

  def Train(self):
    p = super().Train()
    return self.InitializeTokenizer(params=p)

  def Dev(self):
    p = super().Dev()
    return self.InitializeTokenizer(params=p)

  def Test(self):
    p = super().Test()
    return self.InitializeTokenizer(params=p)

  def Task(self):
    p = super().Task()
    p.vocab_size = self.GRAPHEME_VOCAB_SIZE
    p.blank_index = self.BLANK_IDX

    return p


@model_registry.RegisterSingleTaskModel
class Grphm_DO_SpecAug_ConvStk_6x512Bidi(Peoplesspeech100Grapheme):

  def Task(self):
    p = super().Task()

    ep = p.encoder_v2
    ep.use_specaugment = True

    elp = p.encoder_v2.lstm_block
    elp.dropout.keep_prob = 0.8
    elp.lstm_cell_size = 512
    elp.num_lstm_layers = 6
    elp.lstm_type = 'bidi'

    ep.stacking_subsampler = None
    ecp = ep.conv_subsampler
    ecp.input_shape = [None, None, 80, 1]
    return p


@model_registry.RegisterSingleTaskModel
class Grphm_DO_SpecAug_ConvStk_6x512Bidi_40batchsize(Peoplesspeech100Grapheme):

  def Train(self):
    p = super().Train()
    # OOM with 48
    p.bucket_batch_limit = [40] * 8
    return p

  def Task(self):
    p = super().Task()

    ep = p.encoder_v2
    ep.use_specaugment = True

    elp = p.encoder_v2.lstm_block
    elp.dropout.keep_prob = 0.8
    elp.lstm_cell_size = 512
    elp.num_lstm_layers = 6
    elp.lstm_type = 'bidi'

    ep.stacking_subsampler = None
    ecp = ep.conv_subsampler
    ecp.input_shape = [None, None, 80, 1]
    return p
