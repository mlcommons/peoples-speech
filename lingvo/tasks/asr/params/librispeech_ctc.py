from lingvo import model_registry
from lingvo.core import base_model_params
from lingvo.core import datasource
from lingvo.core import program
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.core import tokenizers
from lingvo.tasks.asr import input_generator
from lingvo.tasks.asr import ctc_model


# top-most layer is a Model
# Recursively built of layers, each having params. Analgous torch.nn.Module

# Task:
# Model: Params() -> class CTCModel -> Params() -> key->value
# Dataset: Params()
# How to load dataset (bucketing, sorting, how many passes)
# One or more objective functions
@model_registry.RegisterSingleTaskModel
class Librispeech960Base(base_model_params.SingleTaskModelParams):
  """Base parameters for Librispeech 960 hour task."""

  def _CommonInputParams(self, is_eval):
    """Input generator params for Librispeech."""
    p = input_generator.AsrInput.Params()

    # Insert path to the base directory where the data are stored here.
    # Generated using scripts in lingvo/tasks/asr/tools.
    p.file_datasource = datasource.PrefixedDataSource.Params()
    p.file_datasource.file_type = 'tfrecord'
    # AG TODO: chang this local path to gs path for orig data (this is for testing only)
    p.file_datasource.file_pattern_prefix = 'gs://the-peoples-speech-west-europe/Librispeech'
    # p.file_datasource.file_pattern_prefix = '/home/anjali/data/mlcommons/librispeech/data'
    # TODO: Use an abseil flag for this.
    # p.file_datasource.file_pattern_prefix = '/export/b02/ws15dgalvez/kaldi-data/librispeech'

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

    # AG TODO: For TPU
    p.bucket_batch_limit = [48] * 8
    # AG TODO: For GPU and CPU, both training and evaluation
    # p.bucket_batch_limit = [96] * 8
    # p.bucket_batch_limit = [12] * 8

    return p

  def SetBucketSizes(self, params, bucket_upper_bound, bucket_batch_limit):
    """Sets bucket sizes for batches in params."""
    params.bucket_upper_bound = bucket_upper_bound
    params.bucket_batch_limit = bucket_batch_limit
    return params

  def Train(self):
    p = self._CommonInputParams(is_eval=False)
    p.file_datasource.file_pattern = 'train/train.tfrecords-*'
    p.num_samples = 281241
    return p

  def Dev(self):
    p = self._CommonInputParams(is_eval=True)
    p.file_datasource.file_pattern = (
        'devtest/dev-clean.tfrecords-00000-of-00001')
    p.num_samples = 2703
    return p

  def Devother(self):
    p = self._CommonInputParams(is_eval=True)
    p.file_datasource.file_pattern = (
        'devtest/dev-other.tfrecords-00000-of-00001')
    p.num_samples = 2864
    return p

  def Test(self):
    p = self._CommonInputParams(is_eval=True)
    p.file_datasource.file_pattern = (
        'devtest/test-clean.tfrecords-00000-of-00001')
    p.num_samples = 2620
    return p

  def Testother(self):
    p = self._CommonInputParams(is_eval=True)
    p.file_datasource.file_pattern = (
        'devtest/test-other.tfrecords-00000-of-00001')
    p.num_samples = 2939
    return p

  def Task(self):
    p = ctc_model.CTCModel.Params()
    p.name = 'librispeech'

    # Initialize encoder params.
    ep = p.encoder
    ep.use_specaugment = False
    ep.input_shape = [None, None, 80, 1]
    ep.pad_steps = 0
    ep.lstm_cell_size = 1024
    ep.num_lstm_layers = 5
    ep.lstm_type = 'fwd'
    ep.cnn_tpl.params_init = py_utils.WeightInit.Gaussian(0.001)
    # Disable conv & conv LSTM layers.
    ep.project_lstm_output = False
    ep.num_cnn_layers = 0
    ep.conv_filter_shapes = []
    ep.conv_filter_strides = []
    ep.num_conv_lstm_layers = 0

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
class Librispeech960Grapheme(Librispeech960Base):

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

  def Devother(self):
    p = super().Devother()
    return self.InitializeTokenizer(params=p)

  def Test(self):
    p = super().Test()
    return self.InitializeTokenizer(params=p)

  def Testother(self):
    p = super().Testother()
    return self.InitializeTokenizer(params=p)

  def Task(self):
    p = super().Task()
    p.vocab_size = self.GRAPHEME_VOCAB_SIZE
    p.blank_index = self.BLANK_IDX

    # input_stacking
    p.encoder.input_shape = [None, None, 240, 1]
    sp = p.input_stacking_tpl
    sp.left_context = 1
    sp.right_context = 1
    sp.stride = 3  # L + 1 + R

    return p


class Librispeech960Wpm(Librispeech960Base):

  # Set this to a WPM vocabulary file before training. By default, we use the
  # pre-generated 16K word piece vocabulary checked in under 'tasks/asr/'.
  WPM_SYMBOL_TABLE_FILEPATH = (
      'lingvo/tasks/asr/wpm_16k_librispeech_ascii.vocab')
  WPM_TARGET_SEQUENCE_LENGTH = 140
  WPM_VOCAB_SIZE = 2555   # 16328
  BLANK_IDX = 4

  EMBEDDING_DIMENSION = 96
  NUM_TRAINING_WORKERS = 8

  def InitializeTokenizer(self, params):
    """Initializes a Word Piece Tokenizer."""
    params.tokenizer = tokenizers.WpmTokenizer.Params()
    tokp = params.tokenizer
    tokp.vocab_filepath = self.WPM_SYMBOL_TABLE_FILEPATH
    tokp.vocab_size = self.WPM_VOCAB_SIZE
    tokp.append_eos = False
    tokp.target_unk_id = 0
    tokp.target_sos_id = 1
    tokp.target_eos_id = 2

    params.target_max_length = self.WPM_TARGET_SEQUENCE_LENGTH
    return params

  def Train(self):
    p = super().Train()
    return self.InitializeTokenizer(params=p)

  def Dev(self):
    p = super().Dev()
    return self.InitializeTokenizer(params=p)

  def Devother(self):
    p = super().Devother()
    return self.InitializeTokenizer(params=p)

  def Test(self):
    p = super().Test()
    return self.InitializeTokenizer(params=p)

  def Testother(self):
    p = super().Testother()
    return self.InitializeTokenizer(params=p)

  def Task(self):
    p = super().Task()
    p.vocab_size = self.WPM_VOCAB_SIZE
    p.blank_index = self.BLANK_IDX
    return p


@model_registry.RegisterSingleTaskModel
class Old_Grphm_DO_SpecAug_InptStack_6x1024(Librispeech960Grapheme):
  def Task(self):
    p = super().Task()
    p.encoder_v2 = None

    p.encoder.use_specaugment = True
    p.encoder.input_shape = [None, None, 240, 1]
    p.encoder.lstm_dropout.keep_prob = 0.8
    p.encoder.lstm_cell_size = 1024
    p.encoder.num_lstm_layers = 6

    sp = p.input_stacking_tpl
    sp.left_context = 1
    sp.right_context = 1
    sp.stride = 3  # L + 1 + R

    return p


@model_registry.RegisterSingleTaskModel
class Grphm_DO_SpecAug_InptStack_6x1024(Librispeech960Grapheme):
  def Task(self):
    p = super().Task()

    # disable old style
    p.encoder = None
    p.input_stacking_tpl = None

    # new style encoder
    ep = p.encoder_v2
    ep.use_specaugment = True

    elp = ep.lstm_block
    elp.lstm_cell_size = 1024
    elp.num_lstm_layers = 6
    elp.lstm_type = 'fwd'
    elp.dropout.keep_prob = 0.8

    ep.conv_subsampler = None
    esp = ep.stacking_subsampler.stacking
    esp.left_context = 1
    esp.right_context = 1
    esp.stride = 3  # L + 1 + R

    return p

@model_registry.RegisterSingleTaskModel
class Grphm_DO_SpecAug_InptStack_6x512Bidi(Grphm_DO_SpecAug_InptStack_6x1024):
  def Task(self):
    p = super().Task()
    elp = p.encoder_v2.lstm_block
    elp.lstm_cell_size = 512
    elp.num_lstm_layers = 6
    elp.lstm_type = 'bidi'
    return p

@model_registry.RegisterSingleTaskModel
class Grphm_DO_SpecAug_ConvStack_6x1024(Librispeech960Grapheme):
  def Task(self):
    p = super().Task()

    # disable old style
    p.encoder = None
    p.input_stacking_tpl = None

    # new style encoder
    ep = p.encoder_v2
    ep.use_specaugment = True

    elp = ep.lstm_block
    elp.lstm_cell_size = 1024
    elp.num_lstm_layers = 6
    elp.lstm_type = 'fwd'
    elp.dropout.keep_prob = 0.8

    ep.stacking_subsampler = None
    ecp = ep.conv_subsampler
    ecp.input_shape = [None, None, 80, 1]

    return p

@model_registry.RegisterSingleTaskModel
class Wpm_DO_SpecAug_InptStack_6x1024(Librispeech960Wpm):
  def Task(self):
    p = super().Task()

    # disable old style
    p.encoder = None
    p.input_stacking_tpl = None

    # new style encoder
    ep = p.encoder_v2
    ep.use_specaugment = True

    elp = ep.lstm_block
    elp.lstm_cell_size = 1024
    elp.num_lstm_layers = 6
    elp.lstm_type = 'fwd'
    elp.dropout.keep_prob = 0.8

    ep.conv_subsampler = None
    esp = ep.stacking_subsampler.stacking
    esp.left_context = 2
    esp.right_context = 1
    esp.stride = 4  # L + 1 + R

    return p

@model_registry.RegisterSingleTaskModel
class Grphm_DO_SpecAug_ConvStk_6x512Bidi(Librispeech960Grapheme):
  def Task(self):
    p = super().Task()

    # disable old style
    p.encoder = None
    p.input_stacking_tpl = None

    # new style encoder
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
class Grphm_DO_SpecAug_ConvStk_6x512Bidi_40batchsize(Librispeech960Grapheme):
  def Train(self):
    p = super().Train()
    # OOM with 48
    p.bucket_batch_limit = [40] * 8
    return p

  def Task(self):
    p = super().Task()

    # disable old style
    p.encoder = None
    p.input_stacking_tpl = None

    # new style encoder
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
class Grphm_DO_SpecAug_ConvStk_6x768Bidi_30batchsize(Librispeech960Grapheme):
  def Train(self):
    p = super().Train()
    # OOM with 48
    p.bucket_batch_limit = [30] * 8
    return p

  def Task(self):
    p = super().Task()

    # disable old style
    p.encoder = None
    p.input_stacking_tpl = None

    # new style encoder
    ep = p.encoder_v2
    ep.use_specaugment = True

    elp = p.encoder_v2.lstm_block
    elp.dropout.keep_prob = 0.8
    elp.lstm_cell_size = 768
    elp.num_lstm_layers = 6
    elp.lstm_type = 'bidi'

    ep.stacking_subsampler = None
    ecp = ep.conv_subsampler
    ecp.input_shape = [None, None, 80, 1]
    return p

@model_registry.RegisterSingleTaskModel
class Grphm_DO_SpecAug_ConvStk_6x768Bidi_32batchsize(Librispeech960Grapheme):
  def Train(self):
    p = super().Train()
    # OOM with 48
    p.bucket_batch_limit = [32] * 8
    return p

  def Task(self):
    p = super().Task()

    # disable old style
    p.encoder = None
    p.input_stacking_tpl = None

    # new style encoder
    ep = p.encoder_v2
    ep.use_specaugment = True

    elp = p.encoder_v2.lstm_block
    elp.dropout.keep_prob = 0.8
    elp.lstm_cell_size = 768
    elp.num_lstm_layers = 6
    elp.lstm_type = 'bidi'

    ep.stacking_subsampler = None
    ecp = ep.conv_subsampler
    ecp.input_shape = [None, None, 80, 1]
    return p