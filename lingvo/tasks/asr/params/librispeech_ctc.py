from lingvo import model_registry
import lingvo.compat as tf
from lingvo.core import base_model_params
from lingvo.core import datasource
from lingvo.core import program
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.core import tokenizers
from lingvo.tasks.asr import input_generator
from lingvo.tasks.asr import ctc_model
from lingvo.tasks.asr import frontend as asr_frontend

VOCAB_FILEPATH = "gs://the-peoples-speech-west-europe/Librispeech/tokens2.txt"

tf.flags.DEFINE_integer(
    'ctc_inference_model_num_samples', 0, 'Total number of samples to run inference on. Should be equal to input.num_samples (this is checked)')
tf.flags.DEFINE_integer(
    'ctc_inference_model_batch_size', 0,
  '')
tf.flags.DEFINE_integer(
    'ctc_inference_model_number_of_tpus', 8,
  '')



FLAGS = tf.flags.FLAGS


def intialize_vocab_file_tokenizer_params():
  pt = tokenizers.VocabFileTokenizer.Params()
  pt.append_eos = False

  # TODO: Don't hard-code this path! How can I make it relative?
  pt.token_vocab_filepath = VOCAB_FILEPATH
  pt.tokens_delimiter = ""
  pt.load_token_ids_from_vocab = False
  with tf.io.gfile.GFile(VOCAB_FILEPATH, mode='r') as fh:
    lines = fh.readlines()
    for i, line in enumerate(lines):
      if line == "<unk>\n":
        pt.target_unk_id = i
      elif line == "<s>\n":
        pt.target_sos_id = i
      elif line == "</s>\n":
        pt.target_eos_id = i
    pt.vocab_size = len(lines)
  assert pt.vocab_size == 32
  assert pt.target_unk_id == 0
  assert pt.target_sos_id == 1, pt.target_sos_id
  assert pt.target_eos_id == 2, pt.target_eos_id

  return pt

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
    p.file_datasource.file_pattern_prefix = 'gs://the-peoples-speech-west-europe/Librispeech'

    p.frame_size = 80
    # Interesting. First I've heard of this.
    p.append_eos_frame = False

    if tf.flags.FLAGS.tpu:
      p.pad_to_max_seq_length = True
    else:
      p.pad_to_max_seq_length = False
    p.file_random_seed = 0
    p.file_buffer_size = 10000
    p.file_parallelism = 16

    if is_eval:
      p.source_max_length = 3600
      p.bucket_upper_bound = [639, 1062, 1275, 1377, 1449, 1506, 1563, 3600]
    else:
      # So it looks like
      p.source_max_length = 1710
      p.bucket_upper_bound = [639, 1062, 1275, 1377, 1449, 1506, 1563, 1710]

    p.bucket_batch_limit = [48] * 8


    p.tokenizer = intialize_vocab_file_tokenizer_params()

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
    p.target_max_length = 516
    # p.source_max_length = 3600
    # p.bucket_upper_bound = [3600]
    # p.bucket_batch_limit = [1]
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

    with tf.io.gfile.GFile(VOCAB_FILEPATH, mode='r') as fh:
      lines = fh.readlines()
      assert lines[-1] == "<blk>\n"
      p.vocab_size = len(lines)
      p.blank_index = 31

    assert p.vocab_size == 32
    assert p.blank_index == 31

    ep = p.encoder_v2
    ep.use_stacking_subsampler = True

    # No default encoder params in this class.

    tp = p.train
    tp.learning_rate = 1e-4
    tp.lr_schedule = schedule.ContinuousSchedule.Params().Set(
        start_step=25_000, half_life_steps=5_000, min=1e-6)
    tp.scale_gradients = False
    tp.l2_regularizer_weight = None

    # A value of 0 will force each eval and dev task to iterate over
    # the entire dataset.
    p.eval.samples_per_summary = 0
    p.eval.decoder_samples_per_summary = 0

    return p

  # From what I can tell, this is used only by ExecutorTpu. So perhaps
  # I could change it such that it runs only on the Train set... Right?
  def ProgramSchedule(self):
    return program.SimpleProgramScheduleForTask(
        train_dataset_name='Train',
        train_steps_per_loop=100,
      # I want to compute WER...
        eval_dataset_names=['Dev'],
        eval_steps_per_loop=1,
        decode_steps_per_loop=1,)


# @model_registry.RegisterSingleTaskModel
# class TpuDecoderLibrispeech960Base(Librispeech960Base):

#   SAMPLES_PER_SECOND = 16_000

#   def _InferenceInputParams(self):
#     """
#     Reads in a raw waveform, where samples are float32 and in the range[-1,1]
#     """
#     p = input_generator.RawAsrInputIntegerUttIds.Params()

#     p.file_datasource = datasource.PrefixedDataSource.Params()
#     p.file_datasource.file_type = 'tfrecord'
#     p.file_datasource.file_pattern_prefix = 'REPLACE_ME'
#     p.file_datasource.file_pattern = '*.tfrecord'

#     p.frame_size = 1
#     p.append_eos_frame = False

#     p.pad_to_max_seq_length = True
#     p.file_random_seed = 0
#     p.file_buffer_size = 1
#     # Set this to whatever
#     p.file_parallelism = 1

#     # 15 seconds is our maximum utterance size
#     max_sample_length = 15 * type(self).SAMPLES_PER_SECOND
#     p.source_max_length = max_sample_length
#     p.bucket_upper_bound = [p.source_max_length]

#     p.bucket_batch_limit = [1]

#     return p

#   def Dev(self):
#     return self._InferenceInputParams()

#   def Task(self):
#     p = super().Task()
#     # This is copied from audio_lib.py. Ideally these configurations
#     # would be split off into a function, but I want to minimize merge
#     # conflicts with lingvo for now. Of course, a merge conflict would
#     # indicate that parameters had changed, so my choice is clearly
#     # wrong, but oh well.
#     p.frontend = asr_frontend.MelAsrFrontend.Params()
#     pf = p.frontend
#     pf.sample_rate = float(type(self).SAMPLES_PER_SECOND)
#     pf.frame_size_ms = 25.
#     pf.frame_step_ms = 10.
#     pf.num_bins = 80
#     pf.lower_edge_hertz = 125.
#     pf.upper_edge_hertz = 7600.
#     pf.preemph = 0.97
#     pf.noise_scale = 0.
#     pf.pad_end = False

#     p.inference_compute_only_log_softmax = True
#     return p

#   def ProgramSchedule(self):
#     # This value must be parameterized
#     number_of_inputs = FLAGS.ctc_inference_model_num_samples
#     batch_size = 8
#     import math
#     decode_steps_per_loop = math.ceil(number_of_inputs / batch_size)
#     return program.DecodeProgramSchedule(
#       eval_dataset_names=['Dev'],
#       decode_steps_per_loop=decode_steps_per_loop,  # 1,
#       # I may want to reevaluate this
#       experimental_decoder=True)


@model_registry.RegisterSingleTaskModel
class Librispeech960Grapheme(Librispeech960Base):

  GRAPHEME_TARGET_SEQUENCE_LENGTH = 620
  GRAPHEME_VOCAB_SIZE = 76
  BLANK_IDX = 73

  # This seems bad. Does this undo my VocabFileTokenizer???
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

    return p

@model_registry.RegisterSingleTaskModel
class Grphm_DO_SpecAug_StackingSubSampler(Librispeech960Base):

  def Task(self):
    p = super().Task()

    ep = p.encoder_v2
    ep.use_specaugment = True
    ep.use_conv_subsampler = False
    ep.use_stacking_subsampler = True

    ep.stacking_subsampler.stacking.left_context = 1
    ep.stacking_subsampler.stacking.right_context = 1
    ep.stacking_subsampler.stacking.stride = (
      ep.stacking_subsampler.stacking.left_context +
      1 +
      ep.stacking_subsampler.stacking.right_context
    )

    elp = p.encoder_v2.lstm_block
    elp.dropout.keep_prob = 0.8
    elp.lstm_cell_size = 1024
    elp.num_lstm_layers = 6
    elp.lstm_type = 'fwd'

    return p


@model_registry.RegisterSingleTaskModel
class Grphm_DO_SpecAug_ConvStk_6x512Bidi(Librispeech960Grapheme):

  def Task(self):
    p = super().Task()

    ep = p.encoder_v2
    ep.use_specaugment = True
    ep.use_conv_subsampler = True
    ep.use_stacking_subsampler = False

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

    ep = p.encoder_v2
    ep.use_specaugment = True
    ep.use_conv_subsampler = True
    ep.use_stacking_subsampler = False

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
class TpuDecoderGrphm_DO_SpecAug_StackingSubSampler(Grphm_DO_SpecAug_StackingSubSampler):

  SAMPLES_PER_SECOND = 16_000

  def _InferenceInputParams(self):
    """
    Reads in a raw waveform, where samples are float32 and in the range[-1,1]
    """
    p = input_generator.RawAsrInputIntegerUttIds.Params()

    p.file_datasource = datasource.PrefixedDataSource.Params()
    p.file_datasource.file_type = 'tfrecord'
    p.file_datasource.file_pattern_prefix = 'REPLACE_ME'
    p.file_datasource.file_pattern = '*.tfrecord'

    p.frame_size = 1
    p.append_eos_frame = False

    p.pad_to_max_seq_length = True
    p.file_random_seed = 0
    p.file_buffer_size = 1
    # Set this to whatever
    p.file_parallelism = 1

    # 15 seconds is our maximum utterance size
    max_sample_length = 15 * type(self).SAMPLES_PER_SECOND
    # TODOL: change to max_sample_length
    p.source_max_length = max_sample_length
    p.bucket_upper_bound = [p.source_max_length]

    p.bucket_batch_limit = [FLAGS.ctc_inference_model_batch_size]

    # assert p.num_samples == FLAGS.ctc_inference_model_num_samples

    p.tokenizer = intialize_vocab_file_tokenizer_params()

    return p

  def Dev(self):
    return self._InferenceInputParams()

  def Task(self):
    p = super().Task()
    # This is copied from audio_lib.py. Ideally these configurations
    # would be split off into a function, but I want to minimize merge
    # conflicts with lingvo for now. Of course, a merge conflict would
    # indicate that parameters had changed, so my choice is clearly
    # wrong, but oh well.
    p.frontend = asr_frontend.MelAsrFrontend.Params()
    pf = p.frontend
    pf.sample_rate = float(type(self).SAMPLES_PER_SECOND)
    pf.frame_size_ms = 25.
    pf.frame_step_ms = 10.
    pf.num_bins = 80
    pf.lower_edge_hertz = 125.
    pf.upper_edge_hertz = 7600.
    pf.preemph = 0.97
    pf.noise_scale = 0.
    pf.pad_end = False

    p.inference_compute_only_log_softmax = True

    # TODO: Consider doing this
    # ep = p.encoder_v2
    # ep.use_specaugment = False
    return p

  def ProgramSchedule(self):
    # TODO: Generalize this
    number_of_inputs = FLAGS.ctc_inference_model_num_samples
    assert number_of_inputs != 0
    assert FLAGS.ctc_inference_model_batch_size != 0
    batch_size = FLAGS.ctc_inference_model_batch_size * FLAGS.ctc_inference_model_number_of_tpus
    import math
    decode_steps_per_loop = math.ceil(number_of_inputs / batch_size)
    return program.DecodeProgramSchedule(
      eval_dataset_names=['Dev'],
      decode_steps_per_loop=decode_steps_per_loop,  # 1,
      # I may want to reevaluate this
      experimental_decoder=False)
