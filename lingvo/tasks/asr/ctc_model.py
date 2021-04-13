# Lint as: python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CTC model."""

from collections import defaultdict
import numpy as np
import os

import lingvo.compat as tf
from lingvo.core import base_model
from lingvo.core import metrics
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.tasks.asr import blocks
from lingvo.tasks.asr import encoder_v2
from lingvo.tasks.asr import decoder_utils
from lingvo.tasks.asr import input_generator

# https://stackoverflow.com/a/2912455
class keydefaultdict(defaultdict):
  def __missing__(self, key):
    if self.default_factory is None:
      raise KeyError(key)
    else:
      ret = self[key] = self.default_factory(key) # pylint: disable=not-callable
      return ret

class CTCModel(base_model.BaseTask):
  """
  CTC model without a language model.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
      # Need to set frontend appropriately
        'frontend', None,
        'ASR frontend to extract features from input. Defaults to no frontend '
        'which means that features are taken directly from the input.')

    p.Define('encoder_v2', encoder_v2.AsrEncoder.Params(), 'Encoder V2 Params')

    # Defaults based on graphemes / ascii_tokenizer.cc
    p.Define('vocab_size', 76,
             'Vocabulary size, *including* the blank symbol.')
    p.Define(
        'blank_index', 73, 'Index assigned to epsilon, aka blank for CTC. '
        'This should never appear in the label sequence. Reconsider this.')

    p.Define('inference_compute_only_log_softmax', False,
             'At inference time, compute the output of log softmax, rather '
             'than running any sort of CTC decoder.')
    p.Define('log_softmax_output_directory', '',
             'Path to which to dump log_softmax values')

    tp = p.train
    tp.lr_schedule = (schedule.PiecewiseConstantSchedule.Params().Set(
        boundaries=[350000, 450000, 600000], values=[1.0, 0.1, 0.01, 0.001]))

    tp.vn_start_step = 20_000
    tp.vn_std = 0.075
    tp.l2_regularizer_weight = 1e-6
    tp.clip_gradient_norm_to_value = 1.0
    tp.grad_norm_to_clip_to_zero = 100.0
    tp.tpu_steps_per_loop = 100

    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    if not p.encoder_v2.name:
      p.encoder_v2.name = 'enc'
    self.CreateChild('encoder', p.encoder_v2)

    if p.frontend:
      self.CreateChild('frontend', p.frontend)

    projection_p = blocks.VocabProjectionBlock.Params()
    projection_p.vocab_size = p.vocab_size
    projection_p.input_dim = self.encoder.output_dim
    self.CreateChild('project_to_vocab_size', projection_p)

    if p.inference_compute_only_log_softmax:
      assert p.log_softmax_output_directory, 'Must provide non-empty path to which to dump log softmax values'
      # `ulimit -n` gives me 1024 on this machine. We may need to
      # increase the number of open file descriptors allowed in this
      # process
      class WriterAndRecordsWritten:
        records_written: int
        file_handle: tf.io.TFRecordWriter
        def __init__(self, write_path: str):
          self.records_written = 0
          self.file_handle = tf.io.TFRecordWriter(write_path)
      self._int64_audio_document_id_to_file_handle = keydefaultdict(
        WriterAndRecordsWritten)

  def ComputePredictions(self, theta, input_batch):
    return self._FrontendAndEncoderFProp(theta, input_batch.src)

  def DecodeWithTheta(self, theta, input_batch):
    """Constructs the inference graph."""
    p = self.params
    # from IPython import embed; embed()
    with tf.name_scope('decode'), tf.name_scope(p.name):
      with tf.name_scope('encoder'):
        encoder_outputs = self._FrontendAndEncoderFProp(theta, input_batch.src)
      if p.inference_compute_only_log_softmax:
        global_step = tf.train.get_global_step()
        increment_global_step = tf.assign(global_step, global_step + 1)
        with tf.control_dependencies([increment_global_step]):
          log_probabilities = tf.transpose(tf.nn.log_softmax(encoder_outputs.encoded, axis=2),
                                           perm=(1, 0, 2))
        with tf.name_scope('decoder'):
          decoder_outs = self._DecodeCTC(encoder_outputs)
        # encoder_outputs's shape is [T,B,F]
        return {'log_probabilities': log_probabilities,
                'log_probabilities_lengths':
                py_utils.LengthsFromBitMask(encoder_outputs.padding, 0),
                'int64_uttid': input_batch.sample_ids,
                'int64_audio_document_id': input_batch.audio_document_ids,
                'num_utterances_in_audio_document': input_batch.num_utterances_in_audio_document,
                'transcripts': decoder_outs.transcripts,
        }
      with tf.name_scope('decoder'):
        decoder_outs = self._DecodeCTC(encoder_outputs)

      decoder_metrics = py_utils.RunOnTpuHost(self._CalculateErrorRates, decoder_outs, input_batch)
      return decoder_metrics

  def PostProcessDecodeOut(self, decode_out_dict, dec_metrics_dict):
    p = self.params
    if p.inference_compute_only_log_softmax:
      mini_batch_size = decode_out_dict['log_probabilities'].shape[0]
      dec_metrics_dict['num_samples_in_batch'].Update(mini_batch_size)

      for i in range(mini_batch_size):
        int64_uttid = decode_out_dict['int64_uttid'][i]
        int64_audio_document_id = decode_out_dict['int64_audio_document_id'][i]
        if int64_uttid == input_generator.RawAsrInputIntegerUttIds.PAD_INDEX:
          continue
        length = decode_out_dict['log_probabilities_lengths'][i]
        flat_log_probabilities = decode_out_dict['log_probabilities'][i, :length, :].flatten()
        record_bytes = tf.train.Example(
          features=tf.train.Features(feature={
            'int64_uttid': tf.train.Feature(
              int64_list=tf.train.Int64List(value=[int64_uttid])),
            'log_probabilities': tf.train.Feature(
              float_list=tf.train.FloatList(value=flat_log_probabilities)),
            'transcripts': tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[decode_out_dict['transcripts'][i]])),
          })
        ).SerializeToString()
        output_path = os.path.join(p.log_softmax_output_directory, f'int64_audio_document_id={int64_audio_document_id}/shard.tfrecord')
        pair = self._int64_audio_document_id_to_file_handle[output_path]
        pair.file_handle.write(record_bytes)
        pair.records_written += 1
        if pair.records_written == decode_out_dict['num_utterances_in_audio_document'][i]:
          pair.file_handle.close()
          del self._int64_audio_document_id_to_file_handle[output_path]
      return

    gt_transcripts = decode_out_dict['target_transcripts']
    hyp_transcripts = decode_out_dict['decoded_transcripts']
    if not py_utils.use_tpu():
      utt_id = decode_out_dict['utt_id']
    else:
      utt_id = [["TPU does not know utt_id, sorry"]] * len(gt_transcripts)

    for i in range(len(gt_transcripts)):
      # tf.logging.info('utt_id : %s', utt_id[i][0])
      tf.logging.info('ref_str: %s', gt_transcripts[i])
      tf.logging.info('hyp_str: %s', hyp_transcripts[i])

    total_word_err = np.sum(decode_out_dict['num_wrong_words'])
    total_ref_words = np.sum(decode_out_dict['num_ref_words'])
    total_char_err = np.sum(decode_out_dict['num_wrong_chars'])
    total_ref_chars = np.sum(decode_out_dict['num_ref_chars'])

    dec_metrics_dict['num_samples_in_batch'].Update(len(gt_transcripts))
    dec_metrics_dict['wer'].Update(total_word_err / max(1., total_ref_words),
                                   total_ref_words)
    dec_metrics_dict['cer'].Update(total_char_err / max(1., total_ref_chars),
                                   total_ref_chars)
    tf.logging.info(' ]]] CER: %.3f ]]] WER: %.3f',
                    dec_metrics_dict['cer'].value,
                    dec_metrics_dict['wer'].value)

  def ComputeLoss(self, theta, predictions, input_batch):
    output_batch = predictions
    assert self.params.blank_index == 31
    ctc_loss = tf.nn.ctc_loss(
        input_batch.tgt.labels,
        output_batch.encoded,
        py_utils.LengthsFromBitMask(input_batch.tgt.paddings, 1),
        py_utils.LengthsFromBitMask(output_batch.padding, 0),
        logits_time_major=True,
        blank_index=self.params.blank_index)

    # ctc_loss.shape = (B)
    total_loss = tf.reduce_mean(ctc_loss)
    per_sequence_loss = {'loss': ctc_loss}
    return dict(loss=(total_loss, 1.0)), per_sequence_loss

  def CreateDecoderMetrics(self):
    base_metrics = {
        'wer': metrics.AverageMetric(),
        'cer': metrics.AverageMetric(),
        'num_samples_in_batch': metrics.AverageMetric()
    }
    return base_metrics

  @classmethod
  def FPropMeta(cls, params, *args, **kwargs):
    raise NotImplementedError('No FPropMeta available.')

  def _FrontendAndEncoderFProp(self, theta, input_batch_src):
    p = self.params

    if p.frontend:
      input_batch_src = self.frontend.FProp(theta.frontend, input_batch_src)

    encoder_out = self.encoder.FProp(theta.encoder, input_batch_src)

    encoded, padding = self.project_to_vocab_size(encoder_out.encoded,
                                                  encoder_out.padding)
    outputs = py_utils.NestedMap(encoded=encoded, padding=padding)
    return outputs

  def _DecodeCTC(self, output_batch):
    tok_logits = output_batch.encoded  # (T, B, F)

    # GALVEZ: Make beam_width a tunable parameter!
    # (decoded,), _ = py_utils.RunOnTpuHost(tf.nn.ctc_beam_search_decoder, tok_logits,
    #                                       py_utils.LengthsFromBitMask(
    #                                         output_batch.padding, 0),
    #                                       beam_width=100)
    # (decoded,), _ = tf.nn.ctc_beam_search_decoder(tok_logits,
    #                                               py_utils.LengthsFromBitMask(
    #                                                   output_batch.padding, 0),
    #                                               beam_width=100)
    return py_utils.RunOnTpuHost(cpu_tf_graph_decode,
                                 tok_logits, output_batch.padding,
                                 self.params.blank_index, self.input_generator)

  def _CalculateErrorRates(self, dec_outs_dict, input_batch):
    # return {'stuff': dec_outs_dict.sparse_ids.values}
    gt_seq_lens = py_utils.LengthsFromBitMask(input_batch.tgt.paddings, 1)
    gt_transcripts = py_utils.RunOnTpuHost(self.input_generator.IdsToStrings,
                                           input_batch.tgt.labels,
                                           gt_seq_lens)

    # token error rate
    char_dist = tf.edit_distance(tf.string_split(dec_outs_dict.transcripts,
                                                 sep=''),
                                 tf.string_split(gt_transcripts, sep=''),
                                 normalize=False)

    ref_chars = tf.strings.length(gt_transcripts)
    num_wrong_chars = tf.reduce_sum(char_dist)
    num_ref_chars = tf.cast(tf.reduce_sum(ref_chars), tf.float32)
    cer = num_wrong_chars / num_ref_chars

    # word error rate
    word_dist = decoder_utils.ComputeWer(dec_outs_dict.transcripts,
                                         gt_transcripts)  # (B, 2)
    num_wrong_words = tf.reduce_sum(word_dist[:, 0])
    num_ref_words = tf.reduce_sum(word_dist[:, 1])
    wer = num_wrong_words / num_ref_words
    ret_dict = {
        'target_ids': input_batch.tgt.ids,
        'target_labels': input_batch.tgt.labels,
        'target_weights': input_batch.tgt.weights,
        'target_paddings': input_batch.tgt.paddings,
        'target_transcripts': gt_transcripts,
        'decoded_transcripts': dec_outs_dict.transcripts,
        'wer': wer,
        'cer': cer,
        'num_wrong_words': num_wrong_words,
        'num_ref_words': num_ref_words,
        'num_wrong_chars': num_wrong_chars,
        'num_ref_chars': num_ref_chars
    }
    if not py_utils.use_tpu():
      ret_dict['utt_id'] = input_batch.sample_ids

    return ret_dict

def cpu_tf_graph_decode(tok_logits, padding, blank_index, input_generator):
  # (T, B, F)
  # ctc_beam_search_decoder assumes blank_index=0
  assert blank_index == 31

  # TODO: Consider making beam_width larger
  (decoded,), _ = tf.nn.ctc_beam_search_decoder(tok_logits,
                                                py_utils.LengthsFromBitMask(padding, 0),
                                                beam_width=100)
  # Could easily use blank_index here as well, right?
  invalid = tf.constant(-1, tf.int64)
  dense_dec = tf.sparse_to_dense(decoded.indices,
                                 decoded.dense_shape,
                                 decoded.values,
                                 default_value=invalid)

  batch_segments = decoded.indices[:, 0]
  times_in_each_batch = decoded.indices[:, 1]

  decoded_seq_lengths = tf.cast(tf.math.segment_max(times_in_each_batch, batch_segments) + 1, tf.int32)
  # What happens if an empty sequence is output??? Then pad appropriately, tada!
  decoded_seq_lengths = py_utils.PadBatchDimension(decoded_seq_lengths, tf.shape(tok_logits)[1], 0)

  hyp_str = py_utils.RunOnTpuHost(input_generator.IdsToStrings,
                                  tf.cast(dense_dec, tf.int32),
                                  decoded_seq_lengths)

  hyp_str = tf.strings.regex_replace(hyp_str, '(<unk>)+', '')
  hyp_str = tf.strings.regex_replace(hyp_str, '(<s>)+', '')
  hyp_str = tf.strings.regex_replace(hyp_str, '(</s>)+', '')
  return py_utils.NestedMap(sparse_ids=decoded, transcripts=hyp_str)
