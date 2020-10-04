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

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import base_model
from lingvo.core import metrics
from lingvo.core import layers
from lingvo.core import model_helper
from lingvo.core import py_utils
from lingvo.core import rnn_cell
from lingvo.core import schedule
from lingvo.tasks.asr import encoder
from lingvo.tasks.asr import frontend as asr_frontend
from lingvo.tools import audio_lib
from lingvo.tasks.asr import decoder_utils
# from lingvo.tasks.asr import input_generator

class CTCModel(base_model.BaseTask):
  """
  CTC model without a language model.
  """
  BLANK_IDX = 73

  @classmethod
  def Params(cls):
    p = super().Params()
    p.encoder = encoder.AsrEncoder.Params()
    p.Define(
        'frontend', None,
        'ASR frontend to extract features from input. Defaults to no frontend '
        'which means that features are taken directly from the input.')

    p.Define('include_auxiliary_metrics', True,
             'In addition to simple WER, also computes oracle WER, SACC, TER, etc. '
             'Turning off this option will speed up the decoder job.')

    # Based on ascii_tokenizer.cc
    p.Define('vocab_size', 76, 'Vocabulary size, not including the blank symbol.')
    p.Define('vocab_epsilon_index', 73, 'Index assigned to epsilon, aka blank for CTC. '
             'This should never appear in the label sequence, though. Reconsider this.')
    p.Define('input_dim', 80, '')

    tp = p.train
    tp.lr_schedule = (
        schedule.PiecewiseConstantSchedule.Params().Set(
            boundaries=[350000, 450000, 600000],
            values=[1.0, 0.1, 0.01, 0.001]))
    tp.vn_start_step = 20_000
    tp.vn_std = 0.075
    tp.l2_regularizer_weight = 1e-6
    tp.clip_gradient_norm_to_value = 1.0
    tp.grad_norm_to_clip_to_zero = 100.0
    tp.tpu_steps_per_loop = 500

    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    if p.encoder:
      if not p.encoder.name:
        p.encoder.name = 'enc'
      self.CreateChild('encoder', p.encoder)

    if p.frontend:
      self.CreateChild('frontend', p.frontend)

    projection_p = layers.FCLayer.Params()
    projection_p.activation = 'NONE'
    projection_p.output_dim = p.vocab_size
    projection_p.input_dim = p.encoder.lstm_cell_size
    if p.encoder.lstm_type == 'bidi':
      projection_p.input_dim *= 2
    self.CreateChild('project_to_vocab_size', projection_p)

  def ComputePredictions(self, theta, input_batch):
    return self._FrontendAndEncoderFProp(theta, input_batch.src)

  def DecodeWithTheta(self, theta, input_batch):
    """Constructs the inference graph."""
    p = self.params
    with tf.name_scope('decode'), tf.name_scope(p.name):
      with tf.name_scope('encoder'):
        encoder_outputs = self._FrontendAndEncoderFProp(theta, input_batch.src)
      with tf.name_scope('decoder'):
        decoder_outs = self._GreedyDecode(encoder_outputs)

      decoder_metrics = self._CalculateErrorRates(decoder_outs, input_batch)
      return decoder_metrics

  def PostProcessDecodeOut(self, decode_out_dict, dec_metrics_dict):
    gt_transcripts = decode_out_dict['target_transcripts']
    hyp_transcripts = decode_out_dict['decoded_transcripts']
    utt_id = decode_out_dict['utt_id']

    for i in range(len(gt_transcripts)):
      tf.logging.info('utt_id : %s', utt_id[i][0])
      tf.logging.info('ref_str: %s', gt_transcripts[i])
      tf.logging.info('hyp_str: %s', hyp_transcripts[i])

    total_word_err =  decode_out_dict['num_wrong_words']
    total_ref_words = decode_out_dict['num_ref_words']
    total_char_err = decode_out_dict['num_wrong_chars']
    total_ref_chars = decode_out_dict['num_ref_chars']

    dec_metrics_dict['num_samples_in_batch'].Update(len(gt_transcripts))
    dec_metrics_dict['wer'].Update(total_word_err / max(1., total_ref_words), total_ref_words)
    dec_metrics_dict['cer'].Update(total_char_err / max(1., total_ref_chars), total_ref_chars)

  def ComputeLoss(self, theta, predictions, input_batch):
    # output_batch = self._FProp(theta, input_batch)
    # See ascii_tokenizer.cc for 73
    output_batch = predictions
    ctc_loss = tf.nn.ctc_loss(
        input_batch.tgt.labels,
        output_batch.encoded,
        py_utils.LengthsFromBitMask(input_batch.tgt.paddings, 1),
        py_utils.LengthsFromBitMask(output_batch.padding, 0),
        logits_time_major=True,
        blank_index=self.BLANK_IDX
    )

    # ctc_loss.shape = (B)
    total_loss = tf.reduce_mean(ctc_loss)
    # AG TODO: uncomment lines below for GPU/WER calc
    # if py_utils.use_tpu():
    #   err = py_utils.RunOnTpuHost(self._CalculateErrorRates, input_batch, output_batch)
    # else:
    #   err = self._CalculateErrorRates(input_batch, output_batch)
    metrics = dict(loss=(total_loss, 1.0))
    per_sequence_loss = {"loss": ctc_loss}
    return metrics, per_sequence_loss

  def CreateDecoderMetrics(self):
    base_metrics = {
        "wer": metrics.AverageMetric(),
        "cer": metrics.AverageMetric(),
        "num_samples_in_batch": metrics.AverageMetric()
    }
    return base_metrics

  @classmethod
  def FPropMeta(cls, params, *args, **kwargs):
    raise NotImplementedError("No FPropMeta available.")

  def _FrontendAndEncoderFProp(self, theta, input_batch_src):
    p = self.params
    if p.frontend:
      input_batch_src = self.frontend.FProp(theta.frontend, input_batch_src)
    rnn_out = self.encoder.FProp(theta.encoder, input_batch_src)
    encoded = self.project_to_vocab_size(rnn_out.encoded)
    outputs = py_utils.NestedMap(encoded=encoded, padding=rnn_out.padding)
    return outputs

  def _GreedyDecode(self, output_batch):
    # swap row 0 and row 73 because decoder assumes blank is at 0,
    # however we set blank = 73
    tok_logits = output_batch.encoded  # (T, B, F)
    idxs = list(range(tok_logits.shape[-1]))
    idxs[0] = self.BLANK_IDX
    idxs[self.BLANK_IDX] = 0
    tok_logits = tf.stack([tok_logits[:, :, idx] for idx in idxs], axis=-1)

    (decoded,), neg_sum_logits = tf.nn.ctc_greedy_decoder(
      tok_logits,
      py_utils.LengthsFromBitMask(output_batch.padding, 0)
    )

    dense_dec = tf.sparse_to_dense(
        decoded.indices, decoded.dense_shape, decoded.values, default_value=self.BLANK_IDX
    )

    INVALID = tf.constant(self.BLANK_IDX, tf.int64)
    bitMask = tf.cast(tf.math.equal(dense_dec, INVALID), tf.float32)  # (B, T)

    decoded_seq_lengths = py_utils.LengthsFromBitMask(tf.transpose(bitMask), 0)

    hyp_str = self.input_generator.IdsToStrings(
        tf.cast(dense_dec, tf.int32), decoded_seq_lengths
    )

    # Some predictions have start and stop tokens predicted, we dont want to include
    # those in WER calculation
    hyp_str = tf.strings.regex_replace(hyp_str, '(<unk>)+', '')
    hyp_str = tf.strings.regex_replace(hyp_str, '(<s>)+', '')
    hyp_str = tf.strings.regex_replace(hyp_str, '(</s>)+', '')
    return py_utils.NestedMap(sparse_ids=decoded, transcripts=hyp_str)

  def _CalculateErrorRates(self, dec_outs_dict, input_batch):
    sparse_gt_ids = tf.cast(py_utils.SequenceToSparseTensor(
        input_batch.tgt.labels, input_batch.tgt.paddings), tf.int64)

    gt_transcripts = self.input_generator.IdsToStrings(
        input_batch.tgt.labels,
        tf.cast(
            tf.round(tf.reduce_sum(1.0 - input_batch.tgt.paddings, 1) - 1.0),
            tf.int32,
        ),
    )

    # char error rate
    char_dist = tf.edit_distance(dec_outs_dict.sparse_ids, sparse_gt_ids, normalize=False)
    ref_chars =  py_utils.LengthsFromBitMask(input_batch.tgt.paddings, 1)
    num_wrong_chars = tf.reduce_sum(char_dist)
    num_ref_chars = tf.cast(tf.reduce_sum(ref_chars), tf.float32)
    cer = num_wrong_chars / num_ref_chars

    # word error rate
    word_dist = decoder_utils.ComputeWer(dec_outs_dict.transcripts, gt_transcripts)  # (B, 2)
    num_wrong_words = tf.reduce_sum(word_dist[:, 0])
    num_ref_words = tf.reduce_sum(word_dist[:, 1])
    wer = num_wrong_words / num_ref_words

    ret_dict = {
      'utt_id': input_batch.sample_ids,
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

    # if not py_utils.use_tpu():
    #   ret_dict['utt_id'] = input_batch.sample_ids

    # wer = tf.Print(wer, [wer, cer, 
    #                      num_wrong_words, num_ref_words, 
    #                      num_wrong_chars, num_ref_chars])

    # wer = tf.Print(wer, [transcripts[i] for i in range(12)], "_GT_: ")
    # wer = tf.Print(wer, [hyp_str[i] for i in range(12)], "PRED: ")
    return ret_dict