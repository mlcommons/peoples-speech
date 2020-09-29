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
            boundaries=[350_000, 500_000, 600_000],
            values=[1.0, 0.1, 0.01, 0.001]))
    tp.vn_start_step = 20_000
    tp.vn_std = 0.075
    tp.l2_regularizer_weight = 1e-6
    tp.clip_gradient_norm_to_value = 1.0
    tp.grad_norm_to_clip_to_zero = 100.0
    # What does this mean?
    tp.tpu_steps_per_loop = 20

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
    input_batch_src = input_batch.src
    p = self.params
    if p.frontend:
      input_batch_src = self.frontend.FProp(theta.frontend, input_batch_src)
    rnn_out = self.encoder.FProp(theta.encoder, input_batch_src)

    encoded = self.project_to_vocab_size(rnn_out.encoded)

    outputs = py_utils.NestedMap()
    outputs['encoded'] = encoded
    outputs['padding'] = rnn_out.padding
    return outputs

  def _CalculateWER(self, input_batch, output_batch):

    # swap row 0 and row 73 because decoder assumes blank is at 0,
    # however we set blank = 73
    tok_logits = output_batch.encoded  # (T, B, F)
    idxs = list(range(tok_logits.shape[-1]))
    idxs[0] = 73
    idxs[73] = 0
    tok_logits = tf.stack([tok_logits[:, :, idx] for idx in idxs], axis=-1)
    # tok_logits = tf.Print(tok_logits, [tf.shape(tok_logits), enc_seq_lengths], "TOK_LOGITS:", summarize=-1)

    (decoded,), neg_sum_logits = tf.nn.ctc_greedy_decoder(
      tok_logits,
      py_utils.LengthsFromBitMask(output_batch.padding, 0)
    )

    dec = tf.sparse_to_dense(
        decoded.indices, decoded.dense_shape, decoded.values, default_value=73
    )

    INVALID = tf.constant(73, tf.int64)
    bitMask = tf.cast(tf.math.equal(dec, INVALID), tf.float32)  # (B, T)
    # bitMask = tf.Print(bitMask, [tf.shape(bitMask), bitMask], "BITMASK:", summarize=-1)
    # return tf.reduce_sum(tf.cast(bitMask, tf.int32))

    decoded_seq_lengths = py_utils.LengthsFromBitMask(tf.transpose(bitMask), 0)
    # decoded_seq_lengths = tf.Print(decoded_seq_lengths, [decoded_seq_lengths], "SEQLEN:", summarize=-1)
    # return tf.reduce_sum(tf.cast(decoded_seq_lengths, tf.int32))

    hyp_str = self.input_generator.IdsToStrings(
        tf.cast(dec, tf.int32), decoded_seq_lengths
    )

    # Some predictions have start and stop tokens predicted, we dont want to include
    # those in WER calculation
    hyp_str = tf.strings.regex_replace(hyp_str, '(<unk>)+', '')
    hyp_str = tf.strings.regex_replace(hyp_str, '(<s>)+', '')
    hyp_str = tf.strings.regex_replace(hyp_str, '(</s>)+', '')

    transcripts = self.input_generator.IdsToStrings(
        input_batch.tgt.labels,
        tf.cast(
            tf.round(tf.reduce_sum(1.0 - input_batch.tgt.paddings, 1) - 1.0),
            tf.int32,
        ),
    )

    word_dist = decoder_utils.ComputeWer(hyp_str, transcripts)  # (B, 2)
    num_wrong_words = tf.reduce_sum(word_dist[:, 0])
    num_ref_words = tf.reduce_sum(word_dist[:, 1])
    wer = num_wrong_words / num_ref_words
    wer = tf.Print(
      wer,
      [
        wer,  
        # decoded.indices,
        # dec[1],
        # tf.shape(dec),
        hyp_str[1],  # prediction for sample 1
        transcripts[1],  # ground truth for sample 1
        # word_dist[1, 0],  # num wrong words in sample 1 prediction
        # word_dist[1, 1],  # num words in sample 1 ground truth
      ],
      "WER: ",
      summarize=-1
    )
    return wer

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
        blank_index=73,
    )

    # ctc_loss.shape = (B)
    total_loss = tf.reduce_mean(ctc_loss)
    # AG TODO: uncomment lines below for GPU/WER calc
    if py_utils.use_tpu():
      wer = py_utils.RunOnTpuHost(self._CalculateWER, input_batch, output_batch)
    else:
      wer = self._CalculateWER(input_batch, output_batch)
    # metrics = {"loss": (total_loss, 1.0), "wer": (wer, 1.0)}
    # AG TODO: uncomment line below and comment line below that for GPU/WER calc
    metrics = {"loss": (total_loss, 1.0)}
    per_sequence_loss = {"loss": ctc_loss}
    return metrics, per_sequence_loss

  def Inference(self):
    subgraphs = {}
    with tf.name_scope('inference'):
      subgraphs['default'] = self._InferenceSubgraph_Default()
    return subgraphs

  def _InferenceSubgraph_Default(self):
    p = self.params
    with tf.name_scope('default'):
      wav_bytes = tf.placeholder(dtype=tf.string, name='wav')
      frontend = self.frontend if p.frontend else None
      if not frontend:
        # No custom frontend. Instantiate the default.
        frontend_p = asr_frontend.MelAsrFrontend.Params()
        frontend = frontend_p.Instantiate()

      # Decode the wave bytes and use the explicit frontend.
      unused_sample_rate, audio = audio_lib.DecodeWav(wav_bytes)
      # Doing this the "kaldi" way, not the pytorch audio way
      audio *= 32768
      # Remove channel dimension, since we have a single channel.
      audio = tf.squeeze(audio, axis=1)
      # Add batch.
      audio = tf.expand_dims(audio, axis=0)
      input_batch_src = py_utils.NestedMap(
          src_inputs=audio, paddings=tf.zeros_like(audio))
      input_batch_src = frontend.FPropDefaultTheta(input_batch_src)

      encoder_outputs = self.FPropDefaultTheta() # _FProp()
      decoder_outputs = self.decoder.BeamSearchDecode(encoder_outputs)
      topk = self._GetTopK(decoder_outputs)

      feeds = {'wav': wav_bytes}
      fetches = {
          'hypotheses': topk.decoded,
          'scores': topk.scores,
          'src_frames': input_batch_src.src_inputs,
          'encoder_frames': encoder_outputs.encoded
      }

      return fetches, feeds

  @classmethod
  def FPropMeta(cls, params, *args, **kwargs):
    raise NotImplementedError("No FPropMeta available.")
