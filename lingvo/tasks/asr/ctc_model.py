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

class CTCModel(base_model.BaseTask):
  """
  CTC model without a language model.
  """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define(
        'frontend', None,
        'ASR frontend to extract features from input. Defaults to no frontend '
        'which means that features are taken directly from the input.')

    p.Define('include_auxiliary_metrics', True,
             'In addition to simple WER, also computes oracle WER, SACC, TER, etc. '
             'Turning off this option will speed up the decoder job.')
    p.Define('lstm_tpl', rnn_cell.LSTMCellSimple.Params(),
             'Configs template for the RNN layer.')
    p.Define('num_lstm_layers', 5, '')
    p.Define('lstm_cell_size', 256, 'LSTM cell size for the RNN layer.')
    # TODO: Change this back to True and figure out what the problem is.
    p.Define('project_lstm_output', False,
             'Include projection layer after each encoder LSTM layer.')
    p.Define('proj_tpl', layers.ProjectionLayer.Params(),
             'Configs template for the projection layer.')
    p.Define('input_stacking_layer_tpl', layers.StackingOverTime.Params(),
             'Configs template for the stacking layer over time of the input features')
    p.Define('stacking_layer_tpl', layers.StackingOverTime.Params(),
             'Configs template for the stacking layer over time.')
    p.Define('unidi_rnn_type', 'func', 'func is only valid option apparently')
    p.Define(
        'layer_index_before_stacking', -1,
        'The (0-based) index of the lstm layer after which the stacking layer '
        'will be inserted. Negative value means no stacking layer will be '
        'used.')

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

    p.proj_tpl.batch_norm = False
    p.proj_tpl.activation = 'RELU'

    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    name = p.name

    if p.frontend:
      self.CreateChild('frontend', p.frontend)

    self.CreateChild('input_stacking', p.input_stacking_layer_tpl.Copy())

    with tf.variable_scope(name):
      params_rnn_layers = []
      params_proj_layers = []
      output_dim = p.input_dim
      for i in range(p.num_lstm_layers):
        input_dim = output_dim
        forward_p = p.lstm_tpl.Copy()
        forward_p.name = 'fwd_rnn_cell_L%d' % (i)
        forward_p.num_input_nodes = input_dim
        forward_p.num_output_nodes = p.lstm_cell_size
        rnn_p = model_helper.CreateUnidirectionalRNNParams(p, forward_p)
        rnn_p.name = 'fwd_rnn_layer_L%d' % (i)
        params_rnn_layers.append(rnn_p)
        output_dim = p.lstm_cell_size

        # if p.project_lstm_output and (i < p.num_lstm_layers - 1):
        #   proj_p = p.proj_tpl.Copy()
        #   proj_p.input_dim = p.lstm_cell_size
        #   proj_p.output_dim = p.lstm_cell_size
        #   proj_p.name = 'proj_L%d' % (i)
        #   params_proj_layers.append(proj_p)

        # Adds the stacking layer.
        # if p.layer_index_before_stacking == i:
        #   stacking_layer = p.stacking_layer_tpl.Copy()
        #   stacking_layer.name = 'stacking_%d' % (i)
        #   self.CreateChild('stacking', stacking_layer)
        #   stacking_window_len = (
        #       p.stacking_layer_tpl.left_context + 1 +
        #       p.stacking_layer_tpl.right_context)
        #   output_dim *= stacking_window_len

      self.CreateChildren('rnn', params_rnn_layers)
      self.CreateChildren('proj', params_proj_layers)
      projection_p = layers.FCLayer.Params()
      projection_p.activation = 'NONE'
      projection_p.output_dim = p.vocab_size
      projection_p.input_dim = output_dim
      self.CreateChild('project_to_vocab_size', projection_p)

  def ComputePredictions(self, theta, input_batch):
    output_batch = self._FProp(theta, input_batch)
    # ctc_greedy_decoder = merge_repeated = True (def)
    hypotheses = py_utils.RunOnTpuHost(
      tf.nn.ctc_greedy_decoder,
      output_batch.encoder_outputs,
      py_utils.LengthsFromBitMask(
        tf.squeeze(output_batch.encoder_outputs_padding, 2), 0),
      
    )
    tf.print("*******ComputePredictions********:" , hypotheses)
    return hypotheses

  def ComputeLoss(self, theta, predictions, input_batch):
    output_batch = self._FProp(theta, input_batch)
    # See ascii_tokenizer.cc for 73
    ctc_loss = tf.nn.ctc_loss(input_batch.tgt.labels, output_batch.encoder_outputs,
                              py_utils.LengthsFromBitMask(input_batch.tgt.paddings, 1),
                              py_utils.LengthsFromBitMask(tf.squeeze(output_batch.encoder_outputs_padding, 2), 0),
                              logits_time_major=True,
                              blank_index=73)
    total_loss = tf.reduce_mean(ctc_loss)
    tf.print("*******ComputeLoss********:" , total_loss)
    metrics = {"loss": (total_loss, 1.0)}
    per_sequence_loss = {"loss": ctc_loss}
    return metrics, per_sequence_loss

  def _FProp(self, theta, input_batch, state0=None):
    p = self.params
    # This is BxTxFx1. We need TxBxF for the LSTM
    inputs = input_batch.src.src_inputs
    inputs = tf.squeeze(inputs, [-1])
    # This is BxT. We need TxBx1 for the LSTM
    rnn_padding = tf.expand_dims(input_batch.src.paddings, 2)

    # inputs: BxTxF
    # rnn_padding: BxTx1
    inputs, rnn_padding = self.input_stacking.FProp(inputs, rnn_padding)
    inputs = tf.transpose(inputs, [1, 0, 2])
    rnn_padding = tf.transpose(rnn_padding, [1, 0, 2])
    # inputs: TxBxF
    # rnn_padding: TxBx1

    rnn_out = inputs
    outputs = py_utils.NestedMap()
    with tf.name_scope(p.name):
      for i in range(p.num_lstm_layers):
        rnn_out, _ = self.rnn[i].FProp(theta.rnn[i], rnn_out, rnn_padding)
        # if p.project_lstm_output and (i < p.num_lstm_layers - 1):
        #   # Projection layers.
        #   rnn_out = self.proj[i].FProp(theta.proj[i], rnn_out, rnn_padding)
        # if p.layer_index_before_stacking == i:
        #   # Stacking layer expects input tensor shape as [batch, time, feature].
        #   # So transpose the tensors before and after the layer.
        #   # Ugh, I hate transposes like this!
        #   rnn_out, rnn_padding = self.stacking.FProp(
        #       tf.transpose(rnn_out, [1, 0, 2]),
        #       tf.transpose(rnn_padding, [1, 0, 2]))
        #   rnn_out = tf.transpose(rnn_out, [1, 0, 2])
        #   rnn_padding = tf.transpose(rnn_padding, [1, 0, 2])

      rnn_out = self.project_to_vocab_size(rnn_out)
      rnn_out *= (1.0 - rnn_padding)
      outputs['encoder_outputs'] = rnn_out
      outputs['encoder_outputs_padding'] = rnn_padding
      return outputs

  def Inference(self):
    raise NotImplementedError("No Inference available.")
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

      tf.print("*******_InferenceSubgraph_Default********:" , topk.decoded)

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
