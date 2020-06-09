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

import collections
import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import base_model
from lingvo.core import metrics
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.tasks.asr import decoder
from lingvo.tasks.asr import decoder_utils
from lingvo.tasks.asr import encoder
from lingvo.tasks.asr import frontend as asr_frontend
from lingvo.tools import audio_lib

class CTCModel(base_model.BaseTask):

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
    p.Define('lstm_tpl', rnn_cell.LSTMCellSimple.Params(),
             'Configs template for the RNN layer.')
    p.Define('proj_tpl', layers.ProjectionLayer.Params(),
             'Configs template for the projection layer.')

    # Based on ascii_tokenizer.cc
    p.Define('vocab_size', 76, 'Vocabulary size, not including the blank symbol.')
    p.Define('vocab_epsilon_index', 73, 'Index assigned to epsilon, aka blank for CTC. This should never appear in the label sequence, though. Reconsider this.')
    p.Define('input_dim', 80, '')

    p.Define('softmax_tpl', layers.SimpleFullSoftmax.Params(), '')
    p.softmax_tpl.num_classes = vocab_size

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

  @base_layer.initializer
  def __init__(self, params):
    super().__init__(params)
    p = self.params
    name = p.name

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

        if p.project_lstm_output and (i < p.num_lstm_layers - 1):
          proj_p = p.proj_tpl.Copy()
          proj_p.input_dim = p.lstm_cell_size
          proj_p.output_dim = p.lstm_cell_size
          proj_p.name = 'proj_L%d' % (i)
          params_proj_layers.append(proj_p)

        # Adds the stacking layer.
        if p.layer_index_before_stacking == i:
          stacking_layer = p.stacking_layer_tpl.Copy()
          stacking_layer.name = 'stacking_%d' % (i)
          self.CreateChild('stacking', stacking_layer)
          stacking_window_len = (
              p.stacking_layer_tpl.left_context + 1 +
              p.stacking_layer_tpl.right_context)
          output_dim *= stacking_window_len

      self.CreateChildren('rnn', params_rnn_layers)
      self.CreateChildren('proj', params_proj_layers)

      # softmax_p = p.softmax_tpl.Copy()
      # softmax_p.name = 'softmax'
      # self.CreateChild('softmax', softmax_p)

  def ComputePredictions(self, theta, input_batch):
    encoder_outputs = self._FProp(theta, input_batch)
    return tf.nn.ctc_greedy_decoder(encoder_outputs, last_one_value_in_each_sample(input_batch.paddings))

  def ComputeLoss(self, theta, predictions, input_batch):
    encoder_outputs = self._Fprop(theta, input_batch)
    # See ascii_tokenizer.cc for 73
    return tf.nn.ctc_loss(input_batch.tgt, encoder_outputs, input_batch.tgt_length, encoder_outputs_padding, blank_index=73)

  def _FProp(self, theta, input_batch, state0=None):
    p = self.params
    inputs, rnn_padding = input_batch.src_inputs, batch.paddings
    outputs = py_utils.NestedMap()
    with tf.name_scope(p.name):
      for i in range(p.num_lstm_layers):
        rnn_out = self.rnn[i].FProp(theta.rnn[i], rnn_in, rnn_padding)
        if p.project_lstm_output and (i < p.num_lstm_layers - 1):
          # Projection layers.
          rnn_out = self.proj[i].FProp(theta.proj[i], rnn_out, rnn_padding)
        if i == p.num_lstm_layers - 1:
          rnn_out *= (1.0 - rnn_padding)
        if p.layer_index_before_stacking == i:
          # Stacking layer expects input tensor shape as [batch, time, feature].
          # So transpose the tensors before and after the layer.
          # Ugh, I hate transposes like this!
          rnn_out, rnn_padding = self.stacking.FProp(
              tf.transpose(rnn_out, [1, 0, 2]),
              tf.transpose(rnn_padding, [1, 0, 2]))
          rnn_out = tf.transpose(rnn_out, [1, 0, 2])
          rnn_padding = tf.transpose(rnn_padding, [1, 0, 2])

      outputs['encoder_outputs'] = rnn_out
      outputs['encoder_outputs_padding'] = rnn_padding
      return outputs
