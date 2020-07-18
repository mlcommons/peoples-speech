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
"""Prediction network for the RNNT model."""
from lingvo.core import base_layer
from lingvo.core import layers


class Prediction(base_layer.BaseLayer):

  @classmethod
  def Params(cls):
    p = super().Params()
    # This is input to an embedding layer
    p.Define('input_shape', [None, None], 'Shape of the input. This should be a TensorShape with rank 2.')
    p.Define('lstm_cell_size', 2048, 'LSTM cell size for the RNN layer.')
    p.Define('num_lstm_layers', 8, 'Number of rnn layers to create.')
    p.Define('project_lstm_output', True,
             'Include projection layer after each encoder LSTM layer.')
    # How to use?
    StackedFRNNLayerByLayer.Params()
    p.Define('lstm_tpl', rnn_cell.LSTMCellSimple.Params(),
             'Configs template for the RNN layer.')
    p.Define('proj_tpl', layers.ProjectionLayer.Params(),
             'Configs template for the projection layer.')
    p.Define('stacking_layer_tpl', layers.StackingOverTime.Params(),
             'Configs template for the stacking layer over time.')
    p.Define('embedding_tpl', layers.EmbeddingLayer.Params(),
             'Configs template for the embedding layer.')
    p.Define(
        'layer_index_before_stacking', -1,
        'The (0-based) index of the lstm layer after which the stacking layer '
        'will be inserted. Negative value means no stacking layer will be '
        'used.')

    return p

  @base_layer.initializer
  def __init__(self, params):
    super().__init__(params)
    p = self.params
    name = p.name


    (self._first_lstm_input_dim,
     self._first_lstm_input_dim_pad) = self.FirstLstmLayerInputDimAndPadding(
       p.embedding_tpl.embedding_dim, pad_to_multiple=16)


    with tf.variable_scope(name):
      params_rnn_layers = []
      params_proj_layers = []

      embed_p = p.embedding_tpl.Copy()
      embed_p.name = 'embed'
      self.CreateChild('embed', embed_p)

      output_dim = self._first_lstm_input_dim
      for i in range(p.num_lstm_layers):
        input_dim = output_dim
        lstm_p = p.lstm_tpl.Copy()
        lstm_p.name = f"lstm_L{i}"
        lstm_p.num_input_nodes = input_dim
        lstm_p.num_output_nodes = p.lstm_cell_size
        params_rnn_layers.append(lstm_p)

        if p.project_lstm_output and (i < p.num_lstm_layers - 1):
          proj_p = p.proj_tpl.Copy()
          proj_p.input_dim = p.lstm_cell_size
          proj_p.output_dim = p.lstm_cell_size
          proj_p.name = f'proj_L{i}'
          params_proj_layers.append(proj_p)

        if p.layer_index_before_stacking == i:
          stacking_layer = p.stacking_layer_tpl.Copy()
          stacking_layer.name = f'stacking_L{i}'
          self.CreateChild('stacking', stacking_layer)
          stacking_window_len = (
              p.stacking_layer_tpl.left_context + 1 +
              p.stacking_layer_tpl.right_context)
          output_dim *= stacking_window_len
          
      self.CreateChildren('rnn', params_rnn_layers)
      self.CreateChildren('proj', params_proj_layers)

  @property
  def _use_functional(self):
    return True

  @property
  def input_shape(self):
    return self.params.input_shape

  @property
  def supports_streaming(self):
    return True

  def FirstLstmLayerInputDimAndPadding(self,
                                       lstm_input_shape,
                                       pad_to_multiple=16):
    # Makes sure the lstm input dims is multiple of 16 (alignment
    # requirement from FRNN).
    first_lstm_input_dim_unpadded = lstm_input_shape[2]

    if self._use_functional and (first_lstm_input_dim_unpadded % pad_to_multiple
                                 != 0):
      first_lstm_input_dim = int(
          (first_lstm_input_dim_unpadded + pad_to_multiple - 1) /
          pad_to_multiple) * pad_to_multiple
    else:
      first_lstm_input_dim = first_lstm_input_dim_unpadded

    first_lstm_input_dim_padding = (
        first_lstm_input_dim - first_lstm_input_dim_unpadded)
    return first_lstm_input_dim, first_lstm_input_dim_padding

  def FProp(self, theta, batch, state0=None):
    """Encodes source as represented by 'inputs' and 'paddings'.

    Args:
      theta: A NestedMap object containing weights' values of this
        layer and its children layers.
      batch: A NestedMap with fields:

        - tokens - The inputs tensor. It is expected to be of shape [time, batch].
        - paddings - The paddings tensor. It is expected to be of shape [time, batch].
      state0: Recurrent input state.

    Returns:
      A NestedMap containing

      - 'encoded': a feature tensor of shape [batch, time, depth]
      - 'padding': a 0/1 tensor of shape [batch, time]
      - 'state': the updated recurrent state
      - '${layer_type}_${layer_index}': The per-layer encoder output. Each one
        is a NestedMap containing 'encoded' and 'padding' similar to regular
        final outputs, except that 'encoded' from conv or conv_lstm layers are
        of shape [time, batch, depth, channels].
    """
    p = self.params
    if len(batch) == 0:
      
    if state0 is None:
      
