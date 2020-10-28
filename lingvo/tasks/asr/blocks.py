# Lint as: python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Encoders for the speech model."""

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import layers
from lingvo.core import model_helper
from lingvo.core import py_utils
from lingvo.core import rnn_cell
# from lingvo.core import conformer_layer


class LSTMBlock(base_layer.BaseLayer):

  @classmethod
  def Params(cls):
    """Configs for AsrEncoder."""
    p = super().Params()
    p.Define('lstm_tpl', rnn_cell.LSTMCellSimple.Params(),
             'Configs template for the RNN layer.')
    p.Define('lstm_cell_size', 256, 'LSTM cell size for the RNN layer.')
    p.Define('input_feats', None, 'Number of features input to first LSTM.')
    p.Define('lstm_type', 'fwd', 'fwd or bidi')
    p.Define('num_lstm_layers', 3, 'Number of rnn layers to create')
    p.Define('dropout', layers.DropoutLayer.Params(),
             'Apply dropout between LSTM')
    p.Define('bidi_rnn_type', 'func',
             'Options: func. func: BidirectionalFRNN. ')
    p.Define('unidi_rnn_type', 'func',
             'Options: func. func: UnidirectionalFRNN. ')
    # Default config for the rnn layer.
    p.lstm_tpl.params_init = py_utils.WeightInit.Uniform(0.1)
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    params_rnn_layers = []
    lstm_out_size = p.lstm_cell_size

    assert p.lstm_type in {'bidi', 'fwd'}, 'Only fwd, bidi allowed'
    if p.lstm_type == 'bidi':
      lstm_out_size *= 2

    output_dim = p.input_feats
    for i in range(p.num_lstm_layers):
      input_dim = output_dim
      forward_p = p.lstm_tpl.Copy()
      forward_p.name = 'fwd_rnn_L%d' % i
      forward_p.num_input_nodes = input_dim
      forward_p.num_output_nodes = p.lstm_cell_size

      if p.lstm_type == 'bidi':
        backward_p = forward_p.Copy()
        backward_p.name = 'bak_rnn_L%d' % i
        rnn_p = self.CreateBidirectionalRNNParams(forward_p, backward_p)
        rnn_p.name = 'brnn_L%d' % i
      else:
        rnn_p = self.CreateUnidirectionalRNNParams(forward_p)
        rnn_p.name = 'rnn_L%d' % i

      params_rnn_layers.append(rnn_p)
      output_dim = lstm_out_size

    if p.dropout:
      self.CreateChildren('dropout', p.dropout)

    self.CreateChildren('rnn', params_rnn_layers)

  @property
  def output_dim(self):
    multiplier = 2 if self.params.lstm_type == 'bidi' else 1
    return self.params.lstm_cell_size * multiplier

  def CreateBidirectionalRNNParams(self, forward_p, backward_p):
    return model_helper.CreateBidirectionalRNNParams(self.params, forward_p,
                                                     backward_p)

  def CreateUnidirectionalRNNParams(self, forward_p):
    return model_helper.CreateUnidirectionalRNNParams(self.params, forward_p)

  def FProp(self, theta, inputs, paddings):
    """
     inputs: (B, T, F, 1)
     paddings: (B, T)
     outputs: encoded, padding: (T, B, F), (T, B)
    """

    p = self.params
    rnn_in = tf.transpose(tf.squeeze(inputs, axis=3), [1, 0, 2])
    rnn_padding = tf.expand_dims(tf.transpose(paddings), 2)
    # rnn_in is of shape [time, batch, depth]
    # rnn_padding is of shape [time, batch, 1]

    for i in range(p.num_lstm_layers):
      rnn_out = self.rnn[i].FProp(theta.rnn[i], rnn_in, rnn_padding)

      if p.lstm_type == 'fwd':
        rnn_out, _ = rnn_out

      if p.dropout:
        rnn_out = self.dropout.FProp(theta.dropout, rnn_out)

      rnn_in = rnn_out

    encoded = rnn_out * (1.0 - rnn_padding)
    padding = tf.squeeze(rnn_padding, [2])
    return encoded, padding


class ConvolutionalDownsampler(base_layer.BaseLayer):
  """ Use convolution with striding to achieve stacking effect """

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('cnn_tpl', layers.ConvLayer.Params(),
             'Configs template for the conv layer.')
    p.Define('num_cnn_layers', None, 'Number of CNN layers')
    p.Define('conv_filter_shapes', None, 'Filter shapes for each conv layer.')
    p.Define('conv_filter_strides', None,
             'Filter strides for each conv layer (T, F).')
    p.Define('input_shape', [None, None, 80, 1],
             'Shape of the input. This should a TensorShape with rank 4.')

    p.num_cnn_layers = 2
    p.conv_filter_shapes = [(5, 5, 1, 32), (5, 5, 32, 32)]
    p.conv_filter_strides = [(1, 2), (3, 2)]  # (T, F)
    p.cnn_tpl.params_init = py_utils.WeightInit.TruncatedGaussian(0.1)
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    assert p.num_cnn_layers == len(p.conv_filter_shapes)
    assert p.num_cnn_layers == len(p.conv_filter_strides)
    params_conv_layers = []
    for i in range(p.num_cnn_layers):
      conv_p = p.cnn_tpl.Copy()
      conv_p.name = 'conv_L%d' % i
      conv_p.filter_shape = p.conv_filter_shapes[i]
      conv_p.filter_stride = p.conv_filter_strides[i]
      params_conv_layers.append(conv_p)

    self.CreateChildren('conv', params_conv_layers)

    assert p.input_shape is not None
    conv_output_shape = p.input_shape
    for i in range(p.num_cnn_layers):
      conv_output_shape = self.conv[i].OutShape(conv_output_shape)

    assert len(conv_output_shape) == 4  # batch, height, width, channel.
    feat_dim = conv_output_shape[-1] * conv_output_shape[-2]
    self.conv_output_shape = (*conv_output_shape[:2], feat_dim, 1)

  @property
  def output_dim(self):
    return self.conv_output_shape[2]

  def FProp(self, theta, inputs, paddings):
    # BTF1 -> BTF1

    conv_out = inputs
    out_padding = paddings

    for i, conv_layer in enumerate(self.conv):
      conv_out, out_padding = conv_layer.FProp(theta.conv[i], conv_out,
                                               out_padding)

    b, t, f1, f2 = py_utils.GetShape(conv_out)
    conv_out = tf.reshape(conv_out, [b, t, f1 * f2, 1])
    return conv_out, out_padding


class InputStackingDownsampler(base_layer.BaseLayer):

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_shape', [None, None, 80, 1],
             'Shape of the input. This should a TensorShape with rank 4.')
    p.Define('stacking', layers.StackingOverTime.Params(), 'Stacking params')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params
    self.CreateChild('stacking', p.stacking.Copy())
    self.num_feats = p.input_shape[-2] * self.stacking.window_size

  @property
  def output_dim(self):
    return self.num_feats

  def FProp(self, theta, inputs, paddings):
    # BTF1 -> BTF1
    inputs = tf.squeeze(inputs, axis=3)  # BTF1 -> BTF
    paddings = tf.expand_dims(paddings, axis=2)  # BT -> BT1
    encoded, padding = self.stacking.FProp(inputs, paddings)
    encoded = tf.expand_dims(encoded, axis=3)  # BTF -> BTF1
    padding = tf.squeeze(padding, axis=2)  # BT1 -> BT
    return encoded, padding


# Lingvo's conformer models work with TF2.3 but TPUs have issues with TF2.3
# class ConformerBlock(base_layer.BaseLayer):

#   @ classmethod
#   def Params(cls):
#     p = super().Params()

#     p.Define('input_feats', None, 'Required')
#     p.Define('dropout_prob', 0., '')

#     p.Define('atten_num_heads', 4, '')
#     p.Define('atten_local_context', 3, '')
#     p.Define('kernel_size', 32, '')
#     p.Define('fflayer_activation', 'SWISH', '')
#     p.Define('layer_order', 'mhsa_before_conv', '')
#     p.Define('num_conformer_blocks', 1, 'Number of conformer layers')

#     return p

#   def __init__(self, params):
#     super().__init__(params)
#     p = self.params
#     assert p.input_feats is not None

#     conformer_tpl = conformer_layer.ConformerLayer.CommonParams(
#       input_dim=p.input_feats,
#       atten_num_heads=p.atten_num_heads,
#       atten_local_context=p.atten_local_context,
#       kernel_size=p.kernel_size,
#       fflayer_hidden_dim=p.input_feats // 2)

#     for i in range(p.num_conformer_blocks):
#       self.CreateChild(f'conformer_{i}', conformer_tpl.Copy())

#   @property
#   def output_dim(self):
#     return self.params.input_feats

#   def FProp(self, theta, input, paddings):
#     """
#      inputs: (B, T, F, 1)
#      paddings: (B, T)
#      outputs: encoded, padding: (T, B, F), (T, B)
#      same as LSTM block
#     """
#     p = self.params

#     input = tf.squeeze(input, 3)  # BTF1 -> BTF

#     for i in range(p.num_conformer_blocks):
#       conf_block = getattr(self, f'conformer_{i}')
#       conf_theta = getattr(theta, f'conformer_{i}')
#       encoded, padding = conf_block.FProp(conf_theta, input, paddings)
#       input, paddings = encoded, padding

#     encoded = tf.transpose(encoded, [1, 0, 2])
#     padding = tf.transpose(padding)
#     return encoded, padding


class VocabProjectionBlock(base_layer.BaseLayer):

  @classmethod
  def Params(cls):
    p = super().Params()
    p.Define('input_dim', None, 'Required')
    p.Define('vocab_size', None, 'Required')
    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    pp = layers.FCLayer.Params()
    pp.activation = 'NONE'
    pp.output_dim = min(p.vocab_size, 96)
    pp.input_dim = p.input_dim
    pp.params_init = py_utils.WeightInit.Uniform(0.1)
    self.CreateChild('projection', pp.Copy())

    remaining = p.vocab_size - 96
    if remaining > 0:
      rp = layers.FCLayer.Params()
      rp.activation = 'NONE'
      rp.output_dim = p.vocab_size - 96
      rp.input_dim = p.input_dim
      rp.params_init = py_utils.WeightInit.Uniform(0.01)
      self.CreateChild('lowpri_projection', rp.Copy())

  def FProp(self, theta, inputs, paddings):
    encoded = self.projection.FProp(theta.projection, inputs)

    if hasattr(self, 'lowpri_projection'):
      encoded2 = self.lowpri_projection.FProp(theta.lowpri_projection, inputs)
      encoded = tf.concat([encoded, encoded2], axis=2)

    return encoded, paddings
