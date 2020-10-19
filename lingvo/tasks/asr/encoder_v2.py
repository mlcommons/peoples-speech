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
"""Encoders for the speech model."""

import collections
import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import layers
from lingvo.core import model_helper
from lingvo.core import py_utils
from lingvo.core import rnn_cell
from lingvo.core import rnn_layers
from lingvo.core import spectrum_augmenter
from lingvo.core import summary_utils
from lingvo.tasks.asr import blocks

from tensorflow.python.ops import inplace_ops


class AsrEncoder(base_layer.BaseLayer):
  """Speech encoder version 2."""

  @classmethod
  def Params(cls):
    """Configs for AsrEncoder."""
    p = super().Params()

    # spec-augment
    p.Define('specaugment_network',
             spectrum_augmenter.SpectrumAugmenter.Params(),
             'Configs template for the augmentation network.')
    p.Define('use_specaugment', False, 'Use specaugmentation or not.')

    # temporal downsampling, use one of the two
    p.Define('conv_subsampler', blocks.ConvolutionalDownsampler.Params(),
             'Convolution subsampling layer params')
    p.Define('stacking_subsampler', blocks.InputStackingDownsampler.Params(),
             'Stacking subsampling layer params')

    # actual encoding layers, use one of these
    p.Define('lstm_block', blocks.LSTMBlock.Params(), 'LSTM layer params')
    # p.Define('conformer_block', blocks.ConformerBlock.Params(), 'Conformer specs')

    return p

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    ##### Use specAugment or not ####
    if p.use_specaugment:
      self.CreateChild('specaugment', p.specaugment_network.Copy())

    #####  handle sub-sampling ####
    stack_out_feats = None
    has_conv_subsampler = p.conv_subsampler is not None
    has_stacking_subsampler = p.stacking_subsampler is not None

    assert has_conv_subsampler or has_stacking_subsampler, \
        "Better have some sort of time subsampling"

    assert not (has_conv_subsampler and has_stacking_subsampler), \
        "Please use only one form of time subsampling"

    if p.conv_subsampler:
      self.CreateChild('conv_sub', p.conv_subsampler.Copy())
      stack_out_feats = self.conv_sub.output_dim

    if p.stacking_subsampler:
      self.CreateChild('stack_sub', p.stacking_subsampler.Copy())
      stack_out_feats = self.stack_sub.output_dim

    ##### handle encoding #####
    if p.lstm_block is not None:
      if p.lstm_block.input_feats is None:
        p.lstm_block.input_feats = stack_out_feats

      assert p.lstm_block.input_feats == stack_out_feats
      self.CreateChildren('enc', p.lstm_block.Copy())

    # else:
    #   assert p.conformer_block is not None
    #   if p.conformer_block.input_feats is None:
    #     p.conformer_block.input_feats = stack_out_feats

    #   assert p.conformer_block.input_feats == stack_out_feats
    #   self.CreateChildren('enc', p.conformer_block.Copy())

  @property
  def output_dim(self):
    return self.enc.output_dim

  @property
  def _use_functional(self):
    return True

  @property
  def supports_streaming(self):
    return False

  def zero_state(self, theta, batch_size):
    return py_utils.NestedMap()

  def FProp(self, theta, batch, state0=None):
    """Encodes source as represented by 'inputs' and 'paddings'.
    Args:
      theta: A NestedMap object containing weights' values of this
        layer and its children layers.
      batch: A NestedMap with fields:
        - src_inputs - The inputs tensor. It is expected to be of shape [batch,
          time, feature_dim, channels].
        - paddings - The paddings tensor. It is expected to be of shape [batch,
          time].
      state0: Recurrent input state. Not supported/ignored by this encoder.
    Returns:
      A NestedMap containing
      - 'encoded': a feature tensor of shape [time, batch, depth]
      - 'padding': a 0/1 tensor of shape [time, batch]
      - 'state': the updated recurrent state
    """
    p = self.params
    inputs, paddings = batch.src_inputs, batch.paddings

    with tf.name_scope(p.name):

      if p.use_specaugment and not self.do_eval:
        inputs, paddings = self.specaugment.FProp(theta.specaugment, inputs,
                                                  paddings)

      if p.conv_subsampler is not None:
        inputs, paddings = self.conv_sub.FProp(theta.conv_sub, inputs, paddings)

      if p.stacking_subsampler is not None:
        inputs, paddings = self.stack_sub.FProp(theta.stack_sub, inputs, paddings)

      encoded, padding = self.enc.FProp(theta.enc, inputs, paddings)

    return py_utils.NestedMap(encoded=encoded, padding=padding,
                              state=py_utils.NestedMap())
