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

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import py_utils
from lingvo.core import spectrum_augmenter
from lingvo.tasks.asr import blocks


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
    p.Define('use_conv_subsampler', False, 'Enable p.conv_subsampler')
    p.Define('use_stacking_subsampler', False, 'Enable p.stacking_subsampler')

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

    assert not (p.use_conv_subsampler and p.use_stacking_subsampler), \
        'Please use only one form of time subsampling'

    if p.use_conv_subsampler:
      self.CreateChild('sub', p.conv_subsampler.Copy())
    else:
      assert p.use_stacking_subsampler, 'Need one stacking module'
      self.CreateChild('sub', p.stacking_subsampler.Copy())
    stack_out_feats = self.sub.output_dim

    ##### handle encoding #####
    if p.lstm_block is not None:
      if p.lstm_block.input_feats is None:
        p.lstm_block.input_feats = stack_out_feats

      assert p.lstm_block.input_feats == stack_out_feats
      self.CreateChildren('enc', p.lstm_block.Copy())

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

      inputs, paddings = self.sub.FProp(theta.sub, inputs, paddings)

      encoded, padding = self.enc.FProp(theta.enc, inputs, paddings)

    return py_utils.NestedMap(encoded=encoded,
                              padding=padding,
                              state=py_utils.NestedMap())
