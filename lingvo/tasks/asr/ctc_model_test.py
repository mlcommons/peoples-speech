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
"""Tests for Asr Model."""

import lingvo.compat as tf
from lingvo.core import base_layer
from lingvo.core import cluster_factory
from lingvo.core import py_utils
from lingvo.core import schedule
from lingvo.core import summary_utils
from lingvo.core import test_helper
from lingvo.core import test_utils
from lingvo.tasks.asr import decoder
from lingvo.tasks.asr import input_generator
from lingvo.tasks.asr import ctc_model
from lingvo.tasks.asr import model_test_input_generator as tig
import numpy as np
import sys


class AsrCtcModelTest(test_utils.TestCase):
  # bazel test //lingvo/tasks/asr:ctc_model_test --test_output=all (prints on console)

  def _testParams_v1(self):
    input_shape = [12, 16, 80, 1]  # (B, T, F, 1)
    p = ctc_model.CTCModel.Params()
    p.vocab_size = 76
    p.blank_index = 73

    # Initialize encoder params.
    ep = p.encoder
    ep.use_specaugment = True
    # Data consists 240 dimensional frames (80 x 3 frames), which we
    # re-interpret as individual 80 dimensional frames. See also,
    # LibrispeechCommonAsrInputParams.
    ep.input_shape = [None, None, 240, 1]
    ep.lstm_cell_size = 128
    ep.num_lstm_layers = 5
    ep.lstm_type = 'fwd'
    ep.cnn_tpl.params_init = py_utils.WeightInit.Gaussian(0.001)
    # Disable conv & conv LSTM layers.
    ep.project_lstm_output = False
    ep.num_cnn_layers = 0
    ep.conv_filter_shapes = []
    ep.conv_filter_strides = []
    ep.num_conv_lstm_layers = 0

    epd = ep.lstm_dropout
    epd.keep_prob = 0.8

    sp = p.input_stacking_tpl
    sp.left_context = 1
    sp.right_context = 1
    sp.stride = 3  # L + 1 + R

    #p.decoder.target_seq_len = 5
    #p.encoder.input_shape = input_shape
    p.input = tig.TestInputGenerator.Params()
    p.input.target_max_length = 5
    p.input.source_shape = input_shape
    p.input.target_shape = [12, 5]
    p.name = 'test_ctc_mdl'
    return p

  def _testParams_v2(self):
    input_shape = [12, 16, 80, 1]  # (B, T, F, 1)
    p = ctc_model.CTCModel.Params()
    p.vocab_size = 76
    p.blank_index = 73

    # Initialize encoder params.
    p.encoder = None
    ep = p.encoder_v2
    ep.use_specaugment = True

    p.input_stacking_tpl = None
    ep.conv_subsampler = None
    # ep.stacking_subsampler = None
    esp = ep.stacking_subsampler.stacking
    esp.left_context = 1
    esp.right_context = 1
    esp.stride = 3  # L + 1 + R

    elp = ep.lstm_block
    elp.input_feats = 240
    elp.lstm_cell_size = 128
    elp.num_lstm_layers = 5
    elp.lstm_type = 'fwd'
    elp.dropout.keep_prob = 0.8

    p.input = tig.TestInputGenerator.Params()
    p.input.target_max_length = 5
    p.input.source_shape = input_shape
    p.input.target_shape = [12, 5]
    p.name = 'test_ctc_mdl_v2'
    return p

  def _testParams_conv_sub(self):
    input_shape = [12, 16, 80, 1]  # (B, T, F, 1)
    p = ctc_model.CTCModel.Params()
    p.vocab_size = 2560  # simulate wpm
    p.blank_index = 73

    # Initialize encoder params.
    p.encoder = None
    ep = p.encoder_v2
    ep.use_specaugment = True

    p.input_stacking_tpl = None
    ep.stacking_subsampler = None
    sub = ep.conv_subsampler
    sub.input_shape = input_shape

    elp = ep.lstm_block
    elp.input_feats = None # leave as none to infer automatically
    elp.lstm_cell_size = 128
    elp.num_lstm_layers = 5
    elp.lstm_type = 'fwd'
    elp.dropout.keep_prob = 0.8

    p.input = tig.TestInputGenerator.Params()
    p.input.target_max_length = 5
    p.input.source_shape = input_shape
    p.input.target_shape = [12, 5]
    p.name = 'test_ctc_mdl_conv_sub'
    return p

  # def _testParams_conv_conformer(self):
  #   input_shape = [12, 16, 80, 1]  # (B, T, F, 1)
  #   p = ctc_model.CTCModel.Params()
  #   p.vocab_size = 76
  #   p.blank_index = 73

  #   # Initialize encoder params.
  #   p.encoder = None
  #   ep = p.encoder_v2
  #   ep.use_specaugment = True

  #   p.input_stacking_tpl = None
  #   ep.stacking_subsampler = None
  #   ep.conv_subsampler.input_shape = input_shape

  #   ep.lstm_block = None
  #   ecp = ep.conformer_block
  #   ecp.num_conformer_blocks = 5
  #   ecp.name = 'conformer_layer'

  #   p.input = tig.TestInputGenerator.Params()
  #   p.input.target_max_length = 5
  #   p.input.source_shape = input_shape
  #   p.input.target_shape = [12, 5]
  #   p.name = 'test_ctc_mdl_conv_conformer'

  #   return p

  def testFProp_v1(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(93820985)
      p = self._testParams_v1()
      mdl = p.Instantiate()

      # FPropDefaultTheta -> FPropTower -> { ComputePredictions ; ComputeLoss; }
      metrics, per_item_metrics = mdl.FPropDefaultTheta()
      self.evaluate(tf.global_variables_initializer())

      ctc, _ = metrics['loss']
      test_utils.CompareToGoldenSingleFloat(self, 21.02584, ctc.eval())

  def testFProp_v2(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(93820985)
      p = self._testParams_v2()
      mdl = p.Instantiate()

      # FPropDefaultTheta -> FPropTower -> { ComputePredictions ; ComputeLoss; }
      metrics, per_item_metrics = mdl.FPropDefaultTheta()
      self.evaluate(tf.global_variables_initializer())

      ctc, _ = metrics['loss']
      test_utils.CompareToGoldenSingleFloat(self, 21.02584, ctc.eval())

  def testFProp_conv_sub(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(93820985)
      p = self._testParams_conv_sub()
      mdl = p.Instantiate()

      # FPropDefaultTheta -> FPropTower -> { ComputePredictions ; ComputeLoss; }
      metrics, per_item_metrics = mdl.FPropDefaultTheta()
      self.evaluate(tf.global_variables_initializer())

      ctc, _ = metrics['loss']
      test_utils.CompareToGoldenSingleFloat(self, 92.83023, ctc.eval())

  # def testFProp_conv_conformer(self):
  #   with self.session(use_gpu=False):
  #     tf.random.set_seed(93820985)
  #     p = self._testParams_conv_conformer()
  #     mdl = p.Instantiate()

  #     # FPropDefaultTheta -> FPropTower -> { ComputePredictions ; ComputeLoss; }
  #     metrics, per_item_metrics = mdl.FPropDefaultTheta()
  #     self.evaluate(tf.global_variables_initializer())

  #     ctc, _ = metrics['loss']
  #     test_utils.CompareToGoldenSingleFloat(self, 81.75372, ctc.eval())


if __name__ == '__main__':
  tf.test.main()
