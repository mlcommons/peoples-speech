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
from lingvo.core import test_utils
from lingvo.tasks.asr import ctc_model
from lingvo.tasks.asr import model_test_input_generator as tig


class AsrCtcModelTest(test_utils.TestCase):
  # bazel test //lingvo/tasks/asr:ctc_model_test --test_output=all

  def _testParams(self):
    input_shape = [12, 16, 80, 1]  # (B, T, F, 1)
    p = ctc_model.CTCModel.Params()
    p.vocab_size = 76
    p.blank_index = 73

    # Initialize encoder params.
    p.encoder = None
    ep = p.encoder_v2
    ep.use_specaugment = True

    ep.stacking_subsampler = None
    sub = ep.conv_subsampler
    sub.input_shape = input_shape

    elp = ep.lstm_block
    elp.input_feats = None  # leave as none to infer automatically
    elp.lstm_cell_size = 128
    elp.num_lstm_layers = 5
    elp.lstm_type = 'fwd'
    elp.dropout.keep_prob = 0.8

    # ep.lstm_block = None
    # ecp = ep.conformer_block
    # ecp.num_conformer_blocks = 5
    # ecp.name = 'conformer_layer'

    p.input = tig.TestInputGenerator.Params()
    p.input.target_max_length = 5
    p.input.source_shape = input_shape
    p.input.target_shape = [12, 5]
    p.name = 'test_ctc_mdl'
    return p

  def testFProp_conv_sub(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(93820985)
      p = self._testParams()
      mdl = p.Instantiate()

      # FPropDefaultTheta -> FPropTower -> { ComputePredictions ; ComputeLoss; }
      metrics, _ = mdl.FPropDefaultTheta()
      self.evaluate(tf.global_variables_initializer())

      ctc, _ = metrics['loss']
      test_utils.CompareToGoldenSingleFloat(self, 76.710762, ctc.eval())


if __name__ == '__main__':
  tf.test.main()
