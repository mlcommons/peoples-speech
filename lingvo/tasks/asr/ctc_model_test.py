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

  def _testParams(self):
    input_shape = [2, 16, 80, 1]
    p = ctc_model.CTCModel.Params()
    #p.decoder.target_seq_len = 5
    #p.encoder.input_shape = input_shape
    p.input = tig.TestInputGenerator.Params()
    p.input.target_max_length = 5
    p.input.source_shape = input_shape
    p.input.target_shape = [2, 5]
    p.name = 'test_ctc_mdl'
    return p

  def testFProp(self):
    with self.session(use_gpu=False):
      tf.random.set_seed(93820985)
      p = self._testParams()
      mdl = p.Instantiate()
      mdl.FPropDefaultTheta()
      self.evaluate(tf.global_variables_initializer())
      test_utils.CompareToGoldenSingleFloat(self, 50.238464, mdl.loss.eval())
      
if __name__ == '__main__':
  tf.test.main()