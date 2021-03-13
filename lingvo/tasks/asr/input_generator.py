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
"""Speech recognition input generator."""

import lingvo.compat as tf
from lingvo.core import base_input_generator
from lingvo.core import generic_input
from lingvo.core import py_utils

from tensorflow.python.ops import inplace_ops  # pylint:disable=g-direct-tensorflow-import


class AsrInput(base_input_generator.BaseSequenceInputGenerator):
  """Input generator for ASR."""

  @classmethod
  def Params(cls):
    """Defaults params for AsrInput."""
    p = super().Params()
    p.Define('frame_size', 40, 'The number of coefficients in each frame.')
    p.Define('append_eos_frame', True, 'Append an all-zero frame.')
    p.source_max_length = 3000
    return p

  def _DataSourceFromFilePattern(self, file_pattern):

    def Proc(record):
      """Parses a serialized tf.Example record."""
      # There we go! string, string, float32. I hope frames is allowed
      # to be a waveform directly...
      features = [
          ('uttid', tf.io.VarLenFeature(tf.string)),
          ('transcript', tf.io.VarLenFeature(tf.string)),
          ('frames', tf.io.VarLenFeature(tf.float32)),
      ]
      example = tf.io.parse_single_example(record, dict(features))
      fval = {k: v.values for k, v in example.items()}
      # Reshape the flattened vector into its original time-major
      # representation.
      fval['frames'] = tf.reshape(
          fval['frames'], shape=[-1, self.params.frame_size])
      # Input duration determines the bucket.
      bucket_key = tf.cast(tf.shape(fval['frames'])[0], tf.int32)
      if self.params.append_eos_frame:
        bucket_key += 1
      tgt_ids, tgt_labels, tgt_paddings = self.StringsToIds(fval['transcript'])
      src_paddings = tf.zeros([tf.shape(fval['frames'])[0]], dtype=tf.float32)
      return [
          fval['uttid'], tgt_ids, tgt_labels, tgt_paddings, fval['frames'],
          src_paddings
      ], bucket_key

    return generic_input.GenericInput(
        file_pattern=file_pattern,
        processor=Proc,
        dynamic_padding_dimensions=[0] * 6,
        dynamic_padding_constants=[0] * 5 + [1],
        **self.CommonInputOpArgs())

  def _MaybePadSourceInputs(self, src_inputs, src_paddings):
    p = self.params
    if not p.append_eos_frame:
      return src_inputs, src_paddings

    per_src_len = tf.reduce_sum(1 - src_paddings, 1)
    per_src_len += 1
    max_src_len = tf.reduce_max(per_src_len)
    input_shape = tf.shape(src_inputs)
    input_len = tf.maximum(input_shape[1], tf.cast(max_src_len, tf.int32))
    pad_steps = input_len - input_shape[1]
    src_inputs = tf.concat([
        src_inputs,
        tf.zeros(
            inplace_ops.inplace_update(input_shape, 1, pad_steps),
            src_inputs.dtype)
    ], 1)
    src_paddings = 1 - tf.sequence_mask(
        tf.reshape(per_src_len, [input_shape[0]]), tf.reshape(input_len, []),
        src_paddings.dtype)
    return src_inputs, src_paddings

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    (utt_ids, tgt_ids, tgt_labels, tgt_paddings, src_frames,
     src_paddings), self._bucket_keys = self._BuildDataSource()

    self._sample_ids = utt_ids

    src_frames, src_paddings = self._MaybePadSourceInputs(
        src_frames, src_paddings)

    # We expect src_inputs to be of shape
    # [batch_size, num_frames, feature_dim, channels].
    src_frames = tf.expand_dims(src_frames, axis=-1)

    # Convert target ids, labels, paddings, and weights from shape [batch_size,
    # 1, num_frames] to [batch_size, num_frames]
    tgt_ids = tf.squeeze(tgt_ids, axis=1)
    tgt_labels = tf.squeeze(tgt_labels, axis=1)
    tgt_paddings = tf.squeeze(tgt_paddings, axis=1)

    if p.pad_to_max_seq_length:
      assert p.source_max_length
      assert p.target_max_length

      if all(x == p.bucket_batch_limit[0] for x in p.bucket_batch_limit):
        # Set the input batch size as an int rather than a tensor.
        src_frames_shape = (self.InfeedBatchSize(), p.source_max_length,
                            p.frame_size, 1)
        src_paddings_shape = (self.InfeedBatchSize(), p.source_max_length)
        tgt_shape = (self.InfeedBatchSize(), p.target_max_length)
      else:
        tf.logging.warning(
            'Could not set static input shape since not all bucket batch sizes '
            'are the same:', p.bucket_batch_limit)
        src_frames_shape = None
        src_paddings_shape = None
        tgt_shape = None

      src_frames = py_utils.PadBatchDimension(src_frames, self.InfeedBatchSize(), 0)
      src_paddings = py_utils.PadBatchDimension(src_paddings, self.InfeedBatchSize(),
                                                1)
      tgt_ids = py_utils.PadBatchDimension(tgt_ids, self.InfeedBatchSize(), 0)
      tgt_labels = py_utils.PadBatchDimension(tgt_labels, self.InfeedBatchSize(), 0)
      tgt_paddings = py_utils.PadBatchDimension(tgt_paddings, self.InfeedBatchSize(),
                                                1)

      src_frames = py_utils.PadSequenceDimension(
          src_frames, p.source_max_length, 0, shape=src_frames_shape)
      src_paddings = py_utils.PadSequenceDimension(
          src_paddings, p.source_max_length, 1, shape=src_paddings_shape)
      tgt_ids = py_utils.PadSequenceDimension(
          tgt_ids, p.target_max_length, 0, shape=tgt_shape)
      tgt_labels = py_utils.PadSequenceDimension(
          tgt_labels, p.target_max_length, 0, shape=tgt_shape)
      tgt_paddings = py_utils.PadSequenceDimension(
          tgt_paddings, p.target_max_length, 1, shape=tgt_shape)

    tgt = py_utils.NestedMap(
        ids=tgt_ids,
        labels=tgt_labels,
        paddings=tgt_paddings,
        weights=1.0 - tgt_paddings)
    src = py_utils.NestedMap(src_inputs=src_frames, paddings=src_paddings)

    self._tgt = tgt
    self._src = src

  def _InputBatch(self):
    batch = py_utils.NestedMap()
    batch.bucket_keys = self._bucket_keys
    batch.src = self._src
    batch.tgt = self._tgt
    if not py_utils.use_tpu():
      batch.sample_ids = self._sample_ids
    return batch

class RawAsrInputIntegerUttIds(base_input_generator.BaseSequenceInputGenerator):
  """Input generator for ASR. Assumes uttid is tf.int64"""

  PAD_INDEX = -1

  @classmethod
  def Params(cls):
    """Defaults params for AsrInput."""
    p = super().Params()
    p.Define('frame_size', 1, 'The number of coefficients in each frame.')
    p.Define('append_eos_frame', True, 'Append an all-zero frame.')
    # source_max_length is way too small by default when the length's time unit
    # is in samples, given that we sample at 16kHz.
    p.source_max_length = 3000
    return p

  def _DataSourceFromFilePattern(self, file_pattern):

    def Proc(record):
      """Parses a serialized tf.Example record."""
      # There we go! string, string, float32. I hope frames is allowed
      # to be a waveform directly...
      features = [
          ('int64_uttid', tf.io.VarLenFeature(tf.int64)),
          ('int64_audio_document_id', tf.io.VarLenFeature(tf.int64)),
          ('num_utterances_in_audio_document', tf.io.VarLenFeature(tf.int64)),
          ('transcript', tf.io.VarLenFeature(tf.string)),
          ('frames', tf.io.VarLenFeature(tf.float32)),
      ]
      example = tf.io.parse_single_example(record, dict(features))
      fval = {k: v.values for k, v in example.items()}
      # Reshape the flattened vector into its original time-major
      # representation.
      fval['frames'] = tf.reshape(
          fval['frames'], shape=[-1, self.params.frame_size])
      # Input duration determines the bucket.
      bucket_key = tf.cast(tf.shape(fval['frames'])[0], tf.int32)
      if self.params.append_eos_frame:
        bucket_key += 1
      tgt_ids, tgt_labels, tgt_paddings = self.StringsToIds(fval['transcript'])
      src_paddings = tf.zeros([tf.shape(fval['frames'])[0]], dtype=tf.float32)
      return [
          fval['int64_uttid'], fval['int64_audio_document_id'],
          fval['num_utterances_in_audio_document'], tgt_ids,
          tgt_labels, tgt_paddings, fval['frames'],
          src_paddings
      ], bucket_key

    return generic_input.GenericInput(
        file_pattern=file_pattern,
        processor=Proc,
        dynamic_padding_dimensions=[0] * 8,
        dynamic_padding_constants=[0] * 7 + [1],
        **self.CommonInputOpArgs())

  def _MaybePadSourceInputs(self, src_inputs, src_paddings):
    p = self.params
    if not p.append_eos_frame:
      return src_inputs, src_paddings

    per_src_len = tf.reduce_sum(1 - src_paddings, 1)
    per_src_len += 1
    max_src_len = tf.reduce_max(per_src_len)
    input_shape = tf.shape(src_inputs)
    input_len = tf.maximum(input_shape[1], tf.cast(max_src_len, tf.int32))
    pad_steps = input_len - input_shape[1]
    src_inputs = tf.concat([
        src_inputs,
        tf.zeros(
            inplace_ops.inplace_update(input_shape, 1, pad_steps),
            src_inputs.dtype)
    ], 1)
    src_paddings = 1 - tf.sequence_mask(
        tf.reshape(per_src_len, [input_shape[0]]), tf.reshape(input_len, []),
        src_paddings.dtype)
    return src_inputs, src_paddings

  def __init__(self, params):
    super().__init__(params)
    p = self.params

    (utt_ids, audio_document_ids, num_utterances_in_audio_document,
     tgt_ids, tgt_labels, tgt_paddings, src_frames,
     src_paddings), self._bucket_keys = self._BuildDataSource()

    self._sample_ids = utt_ids

    src_frames, src_paddings = self._MaybePadSourceInputs(
        src_frames, src_paddings)

    # We expect src_inputs to be of shape
    # [batch_size, num_frames, feature_dim, channels].
    src_frames = tf.expand_dims(src_frames, axis=-1)

    # Convert target ids, labels, paddings, and weights from shape [batch_size,
    # 1, num_frames] to [batch_size, num_frames]
    tgt_ids = tf.squeeze(tgt_ids, axis=1)
    tgt_labels = tf.squeeze(tgt_labels, axis=1)
    tgt_paddings = tf.squeeze(tgt_paddings, axis=1)

    if p.pad_to_max_seq_length:
      assert p.source_max_length
      assert p.target_max_length

      if all(x == p.bucket_batch_limit[0] for x in p.bucket_batch_limit):
        # Set the input batch size as an int rather than a tensor.
        src_frames_shape = (self.InfeedBatchSize(), p.source_max_length,
                            p.frame_size, 1)
        src_paddings_shape = (self.InfeedBatchSize(), p.source_max_length)
        tgt_shape = (self.InfeedBatchSize(), p.target_max_length)
      else:
        tf.logging.warning(
            'Could not set static input shape since not all bucket batch sizes '
            'are the same:', p.bucket_batch_limit)
        src_frames_shape = None
        src_paddings_shape = None
        tgt_shape = None

      src_frames = py_utils.PadBatchDimension(src_frames, self.InfeedBatchSize(), 0)
      src_paddings = py_utils.PadBatchDimension(src_paddings, self.InfeedBatchSize(),
                                                1)
      tgt_ids = py_utils.PadBatchDimension(tgt_ids, self.InfeedBatchSize(), 0)
      tgt_labels = py_utils.PadBatchDimension(tgt_labels, self.InfeedBatchSize(), 0)
      tgt_paddings = py_utils.PadBatchDimension(tgt_paddings, self.InfeedBatchSize(),
                                                1)
      self._sample_ids = py_utils.PadBatchDimension(self._sample_ids, self.InfeedBatchSize(),
                                                    type(self).PAD_INDEX)
      # For reasons I don't understand, the shape of self._sample_ids after the above is
      # [BatchSize, 1] rather than [BatchSize].
      self._sample_ids = tf.squeeze(self._sample_ids, axis=1)
      self._sample_ids = tf.ensure_shape(self._sample_ids, self.InfeedBatchSize())

      audio_document_ids = py_utils.PadBatchDimension(audio_document_ids, self.InfeedBatchSize(),
                                                    type(self).PAD_INDEX)
      # For reasons I don't understand, the shape of audio_document_ids after the above is
      # [BatchSize, 1] rather than [BatchSize].
      audio_document_ids = tf.squeeze(audio_document_ids, axis=1)
      audio_document_ids = tf.ensure_shape(audio_document_ids, self.InfeedBatchSize())

      num_utterances_in_audio_document = py_utils.PadBatchDimension(num_utterances_in_audio_document, self.InfeedBatchSize(),
                                                    type(self).PAD_INDEX)
      # For reasons I don't understand, the shape of num_utterances_in_audio_document after the above is
      # [BatchSize, 1] rather than [BatchSize].
      num_utterances_in_audio_document = tf.squeeze(num_utterances_in_audio_document, axis=1)
      num_utterances_in_audio_document = tf.ensure_shape(num_utterances_in_audio_document, self.InfeedBatchSize())

      
      src_frames = py_utils.PadSequenceDimension(
          src_frames, p.source_max_length, 0, shape=src_frames_shape)
      src_paddings = py_utils.PadSequenceDimension(
          src_paddings, p.source_max_length, 1, shape=src_paddings_shape)
      tgt_ids = py_utils.PadSequenceDimension(
          tgt_ids, p.target_max_length, 0, shape=tgt_shape)
      tgt_labels = py_utils.PadSequenceDimension(
          tgt_labels, p.target_max_length, 0, shape=tgt_shape)
      tgt_paddings = py_utils.PadSequenceDimension(
          tgt_paddings, p.target_max_length, 1, shape=tgt_shape)

    tgt = py_utils.NestedMap(
        ids=tgt_ids,
        labels=tgt_labels,
        paddings=tgt_paddings,
        weights=1.0 - tgt_paddings)
    src = py_utils.NestedMap(src_inputs=src_frames, paddings=src_paddings)

    self._tgt = tgt
    self._src = src

    self._audio_document_ids = audio_document_ids
    self._num_utterances_in_audio_document = num_utterances_in_audio_document

  def _InputBatch(self):
    batch = py_utils.NestedMap()
    batch.bucket_keys = self._bucket_keys
    batch.src = self._src
    batch.tgt = self._tgt
    batch.sample_ids = self._sample_ids
    batch.audio_document_ids = self._audio_document_ids
    batch.num_utterances_in_audio_document = self._num_utterances_in_audio_document
    return batch
