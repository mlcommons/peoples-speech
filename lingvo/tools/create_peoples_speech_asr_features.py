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
# ==============================================================================
"""Encode the audio tarball contents into tfrecords."""

import os
import csv
import random
import tarfile
import lingvo.compat as tf
from lingvo.tools import audio_lib

tf.flags.DEFINE_string('input_tarball', '', 'Input .tar.gz file.')
tf.flags.DEFINE_string(
    'input_text', '', 'Reference text as csv, filename, transcript, metadata.')
tf.flags.DEFINE_string('output_template', '', 'File of tfrecords.')

tf.flags.DEFINE_bool('dump_transcripts', False,
                     'First pass through the tarball to read the transcripts.')
tf.flags.DEFINE_string('transcripts_filepath', '',
                       'Where to put the transcripts.')
tf.flags.DEFINE_bool('generate_tfrecords', False,
                     'Second pass generates the tf records')

tf.flags.DEFINE_integer('shard_id', -1, 'Processor shard.')
tf.flags.DEFINE_integer(
    'num_shards', -1,
    'Number of processor shards. Must divide num_output_shards.')
tf.flags.DEFINE_integer('output_range_begin', -1, 'Begin of output shard IDs.')
tf.flags.DEFINE_integer('output_range_end', -1, 'End of output shard IDs.')
tf.flags.DEFINE_integer('num_output_shards', -1,
                        'Total number of output shards.')

FLAGS = tf.flags.FLAGS


def _MakeBytesFeature(unicode_array):
  value = [tf.compat.as_bytes(w) for w in unicode_array]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _MakeInt64Feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _MakeFloatFeature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _MakeTfExample(uttid, frames, text):
  flat_frames = frames.flatten()
  feature = {
      'uttid': _MakeBytesFeature([uttid]),
      'transcript': _MakeBytesFeature([text.lower()]),
      'frames': _MakeFloatFeature(flat_frames)
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))


def _ReadTranscriptionsFromCSV():
  trans = {}
  with tf.io.gfile.GFile(FLAGS.input_text, 'r') as f:
    for row in csv.reader(f):
      uttid, txt, metadata = row[:3]
      # remove the gs bucket name and prefix
      uttid = '/'.join(uttid.split('/')[3:])
      trans[uttid] = txt
  return trans


def _LoadTranscriptionsFromFile():
  trans = {}
  with tf.io.gfile.GFile(FLAGS.transcripts_filepath, 'r') as f:
    for line in f.readlines():
      uttid, txt = line.strip('\n').split(' ', 1)
      trans[uttid] = txt
  return trans


def _MakeLogMelFromTensorflowBuiltin(tf_wav_bytes):
  sample_rate, audio = audio_lib.DecodeWav(tf_wav_bytes)
  static_sample_rate = 16000
  with tf.control_dependencies(
      [tf.assert_equal(sample_rate, static_sample_rate)]):
    log_mel = audio_lib.AudioToMfcc(static_sample_rate, audio, 25, 25, 40)
  return log_mel


def _OpenSubShards():
  tf.logging.info('Shards: %d to %d', FLAGS.output_range_begin,
                  FLAGS.output_range_end)
  recordio_writers = []
  for s in range(FLAGS.output_range_begin, FLAGS.output_range_end):
    filepath = FLAGS.output_template % (s, FLAGS.num_output_shards)
    tf.logging.info('Opening output shard: %s', filepath)
    recordio_writers += [tf.python_io.TFRecordWriter(filepath)]
  return recordio_writers


def _CloseSubShards(files):
  for f in files:
    f.close()


def _SelectRandomShard(files):
  subshard = random.randint(0, len(files) - 1)
  return files[subshard]


def _CreateAsrFeatures():
  # First pass: extract transcription files.
  if False:  #os.path.exists(FLAGS.transcripts_filepath):
    trans = _LoadTranscriptionsFromFile()
  else:
    tf.logging.info('Running first pass on the fly')
    trans = _ReadTranscriptionsFromCSV()
  total_utts = len(trans)
  tf.logging.info('Total transcripts: %d', len(trans))
  tf_bytes = tf.placeholder(dtype=tf.string)
  log_mel = audio_lib.ExtractLogMelFeatures(tf_bytes)
  # Second pass: transcode the flac.
  file_obj = tf.io.gfile.GFile(FLAGS.input_tarball, mode='rb')
  tar = tarfile.open(fileobj=file_obj, mode='r:gz')
  n = 0
  recordio_writers = _OpenSubShards()
  tfconf = tf.config_pb2.ConfigProto()
  tfconf.gpu_options.allow_growth = True
  with tf.Session(config=tfconf) as sess:
    for tarinfo in tar:
      # We can actually decode essentially any audio format, but we
      # want to avoid non-audio data. Thus, this condition.
      if not (tarinfo.name.endswith('.flac') or tarinfo.name.endswith('.wav') or
              tarinfo.name.endswith('.mp3')):
        continue
      n += 1
      if n % FLAGS.num_shards != FLAGS.shard_id:
        continue
      f = tar.extractfile(tarinfo)
      fmt = tarinfo.name.split('.')[-1]
      uttid = tarinfo.name
      audio_bytes = f.read()
      f.close()
      try:
        wav_bytes = audio_lib.DecodeToWav(audio_bytes, fmt)
        frames = sess.run(log_mel, feed_dict={tf_bytes: wav_bytes})
      except Exception as e:
        # raise
        trans.pop(uttid)
        tf.logging.info(f'{uttid} FAILED featurization')
        continue
      assert uttid in trans, uttid
      num_words = len(trans[uttid])
      tf.logging.info('utt[%d]: %s [%d frames, %d chars]', n, uttid,
                      frames.shape[1], num_words)
      ex = _MakeTfExample(uttid, frames, trans[uttid])
      outf = _SelectRandomShard(recordio_writers)
      outf.write(ex.SerializeToString())
    tar.close()
  file_obj.close()
  _CloseSubShards(recordio_writers)
  tf.logging.info(f'Processed {len(trans)} / {total_utts}')


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.dump_transcripts:
    assert False, "dump_transcripts option isn't supported. TODO: Remove."
  elif FLAGS.generate_tfrecords:
    _CreateAsrFeatures()
  else:
    tf.logging.error(
        'Nothing to do! Use --dump_transcripts or --generate_tfrecords')


if __name__ == '__main__':
  tf.disable_eager_execution()
  tf.app.run(main)
