import tensorflow as tf

file_name = "gs://the-peoples-speech-west-europe/forced-aligner/vad-segments-dump/Nov_6_2020/ALL_CAPTIONED_DATA_004/part-00000-ee85e658-8458-4dd2-9851-e93ba0db81f5-c000.tfrecord"

# tf.enable_eager_execution()
# print("SUM:", sum(1 for _ in tf.data.TFRecordDataset(file_name)))
features = dict([
  ('uttid', tf.io.VarLenFeature(tf.string)),
  ('transcript', tf.io.VarLenFeature(tf.string)),
  ('frames', tf.io.VarLenFeature(tf.float32)),
  ])
for record in tf.data.TFRecordDataset(file_name):
  example = tf.io.parse_single_example(record, features)
  fval = {k: v.values for k, v in example.items()}
  print(fval['frames'].numpy().size)
  # from IPython import embed; embed()
  # print(['frames'])
  # break
