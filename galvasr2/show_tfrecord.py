import tensorflow as tf
# raw_dataset = tf.data.TFRecordDataset("/home/ws15dgalvez/dev-clean.tfrecords-00000-of-00001")
# raw_dataset = tf.data.TFRecordDataset("gs://the-peoples-speech-west-europe/forced-aligner/vad-segments-dump/segments.tfrecord/uttid=Highway_and_Hedges_Outreach_Ministries_Show_-_Show_49-0/part-00000-e5d1a602-78dd-418e-9c96-65945ce5daf1.c000.tfrecord")
# raw_dataset = tf.data.TFRecordDataset("gs://the-peoples-speech-west-europe/forced-aligner/vad-segments-dump/segments1/id=Highway_and_Hedges_Outreach_Ministries_Show_-_Show_49/part-00000-411b1b96-734e-48c3-b82a-60bb639b0f46.c000.tfrecord")
raw_dataset = tf.data.TFRecordDataset("gs://the-peoples-speech-west-europe/forced-aligner/vad-segments-dump/small_dataset_under_20_MB_each/id=SchoolRules/part-00000-fb274d2c-65c5-4bd5-bce4-97696fc3cbb8.c000.tfrecord")
# raw_dataset = tf.data.TFRecordDataset("gs://the-peoples-speech-west-europe/Librispeech/devtest/dev-clean.tfrecords-00000-of-00001")

for raw_record in raw_dataset: #.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)
