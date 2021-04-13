# from functools import partial
# from io import BytesIO
# import json
# import logging
# import os
# import subprocess
# import shlex
# import tempfile
# from typing import List, Tuple
# import wave

# import ds_ctcdecoder
# from ftfy import fix_text, guess_bytes
# import langid
# import numpy as np
# import pandas as pd
# import re
# import srt

# from absl import app
# from absl import flags

# import pyspark
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, pandas_udf
# import pyspark.sql.functions as F
# import pyspark.sql.types as T
# from pyspark.sql.functions import array, array_contains, count, explode, lit
# from pyspark.sql.types import ArrayType, BinaryType, DoubleType, FloatType, ShortType, StructType, StructField, StringType, IntegerType, LongType

from galvasr2.align.spark.align import *

FLAGS = flags.FLAGS

flags.DEFINE_string('input_dir',
                    'gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/EXPANDED_LICENSES_FILTERED_ACCESS',
                    'Input directory. Exact format of this is a bit undefined right now and will likely change.')
flags.DEFINE_string('input_catalogue',
                    'gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/EXPANDED_LICENSES_FILTERED_ACCESS.jsonl.gz',
                    'Input catalogue. Basically just a dump of archive.org metadata for now.')
flags.DEFINE_string('vad_dir',
                    'gs://the-peoples-speech-west-europe/forced-aligner/Mar_18_2021/vad_pcm_tfrecords',
                    'Input catalogue. Basically just a dump of archive.org metadata for now.')

def main(argv):
  spark = SparkSession.builder \
                      .master("local[*]") \
                      .appName("Forced Aligner") \
                      .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
                      .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1")\
                      .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
                      .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
                      .config("spark.driver.memory", "120g")\
                      .config("spark.executor.memory", "120g")\
                      .config("spark.task.maxFailures", "2")\
                      .config("spark.rpc.askTimeout", "480s")\
                      .config("spark.executor.heartbeatInterval", "20000ms")\
                      .config("spark.eventLog.enabled", "true")\
                      .config("spark.eventLog.dir", "/development/lingvo-source/spark-events")\
                      .config("spark.history.fs.logDirectory", "/development/lingvo-source/spark-events")\
                      .getOrCreate()
  spark.sparkContext.setLogLevel("INFO") # "ALL" for very verbose logging
  logging.getLogger("py4j").setLevel(logging.ERROR)
  catalogue_df = load_audio_id_text_id_mapping(spark, FLAGS.input_catalogue)
  training_sample_rows = catalogue_df.collect()

  transcripts_df = load_transcripts(spark, FLAGS.input_dir, training_sample_rows)
  languages_df = transcripts_df.select(transcripts_df.text_document_id,
                                       infer_language_udf(transcripts_df.transcript).alias('language'))
  vad_df = spark.read.format('tfrecord').load(FLAGS.vad_dir)
  vad_df = vad_df.withColumn('duration', (F.length(vad_df.frames) / 2. / 16_000. / 60. / 60.))
  language_and_duration_df = languages_df.join(catalogue_df, ['text_document_id']).join(vad_df, ['audio_document_id']).select('language', 'duration')

  # language_and_frames_df.printSchema()
  # from IPython import embed; embed()

  # language_and_duration_df = language_and_frames_df.select(language_and_frames_df.language,
  #                                                          (F.length(language_and_frames_df.frames) / 2. / 16_000. / 60. / 60.).alias('duration'))

  rows = (language_and_duration_df.groupBy(language_and_duration_df.language)
          .sum('duration')
          .collect())
  print(rows)
  from IPython import embed; embed()

if __name__ == '__main__':
  app.run(main)

"""
[Row(language='en', sum(duration)=1493288.057928944),
Row(language='ro', sum(duration)=0.00013333333333333334), 
Row(language='pt', sum(duration)=0.7605275173611105), 
Row(language='ms', sum(duration)=10.04390634548606), 
Row(language='tr', sum(duration)=108.71806056423623), 
Row(language='de', sum(duration)=1.0519790277777779), 
Row(language='br', sum(duration)=0.024377508680555552), 
Row(language='es', sum(duration)=0.10690637152777778), 
Row(language='eu', sum(duration)=1.9484548437499973), 
Row(language='it', sum(duration)=0.06852736979166667), 
Row(language='nl', sum(duration)=1.4896868576388875), 
Row(language='mt', sum(duration)=0.01699868923611111), 
Row(language='no', sum(duration)=0.02376923611111111), 
Row(language='cy', sum(duration)=2.8405886631944433), 
Row(language='fr', sum(duration)=0.3955049739583286), 
Row(language='id', sum(duration)=0.17152500000000095), 
Row(language='la', sum(duration)=0.1311827517361111), 
Row(language='fi', sum(duration)=0.008527647569444446)]
"""
