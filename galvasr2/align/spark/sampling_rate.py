import logging
import sys

from absl import app
from absl import flags
import pyspark
import pyspark.sql.functions as F

from galvasr2.align.spark.align_lib import get_audio_seconds_udf, get_audio_sample_rate_udf, load_audio_id_text_id_mapping

flags.DEFINE_string('input_catalogue',
                    'gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS.jsonl.gz',
                    'Input catalogue. Basically just a dump of archive.org metadata for now.')
# input_dir and input_gcs_path must be kept in sync!
flags.DEFINE_string('input_dir',
                    '/root/the-peoples-speech-west-europe-bucket/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS',
                    'Input directory. Exact format of this is a bit undefined right now and will likely change.')
flags.DEFINE_string('input_gcs_path',
                    'gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS',
                    'Input directory. Exact format of this is a bit undefined right now and will likely change.')

FLAGS = flags.FLAGS

def main(argv):
  spark = pyspark.sql.SparkSession.builder \
                        .master("local[*]")\
                        .appName("Sampling Rates")\
                        .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
                        .config("spark.eventLog.enabled", "true")\
                        .config("spark.eventLog.dir", "/spark-events")\
                        .config("spark.history.fs.logDirectory", "/spark-events")\
                        .config("spark.executor.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
                        .config("spark.driver.extraJavaOptions", "-Dio.netty.tryReflectionSetAccessible=true")\
                        .getOrCreate()
  spark.sparkContext.setLogLevel("INFO") # "ALL" for very verbose logging
  logging.getLogger("py4j").setLevel(logging.ERROR)

  catalogue_df = load_audio_id_text_id_mapping(spark, FLAGS.input_catalogue)
  catalogue_df = catalogue_df.withColumn("duration",
                                         get_audio_seconds_udf(F.concat(F.lit(FLAGS.input_gcs_path), F.lit("/"), catalogue_df.identifier, F.lit("/"), catalogue_df.audio_document_id)) / 60. / 60.)
  catalogue_df = catalogue_df.withColumn("sampling_rate",
                                         get_audio_sample_rate_udf(F.concat(F.lit(FLAGS.input_gcs_path), F.lit("/"), catalogue_df.identifier, F.lit("/"), catalogue_df.audio_document_id)))
  catalogue_df = catalogue_df.cache()
  print("GALVEZ:total_count=", catalogue_df.count())
  print("GALVEZ:total_hours=", F.sum(catalogue_df.duration))
  hours_rows = catalogue_df.groupBy(catalogue_df.sampling_rate).sum('duration').collect()
  count_rows = catalogue_df.groupBy(catalogue_df.sampling_rate).count().collect()
  print(hours_rows)
  print(count_rows)
  from IPython import embed; embed()
    
if __name__ == '__main__':
  app.run(main)

"""
[Row(sampling_rate="b'22050'", sum(duration)=8459.5085537325), 
Row(sampling_rate="b'11025'", sum(duration)=39.528464601388876), 
Row(sampling_rate="b'24000'", sum(duration)=1084.5719555555556), 
Row(sampling_rate="b'44100'", sum(duration)=23503.168311761947), 
Row(sampling_rate="b'12000'", sum(duration)=0.5620044444444444), 
Row(sampling_rate="b'8000'", sum(duration)=182.26534), 
Row(sampling_rate="b'32000'", sum(duration)=533.8393130555557), 
Row(sampling_rate='', sum(duration)=0.0), 
Row(sampling_rate="b'48000'", sum(duration)=17733.63398750001), 
Row(sampling_rate="b'16000'", sum(duration)=988.435469166667)]

[Row(sampling_rate="b'22050'", count=13102), 
Row(sampling_rate="b'11025'", count=63), 
Row(sampling_rate="b'24000'", count=2015), 
Row(sampling_rate="b'44100'", count=39439), 
Row(sampling_rate="b'12000'", count=1), 
Row(sampling_rate="b'8000'", count=289), 
Row(sampling_rate="b'32000'", count=880), 
Row(sampling_rate='', count=26), 
Row(sampling_rate="b'48000'", count=19085), 
Row(sampling_rate="b'16000'", count=1727)]

"""
