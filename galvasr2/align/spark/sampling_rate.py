import logging
import sys

from absl import app
from absl import flags
import pyspark
import pyspark.sql.functions as F

from galvasr2.align.spark.align_lib import get_audio_seconds_udf, load_audio_id_text_id_mapping

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
  # catalogue_df = catalogue_df.cache()

  # print("GALVEZ:count=", catalogue_df.count())

  # print("GALVEZ:", "\n".join([str(row) for row in catalogue_df.head(10)]))

  print("GALVEZ:duration=",
        catalogue_df.select(F.sum(get_audio_seconds_udf(F.concat(F.lit(FLAGS.input_dir), F.lit("/"), catalogue_df.identifier, F.lit("/"), catalogue_df.audio_document_id)))).collect())

if __name__ == '__main__':
  app.run(main)
