import logging
import os
import sys

from absl import app
from absl import flags
import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from galvasr2.align.spark.align_lib import (
    load_audio_id_text_id_mapping,
    load_transcripts,
    get_audio_seconds_udf,
    infer_language_udf,
    infer_language_cld2_udf,
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_catalogue",
    "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS.jsonl.gz",
    "Input catalogue. Basically just a dump of archive.org metadata for now.",
)
flags.DEFINE_string(
    "input_gcs_path",
    "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS",
    "Input directory. Exact format of this is a bit undefined right now and will likely change.",
)


def main(argv):
    mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf(
        "SC_PHYS_PAGES"
    )  # e.g. 4015976448
    mem_gib = int((mem_bytes / (1024.0 ** 3)) * 0.9)
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("Hours Per Language")
        .config("spark.sql.files.ignoreMissingFiles", "true")
        .config("spark.sql.files.ignoreCorruptFiles", "true")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config(
            "spark.driver.extraJavaOptions",
            "-Dio.netty.tryReflectionSetAccessible=true",
        )
        .config(
            "spark.executor.extraJavaOptions",
            "-Dio.netty.tryReflectionSetAccessible=true",
        )
        .config("spark.driver.memory", f"{mem_gib}g")
        .config("spark.task.maxFailures", "2")
        .config("spark.rpc.askTimeout", "480s")
        .config("spark.executor.heartbeatInterval", "20000ms")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")  # "ALL" for very verbose logging
    logging.getLogger("py4j").setLevel(logging.ERROR)
    catalogue_df = load_audio_id_text_id_mapping(spark, FLAGS.input_catalogue)

    catalogue_df = catalogue_df.withColumn(
        "duration",
        get_audio_seconds_udf(
            F.concat(
                F.lit(FLAGS.input_gcs_path),
                F.lit("/"),
                catalogue_df.identifier,
                F.lit("/"),
                catalogue_df.audio_document_id,
            )
        )
        / 60.0
        / 60.0,
    )
    # catalogue_df = catalogue_df.cache()
    # Uncomment this line if you want to run the entire pipeline in a relatively short amount of time.
    # catalogue_df = catalogue_df.limit(10)
    catalogue_df = catalogue_df.collect()
    transcripts_df = load_transcripts(spark, FLAGS.input_gcs_path, training_sample_rows)
    languages_df = transcripts_df.select(
        transcripts_df.identifier,
        transcripts_df.text_document_id,
        infer_language_cld2_udf(transcripts_df.transcript).alias("language"),
    )

    catalogue_df = catalogue_df.join(languages_df, ["identifier", "text_document_id"])
    rows = catalogue_df.groupBy(catalogue_df.language).sum("duration").collect()
    print(rows)
    from IPython import embed

    embed()


if __name__ == "__main__":
    app.run(main)

"""
# More recent run
[Row(language='en', sum(duration)=51891.210179162255), 
Row(language='ro', sum(duration)=0.18604722861111112), 
Row(language='sk', sum(duration)=0.015528331944444444), 
Row(language='pt', sum(duration)=0.8029441672222222), 
Row(language='ms', sum(duration)=39.340964462500004), 
Row(language='tr', sum(duration)=0.0489819425), 
Row(language='de', sum(duration)=9.84690971388889), 
Row(language='br', sum(duration)=0.024625554166666667), 
Row(language='es', sum(duration)=4.928195001111111), 
Row(language='eu', sum(duration)=3.870641945), 
Row(language='it', sum(duration)=0.7015947202777778), 
Row(language='sv', sum(duration)=0.6141902797222223), 
Row(language='nl', sum(duration)=1.8030086005555555), 
Row(language='ru', sum(duration)=10.924616139444447), 
Row(language='mt', sum(duration)=0.01752277777777778), 
Row(language='no', sum(duration)=13.603658894999997), 
Row(language='bg', sum(duration)=0.3130866666666667), 
Row(language='cy', sum(duration)=5.814326111111112), 
Row(language='zu', sum(duration)=0.4729961111111111), 
Row(language='', sum(duration)=512.8351283508333), 
Row(language='se', sum(duration)=12.611557778888889), 
Row(language='fr', sum(duration)=3.429911108611111), 
Row(language='id', sum(duration)=0.3214013883333333), 
Row(language='la', sum(duration)=1.3733055555555556), 
Row(language='fi', sum(duration)=0.025745555555555556)]
"""

"""
[Row(language='en', sum(duration)=79043.2241041677), 
Row(language='ro', sum(duration)=0.18604722861111112), 
Row(language='sk', sum(duration)=0.015528331944444444), 
Row(language='pt', sum(duration)=0.8029441672222222), 
Row(language='ms', sum(duration)=39.340964462500004), 
Row(language='tr', sum(duration)=0.0489819425), 
Row(language='de', sum(duration)=11.079705825), 
Row(language='br', sum(duration)=0.024625554166666667), 
Row(language='es', sum(duration)=4.928195001111111), 
Row(language='eu', sum(duration)=3.870641945), 
Row(language='it', sum(duration)=0.7015947202777778), 
Row(language='sv', sum(duration)=0.6141902797222223), 
Row(language='nl', sum(duration)=1.8030086005555555), 
Row(language='ru', sum(duration)=10.924616139444442), 
Row(language='mt', sum(duration)=0.01752277777777778), 
Row(language='no', sum(duration)=13.603658894999999), 
Row(language='bg', sum(duration)=0.3130866666666667), 
Row(language='cy', sum(duration)=5.814326111111112), 
Row(language='zu', sum(duration)=0.4729961111111111), 
Row(language='', sum(duration)=512.8351283508333), 
Row(language='fr', sum(duration)=3.4299111086111114), 
Row(language='se', sum(duration)=12.611557778888889), 
Row(language='id', sum(duration)=0.3214013883333333), 
Row(language='la', sum(duration)=1.3733055555555556), 
Row(language='fi', sum(duration)=0.025745555555555556)]
"""
