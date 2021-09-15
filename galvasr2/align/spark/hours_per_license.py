import logging
import sys

from absl import app
from absl import flags
import pyspark
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

from galvasr2.align.spark.align_lib import (
    load_audio_id_text_id_mapping,
    load_transcripts,
    get_audio_seconds_udf,
    infer_language_udf,
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_catalogue",
    "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS.jsonl.gz",
    "Input catalogue. Basically just a dump of archive.org metadata for now.",
)
# input_dir and input_gcs_path must be kept in sync!
flags.DEFINE_string(
    "input_dir",
    "/root/the-peoples-speech-west-europe-bucket/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS",
    "Input directory. Exact format of this is a bit undefined right now and will likely change.",
)
flags.DEFINE_string(
    "input_gcs_path",
    "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS",
    "Input directory. Exact format of this is a bit undefined right now and will likely change.",
)


def main(argv):
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
        .config("spark.driver.memory", "40g")
        .config("spark.executor.memory", "40g")
        .config("spark.task.maxFailures", "2")
        .config("spark.rpc.askTimeout", "480s")
        .config("spark.executor.heartbeatInterval", "20000ms")
        .config("spark.eventLog.enabled", "true")
        .config("spark.eventLog.dir", "/development/lingvo-source/spark-events")
        .config(
            "spark.history.fs.logDirectory", "/development/lingvo-source/spark-events"
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("INFO")  # "ALL" for very verbose logging
    logging.getLogger("py4j").setLevel(logging.ERROR)
    catalogue_df = load_audio_id_text_id_mapping(spark, FLAGS.input_catalogue)

    # Uncomment this line if you want to run the entire pipeline in a relatively short amount of time.
    # catalogue_df = catalogue_df.limit(100)

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
    catalogue_df = catalogue_df.cache()

    # Switch the following 'rows' assignment for more granular license data.
    rows = catalogue_df.groupBy(normalize_license_udf(catalogue_df.licenseurl)).sum(
        "duration"
    )
    # rows = catalogue_df.groupBy(catalogue_df.licenseurl).sum('duration')

    # rows = rows.collect()
    for row in rows.collect():
        print(row)

    from IPython import embed

    embed()


NORMALIZE_LICENSE_RETURN_TYPE = T.StringType()


@F.pandas_udf(NORMALIZE_LICENSE_RETURN_TYPE)
def normalize_license_udf(licenseurl_series: pd.Series) -> pd.Series:
    normalized_licenses = []
    for licenseurl in licenseurl_series:
        if "government-works" in licenseurl:
            normalized_licenses.append("US Government Public Domain Work")
        elif "publicdomain" in licenseurl:
            normalized_licenses.append("Public Domain Work")
        elif "by-sa" in licenseurl:
            normalized_licenses.append("CC-BY-SA")
        elif "by" in licenseurl:
            # This doesn't every match by-sa, because a previous elif statement catches that.
            normalized_licenses.append("CC-BY")
        else:
            normalized_licenses.append(f"Unknown licenseurl: {licenseurl}")
    return pd.Series(normalized_licenses)


if __name__ == "__main__":
    app.run(main)
