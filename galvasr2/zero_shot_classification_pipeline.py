import pandas as pd
from flair.data import Sentence
from flair.models import SequenceTagger
import logging
from absl import flags
from multiprocessing.pool import ThreadPool
from pyspark.sql import SparkSession
from tqdm import tqdm

from galvasr2.align.spark.align_lib import (
    load_audio_id_text_id_mapping,
    load_transcripts,
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_transcription_index",
    "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS.jsonl.gz",
    "Ubication of the path with the transcription index data",
)
flags.DEFINE_string(
    "input_transcription_data",
    "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS",
    "Ubication of the path with the transcription data",
)
flags.DEFINE_string(
    "num_rows",
    10,
    "Number of transcripts to analyze",
)


class ZSClassification:
    def __init__(self, num_rows: int):
        self.num_rows = num_rows

    def read_transcription_data(
        self,
        spark: SparkSession,
        data_trans_index: str,
        data_trans: str,
    ) -> pd.DataFrame:
        # spark.sparkContext.setLogLevel("INFO") # "ALL" for very verbose logging
        logging.getLogger("py4j").setLevel(logging.ERROR)
        catalogue_df = load_audio_id_text_id_mapping(spark, data_trans_index)
        training_sample_rows = catalogue_df.collect()

        # Comment this out to load everything. It might takes ~15 minute, in my experience, on an 8 core machine.
        training_sample_rows = training_sample_rows[: self.num_rows]
        transcripts_df = load_transcripts(spark, data_trans, training_sample_rows)
        transcripts_pdf = transcripts_df.toPandas()
        return transcripts_pdf

    def get_text_classification(transcripts_pdf: pd.DataFrame) -> pd.DataFrame:
        zsl = text.ZeroShotClassifier()
        labels = [
            "politics",
            "elections",
            "sports",
            "films",
            "television",
            "artificial intelligence",
            "food",
            "healthy",
            "Information technology",
            "financials",
            "communication services",
        ]

        def get_top_class(row):
            try:
                row = row[:20000]
                classification = zsl.predict(
                    row,
                    labels=labels,
                    include_labels=True,
                    batch_size=1,
                    multilabel=False,
                )
                classification.sort(key=lambda tup: tup[1])
                return classification[-1][0]
            except:
                return "problem"

        transcripts_pdf["classification"] = transcripts_pdf["transcript"].apply(
            get_top_class
        )
        return transcripts_pdf


def main():
    classification_model = ZSClassification(FLAGS.num_rows)
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("ZSClassification")
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
        .config("spark.rpc.askTimeout", "480s")
        .config("spark.executor.heartbeatInterval", "20000ms")
        .config("spark.eventLog.enabled", "true")
        .getOrCreate()
    )
    transcripts_pdf = classification_model.read_transcription_data(
        spark, FLAGS.data_trans_index, FLAGS.data_trans
    )
    result = classification_model.get_text_classification(transcripts_pdf)
    result.to_csv("DataClassification.csv", index=None)
    return "save file successful"


if __name__ == "__main__":
    main()
