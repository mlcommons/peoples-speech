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


class NER:
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

    def load_ner_model(self) -> SequenceTagger:
        model = SequenceTagger.load("flair/ner-english-ontonotes-fast")
        return model

    def get_ner_transcription(self, transcripts_pdf: pd.DataFrame) -> dict:
        ner_entities = {
            "CARDINAL": [],
            "DATE": [],
            "EVENT": [],
            "FAC": [],
            "GPE": [],
            "LANGUAGE": [],
            "LAW": [],
            "LOC": [],
            "MONEY": [],
            "NORP": [],
            "ORDINAL": [],
            "ORG": [],
            "PERCENT": [],
            "PERSON": [],
            "PRODUCT": [],
            "QUANTITY": [],
            "TIME": [],
            "WORK_OF_ART": [],
        }

        def get_top_class(tagger, row):
            try:
                sentence = Sentence(row)
                tagger.predict(sentence)
                for entity in sentence.get_spans("ner"):
                    ner_entities[entity.tag].append(entity.text)
            except:
                return "problem"

        model = self.load_ner_model()
        transcripts = transcripts_pdf["transcript"].values
        for i in tqdm(range(transcripts_pdf.shape[0])):
            get_top_class(model, transcripts[i])
        df_ner = pd.DataFrame.from_dict(ner_entities, orient="index").T
        return df_ner


def main():
    ner_model = NER(FLAGS.num_rows)
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("NER")
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
    transcripts_pdf = ner_model.read_transcription_data(
        spark, FLAGS.data_trans_index, FLAGS.data_trans
    )
    result = ner_model.get_ner_transcription(transcripts_pdf)
    result.to_csv("transcriptNER.csv", index=None)
    return "save file successful"


if __name__ == "__main__":
    main()
