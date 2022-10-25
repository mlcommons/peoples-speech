"""
Alignment using NeMo models
"""

from collections.abc import Iterable
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import glob
import math
import shlex
import subprocess
import logging
import os
import re
import sys
from typing import Dict, List, Tuple

from absl import app
from absl import flags
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.streaming_utils import (
    AudioFeatureIterator,
    BatchedFeatureFrameBufferer,
    FrameBatchASR,
)
import numpy as np
import pandas as pd
import pyarrow as pa
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T
import torch

import galvasr2
from galvasr2.align.spark.align_lib import (
    create_audio_segments_udf,
    create_audio_segment_names_udf,
    fix_text_udf,
    load_audio_and_text_dfs,
    load_audio_id_text_id_mapping,
    load_transcripts,
    normalize_english_text_udf,
    prepare_create_audio_segments_udf,
    repartition_tar_files,
    TemporaryMountDirectory,
)
from galvasr2.align.spark.dsalign_lib import prepare_align_udf
from galvasr2.align.spark.schemas import ARCHIVE_ORG_SCHEMA
from galvasr2.utils import find_runfiles
from galvasr2.align import dsalign_main


FLAGS = flags.FLAGS

flags.DEFINE_integer("stage", 2, "")
flags.DEFINE_integer("end_stage", sys.maxsize, "")
flags.DEFINE_string(
    "input_catalogue",
    "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS.jsonl.gz",
    "Input catalogue. Basically just a dump of archive.org metadata for now.",
)
flags.DEFINE_string(
    "input_gcs_path",
    "/PLEASE_SET",
    "Input directory. Exact format of this is a bit undefined right now and will likely change. Write now it is created by scripts/archive.org/download_items.py",
)
flags.DEFINE_string("work_dir", "", "")
flags.DEFINE_string(
    "spark_local_dir",
    "/mnt/disks/spark-scratch/",
    "Hard drive that should be at least 1 TiB in size. Used for shuffling.",
)


def main(argv):
    mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf(
        "SC_PHYS_PAGES"
    )  # e.g. 4015976448
    mem_gib = int((mem_bytes / (1024.0**3)) * 0.9)
    os.makedirs("/tmp/spark-events", exist_ok=True)
    spark = (
        pyspark.sql.SparkSession.builder.master(f"local[{os.cpu_count()}]")
        .config("spark.eventLog.enabled", "true")
        .config(
            "spark.hadoop.fs.AbstractFileSystem.gs.impl",
            "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS",
        )
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
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1")
        .config("spark.local.dir", FLAGS.spark_local_dir)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("INFO")  # "ALL" for very verbose logging
    logging.getLogger("py4j").setLevel(logging.ERROR)

    catalogue_df = load_audio_id_text_id_mapping(spark, FLAGS.input_catalogue)
    audio_paths = F.concat(
        F.lit(FLAGS.input_gcs_path),
        F.lit("/"),
        F.col("identifier"),
        F.lit("/"),
        F.col("audio_document_id"),
    )
    srt_paths = F.concat(
        F.lit(FLAGS.input_gcs_path),
        F.lit("/"),
        F.col("identifier"),
        F.lit("/"),
        F.col("text_document_id"),
    )

    audio_paths_rows = catalogue_df.select(audio_paths).head(10)
    audio_paths_py = [row[0] for row in audio_paths_rows][:2]

    spark_single_executor = (
        pyspark.sql.SparkSession.builder.master("local[1]")
        .config("spark.eventLog.enabled", "true")
        .config(
            "spark.hadoop.fs.AbstractFileSystem.gs.impl",
            "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS",
        )
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
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "2")
        .config("spark.local.dir", FLAGS.spark_local_dir)
        .getOrCreate()
    )
    spark_single_executor.sparkContext.setLogLevel(
        "INFO"
    )  # "ALL" for very verbose logging
    logging.getLogger("py4j").setLevel(logging.ERROR)

    audio_df = (
        spark_single_executor.read.format("binaryFile")
        .load(audio_paths_py)
        .drop("path", "length", "modificationTime")
    )

    mp3_decode_udf, mp3_decode_return_schema = prepare_mp3_decode_udf(16_000)
    # Problem with conformer is that the framing is a little weird due
    # to convolutional subsampling.
    nemo_transcribe_udf, nemo_transcribe_return_schema = prepare_nemo_transcribe_udf(
        "stt_en_conformer_ctc_small",
        int(
            spark_single_executor.conf.get(
                "spark.sql.execution.arrow.maxRecordsPerBatch"
            )
        ),
        1.6,
        4.0,
    )
    transcripts_df = audio_df.mapInArrow(
        mp3_decode_udf, mp3_decode_return_schema
    ).mapInArrow(nemo_transcribe_udf, nemo_transcribe_return_schema)
    print("GALVEZ:", transcripts_df.head())


def load_raw_audios(
    frame_batch_asr, waveforms_batch: List[np.array], delay, model_stride_in_secs
):
    assert len(waveforms_batch) == frame_batch_asr.batch_size

    # Read in a batch of audio files, one by one
    for idx in range(frame_batch_asr.batch_size):
        samples = np.pad(
            waveforms_batch[idx],
            (
                0,
                int(
                    delay
                    * model_stride_in_secs
                    * frame_batch_asr.asr_model._cfg.sample_rate
                ),
            ),
        )
        frame_reader = AudioFeatureIterator(
            samples,
            frame_batch_asr.frame_len,
            frame_batch_asr.raw_preprocessor,
            frame_batch_asr.asr_model.device,
        )
        frame_batch_asr.set_frame_reader(frame_reader, idx)


def prepare_mp3_decode_udf(sample_rate: int):
    mp3_decode_return_schema = T.StructType([T.StructField("data", T.BinaryType())])

    def mp3_decode(batches: Iterable[pa.RecordBatch]):
        for batch in batches:
            return_batch = []
            for byte_stream in batch.column("content"):
                cmd = f"sox -t mp3 - -t raw --channels 1 --rate {sample_rate} --encoding signed --bits 16 -"
                with subprocess.Popen(
                    shlex.split(cmd),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                ) as p:
                    print("GALVEZ:input_length=", len(byte_stream.as_py()))
                    out, err = p.communicate(input=byte_stream.as_py())
                    print("GALVEZ:stderr=", err)
                    print("GALVEZ:output_length=", len(out))
                    return_batch.append(out)
            yield pa.RecordBatch.from_pydict({"data": return_batch})
            # yield pa.Table.from_arrays([return_batch], names=["data"])

    return mp3_decode, mp3_decode_return_schema


# batch_size should be derived from "spark.sql.execution.arrow.maxRecordsPerBatch"
def prepare_nemo_transcribe_udf(
    model_name: str, batch_size: int, chunk_len, total_buffer_in_secs
):
    nemo_transcribe_return_schema = T.StructType(
        [T.StructField("transcripts", T.StringType())]
    )

    def nemo_transcribe(batches: Iterable[pa.RecordBatch]):
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
            model_name=model_name
        )

        decoding_cfg = OmegaConf.create({})
        with open_dict(decoding_cfg):
            decoding_cfg.strategy = "greedy"
            decoding_cfg.compute_timestamps = True
        asr_model.change_decoding_strategy(decoding_cfg)

        asr_model.preprocessor.featurizer.dither = 0.0
        asr_model.preprocessor.featurizer.pad_to = 0
        asr_model.encoder.freeze()
        asr_model.decoder.freeze()

        asr_model.eval()
        # Why in the world is this necessary?
        asr_model = asr_model.to(asr_model.device)
        # frame_asr = BatchedFrameBatchASR(
        #     asr_model=asr_model, frame_len=chunk_len,
        #     total_buffer=total_buffer_in_secs, batch_size=batch_size,
        # )

        # model_stride_in_seconds = 80 / 1000  # Make sure to change this as appropriate...
        # tokens_per_chunk = math.ceil(chunk_len / model_stride_in_seconds)
        # mid_delay = math.ceil((chunk_len + (total_buffer_in_secs - chunk_len) / 2) / model_stride_in_seconds)

        for batch in batches:
            assert len(batch) <= batch_size

            batch_transcripts = []

            for raw_audio in batch.column("data"):
                print("GALVEZ:raw_audio_length=", len(raw_audio.as_py()))
                input_signal = torch.from_numpy(
                    np.frombuffer(raw_audio.as_py(), dtype=np.int16)
                )
                input_signal = input_signal.unsqueeze(0)

                # We need to do batches in reasonable sizes. e.g., 10 minutes

                # Do 15 second chunks to match training situation

                # Make everything the same size. Eitehr throw out or
                # run as a singletone the final chunk.
                N_MINUTES_IN_SAMPLES = 5 * 60 * 16_000
                greedy_prediction_chunks = []
                # logits_len = torch.tensor([0], dtype=torch.int64)
                # torch.set_printoptions(profile="full")
                # May want to overlap somehow...
                # How does NeMo subsample inputs?

                # valid or full convolutions?
                for start_sample in range(
                    0, input_signal.shape[1], N_MINUTES_IN_SAMPLES
                ):
                    end_sample = min(
                        start_sample + N_MINUTES_IN_SAMPLES, input_signal.shape[1]
                    )
                    chunk_input_signal = input_signal[:, start_sample:end_sample]
                    chunk_input_signal_length = torch.tensor(
                        [input_signal.shape[1]], dtype=torch.int64
                    )
                    (
                        chunk_logits,
                        chunk_logits_len,
                        greedy_predictions,
                    ) = asr_model.forward(
                        input_signal=chunk_input_signal,
                        input_signal_length=chunk_input_signal_length,
                    )

                    (
                        current_hypotheses,
                        _,
                    ) = asr_model.decoding.ctc_decoder_predictions_tensor(
                        chunk_logits,
                        decoder_lengths=chunk_logits_len,
                        return_hypotheses=True,
                    )

                    for i in range(len(current_hypotheses)):
                        current_hypotheses[i].y_sequence = chunk_logits[i][
                            : chunk_logits_len[i]
                        ]

                    # Need to convert integer timesteps to actual times somehow...
                    # Check out how riva does it?

                    # logits_len += chunk_logits_len
                    print("GALVEZ:greedy_prediction=", greedy_predictions.shape)
                    greedy_prediction_chunks.append(greedy_predictions)
                    break
                greedy_prediction = torch.cat(greedy_prediction_chunks, dim=1)
                print("GALVEZ:greedy_prediction_full=", greedy_prediction.shape)
                hypothesis = asr_model._wer.ctc_decoder_predictions_tensor(
                    greedy_prediction,
                    # predictions_len=logits_len,
                    return_hypotheses=True,
                )
                print("GALVEZ:hypothesis=", hypothesis)
                batch_transcripts.append(hypothesis)
                # batch_transcripts.append(" ".join(greedy_prediction_chunks))

            # load_raw_audios(frame_asr,
            #                 [np.frombuffer(raw_audio.as_py(), dtype=np.int16) for raw_audio in batch.column("data")],
            #                 mid_delay,
            #                 model_stride_in_seconds
            # )
            # # I need logits, eventually, in order to do ctc segmentation...
            # _ = frame_asr.transcribe(tokens_per_chunk, mid_delay)

            # batch_transcripts = [self.tokenizer.ids_to_text(blanks_transcript)
            #                      for blanks_transcript in frame_asr.unmerged]

            # frame_asr.reset()

            yield pa.RecordBatch.from_pydict({"transcripts": batch_transcripts})
            # yield pa.Table.from_arrays([batch_transcripts], names=["transcripts"])

    return nemo_transcribe, nemo_transcribe_return_schema


# utils.py
# timings, char_probs, char_list = cs.ctc_segmentation(config, log_probs, ground_truth_mat)


if __name__ == "__main__":
    app.run(main)
