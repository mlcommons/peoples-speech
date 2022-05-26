"""
New forced alignment systemusing kaldi's cudadecoder implementation.
"""

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import glob
import shlex
import subprocess
import logging
import os
import re
import sys
from typing import Dict, List, Tuple

from absl import app
from absl import flags
import pandas as pd
import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T

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
    "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS",
    "Input directory. Exact format of this is a bit undefined right now and will likely change.",
)
flags.DEFINE_string("input_gcs_bucket", "gs://the-peoples-speech-west-europe", "")
flags.DEFINE_string(
    "input_dir",
    "/root/the-peoples-speech-west-europe-bucket/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS",
    "Input directory. Exact format of this is a bit undefined right now and will likely change.",
)
flags.DEFINE_string(
    "work_dir",
    "/development/lingvo-source/output_work_dir_3h",
    "Input directory. Exact format of this is a bit undefined right now and will likely change.",
)
flags.DEFINE_string(
    "alignments_work_dir", "/development/lingvo-source/output_work_dir_3h", ""
)
flags.DEFINE_float(
    "max_cer",
    20.0,  # 36.0,
    "Aligned utterances whose CER between alignment transcript and groundtruth transcript is hgiher than max_cer will be removed.",
)
flags.DEFINE_float(
    "min_cer",
    0.0,
    "Aligned utterances whose CER between alignment transcript and groundtruth transcript is <= min_cer will be removed.",
)
flags.DEFINE_integer(
    "max_duration_ms",
    16_700,  # 20_000,
    "Aligned utterances longer than max_duration_ms in length (in milliseconds) will be removed",
)
flags.DEFINE_integer(
    "min_duration_ms",
    1_000,
    "Aligned utterances shorter than min_duration_ms in length (in milliseconds) will be removed",
)
flags.DEFINE_integer(
    "number_of_shards",
    1024,
    "Aligned utterances longer than max_duration_ms in length (in milliseconds) will be removed",
)

flags.DEFINE_string("license_filter", "", "")


# def is_cc_by(column):
#     regexp = r"(http|https)://creativecommons.org/licenses/by/(1[.]0|2[.]0|2[.]5|3[.]0|4[.]0)"
#     return column.rlike(regexp)


def is_cc_by_sa(column):
    regexp = r"(http|https)://creativecommons.org/licenses/by-sa/(1[.]0|2[.]0|2[.]5|3[.]0|4[.]0)"
    return column.rlike(regexp)


def create_wav_scp(
    wav_scp_file_name: str, rows: List[pyspark.Row], base_path: str, ctm_path: str
):
    with open(wav_scp_file_name, "w") as fh:
        import tqdm

        for row in tqdm.tqdm(rows):
            key = os.path.join(row.identifier, row.audio_document_id)
            if ctm_path is not None:
                output_file_name = os.path.join(
                    ctm_path, row.kaldi_normalized_uttid + ".ctm"
                )
                if tf.io.gfile.exists(output_file_name):
                    continue
            path = os.path.join(base_path, key)
            value = f'/usr/bin/sox "{path}" -t wav --channels 1 --rate 8000 --encoding signed --bits 16 - |'
            line = f"{row.kaldi_normalized_uttid} {value}\n"
            fh.write(line)
        # fh.writelines(lines)


def split_wav_scp(posix_wav_scp, work_dir, num_splits):
    file_names = []
    file_handles = []
    split_dir = os.path.join(work_dir, "wav_scp_splits")
    os.makedirs(split_dir, exist_ok=True)
    for split in range(num_splits):
        file_names.append(os.path.join(split_dir, f"wav{split}.scp"))
        file_handles.append(open(file_names[-1], "w"))
    with open(posix_wav_scp) as fh:
        for i, line in enumerate(fh):
            i = i % num_splits
            file_handles[i].write(line)
    for fh in file_handles:
        fh.close()
    return file_names


def main(argv):
    mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf(
        "SC_PHYS_PAGES"
    )  # e.g. 4015976448
    mem_gib = int((mem_bytes / (1024.0**3)) * 0.9)
    (jar_path,) = galvasr2.__path__
    jars = ",".join(glob.glob(os.path.join(jar_path, "*.jar")))
    print("GALVEZ:jars=", jars)
    os.makedirs("/tmp/spark-events", exist_ok=True)
    spark = (
        pyspark.sql.SparkSession.builder.master(f"local[{os.cpu_count()}]")
        .config("spark.eventLog.enabled", "true")
        # .config("spark.eventLog.dir", "/spark-events")
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
        # .config("spark.history.fs.logDirectory", "/spark-events")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1")
        .config("spark.jars", jars)
        .config("spark.local.dir", "/mnt/disks/spark-scratch/")
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
    temp_catalogue_df = catalogue_df.withColumn("audio_paths", audio_paths).withColumn(
        "srt_paths", srt_paths
    )
    # print("GALVEZ:schema=")
    # temp_catalogue_df.printSchema()
    temp_catalogue_df.toPandas().to_json(
        "audio_id_text_id_mapping.json", orient="records", lines=True
    )

    _, licenseurl_df = load_audio_and_text_dfs(spark, FLAGS.input_catalogue)
    licenseurl_df = licenseurl_df.select(
        [F.col("identifier"), F.col("text_document_id"), F.col("licenseurl")]
    )

    # Kaldi's wav.scp format does not support space characters in the key field of a wav.scp file
    # We write the transcript to a file called "{kaldi_normalized_uttid}.ctm", so we also need to change all instances of "/" to "_"
    catalogue_df = catalogue_df.withColumn(
        "kaldi_normalized_uttid",
        F.concat_ws(
            "-",
            F.translate(catalogue_df.identifier, " /", "__"),
            F.translate(catalogue_df.audio_document_id, " /", "__"),
        ),
    )
    # key_int_mapping = os.path.join(FLAGS.work_dir, "key_int_mapping_csv")
    if not FLAGS.work_dir.startswith("gs://"):
        os.makedirs(FLAGS.work_dir, exist_ok=True)
    wav_scp = os.path.join(FLAGS.work_dir, "wav.scp")
    ctm_out_dir = os.path.join(FLAGS.work_dir, "decoder_ctm_dir")
    if FLAGS.stage <= 0 and FLAGS.end_stage >= 0:
        catalogue_df = catalogue_df.cache()
        # catalogue_df.write.mode("overwrite").format("csv").options(header="true").save(key_int_mapping)
        training_sample_rows = catalogue_df.collect()
        catalogue_df.unpersist()

        with TemporaryMountDirectory(
            mount_cmd=[
                "gcsfuse",
                "--implicit-dirs",
                FLAGS.input_gcs_bucket.lstrip("gs://"),
            ],
            unmount_cmd=["fusermount", "-u"],
        ) as temp_dir_name:
            posix_wav_scp = re.sub(
                r"^{0}".format(FLAGS.input_gcs_bucket), temp_dir_name, wav_scp
            )
            create_wav_scp(
                posix_wav_scp, training_sample_rows, FLAGS.input_dir, ctm_out_dir
            )

    # /development/lingvo-source/output_ctm_dir/

    # nvprof --analysis-metrics -o  decoder-analysis.nvprof \
    # We want only the best path, so we set lattice-beam to 0.1
    # --main-q-capacity=35000 \
    # Can get 266x RTF with this configuration. Keep it?
    # bath size of 100 and num channels of 100 works just fine

    if FLAGS.stage <= 1 and FLAGS.end_stage >= 1:
        if not FLAGS.work_dir.startswith("gs://"):
            os.makedirs(ctm_out_dir, exist_ok=True)
        with TemporaryMountDirectory(
            mount_cmd=[
                "gcsfuse",
                "--implicit-dirs",
                FLAGS.input_gcs_bucket.lstrip("gs://"),
            ],
            unmount_cmd=["fusermount", "-u"],
        ) as temp_dir_name:

            posix_ctm_out_dir = re.sub(
                r"^{0}".format(FLAGS.input_gcs_bucket), temp_dir_name, ctm_out_dir
            )
            posix_wav_scp = re.sub(
                r"^{0}".format(FLAGS.input_gcs_bucket), temp_dir_name, wav_scp
            )
            posix_work_dir = re.sub(
                r"^{0}".format(FLAGS.input_gcs_bucket), temp_dir_name, FLAGS.work_dir
            )
            num_gpus = 4
            posix_wav_scp_shards = split_wav_scp(
                posix_wav_scp, posix_work_dir, num_gpus
            )

            executor = ThreadPoolExecutor(max_workers=num_gpus)

            def run_gpu(posix_wav_scp_shard, gpu_number):
                cmd = f"""\
  /opt/kaldi/src/cudadecoderbin/batched-wav-nnet3-cuda3 \
  --frame-subsampling-factor=3 \
  --config=/opt/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/conf/online.conf \
  --max-active=7000 \
  --beam=15.0 \
  --lattice-beam=0.1 \
  --acoustic-scale=1.0 \
  --cuda-decoder-copy-threads=2 \
  --cuda-worker-threads={os.cpu_count() // num_gpus} \
  --segmentation=true \
  --cuda-use-tensor-cores=true \
  --max-batch-size=150 \
  --num-channels=250 \
  --lattice-postprocessor-rxfilename=/development/lingvo-source/lattice_postprocess.conf \
  --word-symbol-table=/opt/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/words.txt \
  /opt/kaldi/egs/aspire/s5/exp/chain/tdnn_7b/final.mdl \
  /opt/kaldi/egs/aspire/s5/exp/tdnn_7b_chain_online/graph_pp/HCLG.fst \
  scp,p:{posix_wav_scp_shard} \
  {posix_ctm_out_dir}
  """
                env = deepcopy(os.environ)
                env["CUDA_VISIBLE_DEVICES"] = f"{gpu_number}"
                subprocess.check_call(shlex.split(cmd), env=env)

            for i, shard in enumerate(posix_wav_scp_shards):
                executor.submit(run_gpu, shard, i)
            executor.shutdown(wait=True)

    alignments_dir = os.path.join(FLAGS.alignments_work_dir, "alignments_json_jul_28")
    if FLAGS.stage <= 2 and FLAGS.end_stage >= 2:
        # TODO: Add options to DSAlign here
        dsalign_args = dsalign_main.parse_args(
            ["--output-wer", "--output-cer"]
        )  # , "--output-sws", "--output-levenshtein"])

        alphabet_normalized_path = (
            "/development/lingvo-source/galvasr2/align/spark/alphabet2.txt"
        )
        align_udf = prepare_align_udf(
            dsalign_args, alphabet_normalized_path, 15_000, 3_000
        )

        ctm_df = (
            spark.read.format("binaryFile")
            .option("pathGlobFilter", "*.ctm")
            .load(ctm_out_dir)
        )
        ctm_df = ctm_df.withColumn(
            "kaldi_normalized_uttid",
            F.regexp_replace(F.reverse(F.split(ctm_df.path, "/"))[0], r"[.]ctm$", ""),
        )
        ctm_df = ctm_df.withColumn("ctm_content", fix_text_udf(F.col("content"))).drop(
            "path", "length", "modificationTime", "content"
        )

        ctm_df = ctm_df.join(catalogue_df, "kaldi_normalized_uttid")
        downsampled_catalogue_df = ctm_df.drop("ctm_content")

        training_sample_rows = downsampled_catalogue_df.collect()
        transcripts_df = load_transcripts(
            spark, FLAGS.input_gcs_path, training_sample_rows
        )
        transcripts_df = transcripts_df.withColumn(
            "transcript", normalize_english_text_udf(transcripts_df.transcript)
        )
        ctm_df = ctm_df.join(transcripts_df, ["identifier", "text_document_id"])
        ctm_df = ctm_df.repartition(960)

        # alignments_df = ctm_df.select(align_udf(F.concat(ctm_df.identifier, F.lit("/"), ctm_df.text_document_id),
        #                                         F.concat(ctm_df.identifier, F.lit("/"), ctm_df.audio_document_id),
        #                                         ctm_df.transcript, ctm_df.ctm_content))
        alignments_df = ctm_df.withColumn(
            "alignments",
            align_udf(
                F.concat(ctm_df.identifier, F.lit("/"), ctm_df.text_document_id),
                F.concat(ctm_df.identifier, F.lit("/"), ctm_df.audio_document_id),
                ctm_df.transcript,
                ctm_df.ctm_content,
            ),
        ).drop("ctm_content")
        print("GALVEZ:schema")
        alignments_df.printSchema()

        sys.stdout.flush()

        alignments_df.write.mode("overwrite").format("json").save(alignments_dir)

    manifest_dir = os.path.join(FLAGS.work_dir, "dataset_manifest")
    tars_dir = os.path.join(FLAGS.work_dir, "dataset_tars")
    if FLAGS.stage <= 3 and FLAGS.end_stage >= 3:
        duplicate_data_path = "gs://the-peoples-speech-west-europe/forced-aligner/data_deduplication/data_deduplication_v2_lines.json"
        duplicates_df = spark.read.format("json").load(duplicate_data_path)

        alignments_df = spark.read.json(alignments_dir)

        alignments_df = alignments_df.join(
            duplicates_df,
            on=(alignments_df.identifier == duplicates_df.identifier)
            & (alignments_df.text_document_id == duplicates_df.text_document_id),
            how="anti",
        )

        if FLAGS.license_filter == "":
            pass
        else:
            if FLAGS.license_filter == "Not CC-BY-SA":
                filtered_licenseurl_df = licenseurl_df.filter(
                    ~is_cc_by_sa(F.col("licenseurl"))
                )
            elif FLAGS.license_filter == "CC-BY-SA":
                filtered_licenseurl_df = licenseurl_df.filter(
                    is_cc_by_sa(F.col("licenseurl"))
                )
            else:
                raise Exception("Unknown license_filter provided.")
            filtered_licenseurl_df = filtered_licenseurl_df.drop("licenseurl")

            alignments_df = alignments_df.join(
                filtered_licenseurl_df,
                on=(alignments_df.identifier == filtered_licenseurl_df.identifier)
                & (
                    alignments_df.text_document_id
                    == filtered_licenseurl_df.text_document_id
                ),
                how="inner",
            )
            alignments_df = alignments_df.drop(filtered_licenseurl_df.identifier).drop(
                filtered_licenseurl_df.text_document_id
            )

        # We would like the number of partitions to be some large multiple
        # of the number of executors. Not every audio file is the same
        # length, so this helps with load balancing.
        alignments_df = alignments_df.withColumn(
            "duration_ms",
            F.expr(
                "transform(arrays_zip(alignments.end_ms, alignments.start_ms), x -> x.end_ms - x.start_ms)"
            ),
        )
        alignments_df = alignments_df.withColumn(
            "alignments",
            F.arrays_zip(
                alignments_df.alignments.cer,
                alignments_df.alignments.end_ms,
                alignments_df.alignments.label,
                alignments_df.alignments.start_ms,
                alignments_df.alignments.wer,
                alignments_df.duration_ms,
            ).cast(
                T.ArrayType(
                    T.StructType(
                        [
                            T.StructField("cer", T.FloatType()),
                            T.StructField("end_ms", T.LongType()),
                            T.StructField("label", T.StringType()),
                            T.StructField("start_ms", T.LongType()),
                            T.StructField("wer", T.FloatType()),
                            T.StructField("duration_ms", T.LongType()),
                        ]
                    )
                )
            ),
        )

        alignments_df = alignments_df.drop("duration_ms")

        alignments_df = alignments_df.withColumn(
            "alignments",
            F.filter(
                alignments_df.alignments,
                # Need to select this filter such that total number of
                # hours is 31,400
                lambda alignment: (alignment.duration_ms < FLAGS.max_duration_ms)
                & (alignment.duration_ms >= FLAGS.min_duration_ms)
                & (alignment.cer < FLAGS.max_cer)
                & (alignment.cer >= FLAGS.min_cer),
            ),
        )
        alignments_df = alignments_df.withColumn(
            "alignments",
            F.struct(
                alignments_df.alignments.cer,
                alignments_df.alignments.end_ms,
                alignments_df.alignments.label,
                alignments_df.alignments.start_ms,
                alignments_df.alignments.wer,
                alignments_df.alignments.duration_ms,
            ).cast(
                T.StructType(
                    [
                        T.StructField("cer", T.ArrayType(T.FloatType())),
                        T.StructField("end_ms", T.ArrayType(T.LongType())),
                        T.StructField("label", T.ArrayType(T.StringType())),
                        T.StructField("start_ms", T.ArrayType(T.LongType())),
                        T.StructField("wer", T.ArrayType(T.FloatType())),
                        T.StructField("duration_ms", T.ArrayType(T.LongType())),
                    ]
                )
            ),
        )

        alignments_df = alignments_df.repartition(960)

        abc = alignments_df.select(
            F.sum(F.expr("aggregate(alignments.duration_ms, 0L, (x, acc) -> acc + x)"))
            / 1000.0
            / 60.0
            / 60.0
        ).collect()
        print("GALVEZ:total number of hours=", abc)
        sys.stdout.flush()

        alignments_df = alignments_df.select(
            alignments_df.identifier,
            alignments_df.audio_document_id,
            alignments_df.text_document_id,
            alignments_df.alignments,
        )

        alignments_df = F.broadcast(alignments_df)

        audio_paths = F.concat(
            F.lit(FLAGS.input_gcs_path),
            F.lit("/"),
            F.col("identifier"),
            F.lit("/"),
            F.col("audio_document_id"),
        )
        rows = alignments_df.select(audio_paths).collect()
        paths = [row[0] for row in rows]  # [:1] # GALVEZ: WARNING test!
        # print(f"number of paths = {len(paths)}")
        audio_df = (
            spark.read.format("binaryFile")
            .load(paths)
            .drop("modificationTime", "length")
        )

        alignments_audio_df = alignments_df.join(audio_df, audio_paths == audio_df.path)
        # from IPython import embed; embed()

        # Remove "/" so that, if someat untars the tar files, everything will be dumped into one directory
        # Remove "." becasue it has special meaning in webdataset format.
        # Remove " " because kaldi keys may not contain " " (this is not striclty necessary, but convenient)
        name = F.concat(F.col("identifier"), F.lit("/"), F.col("audio_document_id"))
        # name = F.regexp_replace(name, r"/", "_SLASH_")
        name = F.regexp_replace(name, r"\.", "_DOT_")
        name = F.regexp_replace(name, r" ", "_SPACE_")
        # glob.glob("**/*.flac")

        pdf = df.select(name).collect()
        for name in pdf.name:
            assert len(name) < 4096
            for chunk in "/".split(name):
                assert len(chunk) < 256
        # name = F.regexp_replace(F.concat(F.col("identifier"),
        #                                  F.lit("-"),
        #                                  F.col("audio_document_id")),
        #                         r"(\.|/)",
        #                         "_"
        # )

        # The name of each thing in the tar file. May not exceed 100 characters in length
        # substr indexes from 1!
        # name = name.substr(
        #     F.length(name) - F.least(F.length(name), F.lit(88)) + 1,
        #     F.least(F.length(name), F.lit(88))
        # )

        alignments_audio_df = alignments_audio_df.withColumn(
            "aligned_chunks",
            create_audio_segments_udf(
                alignments_audio_df.content,
                F.lit("mp3"),
                name,
                alignments_audio_df.alignments.start_ms,
                alignments_audio_df.alignments.end_ms,
                F.lit("flac"),
            ),
        )
        a = alignments_audio_df.select(
            F.explode(F.arrays_zip("aligned_chunks.audio_name", "aligned_chunks.audio"))
        ).select("col.0", "col.1")
        a.write.mode("overwrite").format("tar").save(tars_dir)

        output_df = alignments_audio_df.select(
            alignments_audio_df.identifier,
            alignments_audio_df.audio_document_id,
            alignments_audio_df.text_document_id,
            F.struct(
                alignments_audio_df.alignments.label.alias("label"),
                create_audio_segment_names_udf(
                    # Is F.size right here?
                    name,
                    F.size(alignments_audio_df.alignments.start_ms),
                    F.lit("flac"),
                ).alias("name"),
                alignments_audio_df.alignments.duration_ms.alias("duration_ms"),
            ).alias("training_data"),
        )
        output_df = output_df.coalesce(960)

        # coalesce(1) seems to make the create_audio_segments_udf function run serially
        output_df.write.mode("overwrite").json(manifest_dir)

    repartitioned_tars_dir = os.path.join(FLAGS.work_dir, "repartitioned_dataset_tars")
    tmp_tars_dir = os.path.join(FLAGS.work_dir, "repartitioned_dataset_tmp_dir")
    if FLAGS.stage <= 4 and FLAGS.end_stage >= 4:
        tars_df = spark.read.format("tar").load(tars_dir)  # .limit(100)
        number_of_rows = tars_df.count()

        spark2 = spark.newSession()
        spark2.conf.set(
            "spark.sql.execution.rangeExchange.sampleSizePerPartition", number_of_rows
        )
        spark2.conf.set("spark.sql.files.minPartitionNum", FLAGS.number_of_shards)
        # tars_df = spark2.read.format("tar").load(tars_dir)#.limit(100)

        # print("GALVEZ:", tars_df.select(F.col("key")).collect())
        # import sys; sys.exit()
        tars_df = spark2.read.format("tar").load(tars_dir)  # .limit(100)
        tars_df = tars_df.repartitionByRange(FLAGS.number_of_shards, F.col("key"))
        # # May need to write this out to GCS, and then delete it, to prevent different behavior between runs.
        # # tars_df = tars_df.persist()
        tars_df.write.mode("overwrite").format("tar").save(tmp_tars_dir)
        tars_df = spark2.read.format("tar").load(
            tmp_tars_dir
        )  # .repartitionByRange()  # coalesce(1024)
        # counts_df = (
        #     tars_df.withColumn("partitionId", F.spark_partition_id())
        #     .groupBy("partitionId")
        #     .count()
        # )
        # num_rows_to_keep = counts_df.select(F.min(F.col("count"))).collect()[0][0]
        # # Consider doing this in java
        # def drop_final_rows(rows):
        #     for _ in range(num_rows_to_keep):
        #         yield next(rows)
        #     for _ in rows:
        #         pass
        #     return

        # print("GALVEZ:before=", tars_df.rdd.getNumPartitions())
        # # , preservesPartitioning=True
        # tars_df = spark2.createDataFrame(
        #     tars_df.rdd.mapPartitions(drop_final_rows), schema=tars_df.schema
        # )
        # print("GALVEZ:after=", tars_df.rdd.getNumPartitions())
        # import sys

        # sys.stdout.flush()
        # # Don't actually write this out right now. It doesn't benefit us unless we are doing nemo training in a specific mode.
        # tars_df.write.mode("overwrite").format("tar").save(repartitioned_tars_dir)

        # manifest_df = spark2.read.json(manifest_dir)
        # number_of_utterances = manifest_df.select(F.explode(F.col("training_data.name"))).count()
        # print(f"GALVEZ:number_of_utterances={number_of_utterances}")
        # utterances_per_shard = number_of_utterances // FLAGS.number_of_shards
        # repartition_tar_files(os.path.join(tars_dir, "*.tar"), repartitioned_tars_dir, utterances_per_shard)

    nemo_manifest_dir = os.path.join(FLAGS.work_dir, "dataset_manifest_nemo")
    nemo_single_manifest_dir = os.path.join(
        FLAGS.work_dir, "dataset_manifest_nemo_single"
    )

    if FLAGS.stage <= 5 and FLAGS.end_stage >= 5:
        json_df = spark.read.format("json").load(manifest_dir)
        nemo_df = json_df.select(
            F.explode(
                F.arrays_zip(
                    F.col("training_data.name").alias("audio_filepath"),
                    F.col("training_data.label").alias("text"),
                    F.col("training_data.duration_ms").alias("duration_ms"),
                )
            )
        )
        nemo_df = nemo_df.select(
            F.col("col.name").alias("audio_filepath"),
            F.col("col.label").alias("text"),
            (F.col("col.duration_ms").cast(T.DoubleType()) / 1000.0).alias("duration"),
            F.lit(-1).alias("shard_id"),
        )
        if False:
            tars_df = spark.read.format("tar").load(repartitioned_tars_dir)
            tars_df = tars_df.select(tars_df.key)
            nemo_df = F.broadcast(nemo_df)
            nemo_df = nemo_df.join(
                tars_df, F.col("audio_filepath") == F.col("key")
            ).drop(F.col("key"))

        # TODO: Join against tar files that have been made to contain the
        # same number of files to filter out removed files
        nemo_df.write.mode("overwrite").format("json").save(nemo_manifest_dir)

        nemo_single_df = spark.read.format("json").load(nemo_manifest_dir)
        nemo_single_df.coalesce(1).write.mode("overwrite").format("json").save(
            nemo_single_manifest_dir
        )

    single_manifest_dir = os.path.join(FLAGS.work_dir, "dataset_manifest_single")
    single_tar_dir = os.path.join(FLAGS.work_dir, "dataset_tars_single")
    # Create single tar file and single json file
    if FLAGS.stage <= 6 and FLAGS.end_stage >= 6:
        json_df = spark.read.format("json").load(manifest_dir)
        json_df.coalesce(1).write.format("json").mode("overwrite").save(
            single_manifest_dir
        )

        tars_df = spark.read.format("tar").load(tmp_tars_dir)
        tars_df.coalesce(1).write.format("tar").mode("overwrite").save(single_tar_dir)


if __name__ == "__main__":
    app.run(main)
