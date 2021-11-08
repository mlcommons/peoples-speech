# Copyright 2021 NVIDIA

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import unittest
import warnings
import wave

import pyspark
import pyspark.sql.functions as F
import pyspark.sql.types as T


from galvasr2.align.spark.align_lib import (
    create_audio_segments_udf,
    create_audio_segment_names_udf,
    file_exists_udf,
    load_audio_and_text_dfs,
    load_audio_id_text_id_mapping,
)

from galvasr2.utils import find_runfiles


# class AudioIdTextIdMappingTest(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         conf = (
#             pyspark.SparkConf()
#             .setMaster("local[1]")
#             .setAppName(__file__)
#             .set("spark.sql.autoBroadcastJoinThreshold", "-1")
#             .set("spark.sql.debug.maxToStringFields", "1000")
#         )
#         cls.sc = pyspark.SparkContext(conf=conf)
#         cls.spark = pyspark.sql.SparkSession(cls.sc)
#         # pylint: enable=line-too-long
#         # This is done to disable these warnings:
#         # I have no idea what the cause is. When I search it, there is no known
#         # SPARK JIRA issue for this warning.
#         # /install/miniconda3/envs/100k-hours-lingvo-3/lib/python3.7/site-packages/pyspark/python/lib/pyspark.zip/pyspark/daemon.py:186: ResourceWarning: unclosed file <_io.BufferedReader name=4>
#         # pylint: disable=line-too-long
#         warnings.simplefilter("ignore", ResourceWarning)

#     def setUp(self):
#         # The line in setupClass doesn't appear to work sometimes. No idea why...
#         warnings.simplefilter("ignore", ResourceWarning)

#     @classmethod
#     def tearDownClass(cls):
#         cls.sc.stop()

#     @unittest.skip(
#         "Very slow to run. Basically the same as test_load_catalogue(), except that it does the fuzzy matchinh. May enable later."
#     )
#     def test_load_audio_id_text_id_mapping(self):
#         base_dir = "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS"
#         catalogue_path = "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS.jsonl.gz"
#         catalogue_df = load_audio_id_text_id_mapping(
#             self.spark, catalogue_path, base_dir
#         )

#         def full_path(
#             identifier_column, audio_or_text_id_column: pyspark.sql.column.Column
#         ):
#             return F.concat(
#                 F.lit(base_dir),
#                 F.lit("/"),
#                 identifier_column,
#                 F.lit("/"),
#                 audio_or_text_id_column,
#             )

#         audio_file_exists = file_exists_udf(
#             full_path(catalogue_df.identifier, catalogue_df.audio_document_id)
#         )
#         text_file_exists = file_exists_udf(
#             full_path(catalogue_df.identifier, catalogue_df.text_document_id)
#         )
#         files = catalogue_df.collect()

#     # Warning: Takes 76.7 minutes to run
#     def test_load_catalogue(self):
#         base_dir = "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS"
#         catalogue_path = "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS.jsonl.gz"
#         audio_df, text_df = load_audio_and_text_dfs(self.spark, catalogue_path)

#         def full_path(
#             identifier_column, audio_or_text_id_column: pyspark.sql.column.Column
#         ):
#             return F.concat(
#                 F.lit(base_dir),
#                 F.lit("/"),
#                 identifier_column,
#                 F.lit("/"),
#                 audio_or_text_id_column,
#             )

#         audio_file_exists = file_exists_udf(
#             full_path(audio_df.identifier, audio_df.audio_document_id)
#         )
#         text_file_exists = file_exists_udf(
#             full_path(text_df.identifier, text_df.text_document_id)
#         )
#         missing_audio_files = audio_df.filter(~audio_file_exists)
#         missing_audio_files_rows = missing_audio_files.collect()
#         assert len(missing_audio_files_rows) == 0, missing_audio_files_rows
#         missing_text_files = text_df.filter(~text_file_exists)
#         missing_text_files_rows = missing_text_files.collect()
#         assert len(missing_text_files_rows) == 0, missing_text_files_rows


class CreateAudioSegmentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf(
            "SC_PHYS_PAGES"
        )  # e.g. 4015976448
        mem_gib = int((mem_bytes / (1024.0 ** 3)) * 0.9)
        tar_jar = os.path.join(
            find_runfiles(), "__main__/galvasr2/spark/tar_spark_datasource.jar"
        )
        conf = (
            pyspark.SparkConf()
            .setMaster("local[*]")
            .setAppName(__file__)
            .set("spark.sql.autoBroadcastJoinThreshold", "-1")
            .set("spark.sql.debug.maxToStringFields", "1000")
            .set("spark.eventLog.enabled", "false")
            .set("spark.jars", tar_jar)
            .set("spark.driver.memory", f"{mem_gib}g")
            .set("spark.driver.maxResultSize", "4g")
            # .set("spark.history.fs.logDirectory",      /mycustomdir
        )
        cls.sc = pyspark.SparkContext(conf=conf)
        cls.spark = pyspark.sql.SparkSession(cls.sc)
        # pylint: enable=line-too-long
        # This is done to disable these warnings:
        # I have no idea what the cause is. When I search it, there is no known
        # SPARK JIRA issue for this warning.
        # /install/miniconda3/envs/100k-hours-lingvo-3/lib/python3.7/site-packages/pyspark/python/lib/pyspark.zip/pyspark/daemon.py:186: ResourceWarning: unclosed file <_io.BufferedReader name=4>
        # pylint: disable=line-too-long
        warnings.simplefilter("ignore", ResourceWarning)

    def setUp(self):
        # The line in setupClass doesn't appear to work sometimes. No idea why...
        warnings.simplefilter("ignore", ResourceWarning)

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()

    # def test_create_audio_segments(self):
    #     input_gcs_path = "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS"
    #     work_dir = "gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b"
    #     alignments_dir = os.path.join(work_dir, "alignments_json_jul_28")
    #     tars_dir = "tars_dir"
    #     manifest_dir = "manifest"

    #     spark = self.spark
    #     alignments_df = spark.read.json(alignments_dir)
    #     alignments_df = alignments_df.withColumn(
    #         "duration_ms",
    #         F.expr(
    #             "transform(arrays_zip(alignments.end_ms, alignments.start_ms), x -> x.end_ms - x.start_ms)"
    #         ),
    #     )

    #     alignments_df = alignments_df.where(alignments_df.identifier.contains("07282016HFUUforum"))

    #     assert alignments_df.count() == 1

    #     audio_paths = F.concat(
    #         F.lit(input_gcs_path),
    #         F.lit("/"),
    #         F.col("identifier"),
    #         F.lit("/"),
    #         F.col("audio_document_id"),
    #     )
    #     # paths = [alignments_df.select(audio_paths).head()[0]]
    #     rows = alignments_df.select(audio_paths).head(10)
    #     paths = [row[0] for row in rows]
    #     print(paths)
    #     audio_df = (
    #         spark.read.format("binaryFile")
    #         .load(paths)
    #         .drop("modificationTime", "length")
    #     )
    #     alignments_df = F.broadcast(alignments_df)
    #     alignments_audio_df = alignments_df.join(audio_df, audio_paths == audio_df.path)

    #     name = F.concat(F.col("identifier"), F.lit("-"), F.col("audio_document_id"))
    #     name = F.regexp_replace(name, r"/", "_SLASH_")
    #     name = F.regexp_replace(name, r"\.", "_DOT_")
    #     name = F.regexp_replace(name, r" ", "_SPACE_")
        
    #     alignments_audio_df = alignments_audio_df.withColumn(
    #         "aligned_chunks",
    #         create_audio_segments_udf(
    #             alignments_audio_df.content,
    #             F.lit("mp3"),
    #             name,
    #             alignments_audio_df.alignments.start_ms,
    #             alignments_audio_df.alignments.end_ms,
    #             F.lit("wav")
    #         ),
    #     )
    #     a = alignments_audio_df.select(
    #         F.explode(F.arrays_zip("aligned_chunks.audio_name", "aligned_chunks.audio"))
    #     ).select("col.0", "col.1")
    #     a.write.mode("overwrite").format("tar").save(tars_dir)

    #     output_df = alignments_audio_df.select(
    #         alignments_audio_df.identifier,
    #         alignments_audio_df.audio_document_id,
    #         alignments_audio_df.text_document_id,
    #         F.struct(
    #             alignments_audio_df.alignments.label.alias("label"),
    #             create_audio_segment_names_udf(
    #                 name, F.size(alignments_audio_df.alignments.start_ms), F.lit("wav")
    #             ).alias("name"),
    #             alignments_audio_df.duration_ms.alias("duration_ms"),
    #         ).alias("training_data"),
    #     )
    #     output_df.write.mode("overwrite").json(manifest_dir)

    #     tar_df = spark.read.format("tar").load(tars_dir)
    #     tar_df = spark.createDataFrame(tar_df.rdd.map(count_seconds),
    #                                    schema=
    #                                    T.StructType([T.StructField("key",T.StringType()),
    #                                                  T.StructField("duration", T.LongType())]))

    #     manifest_df = spark.read.format("json").load(manifest_dir)
    #     manifest_df = manifest_df.select(F.explode(F.arrays_zip(F.col("training_data.name"),
    #                                                             F.col("training_data.duration_ms"))))

    #     manifest_df = F.broadcast(manifest_df)

    #     df = manifest_df.join(tar_df, F.col("col.0") == tar_df.key, 'inner')

    #     df.write.mode("overwrite").format("json").save("my_results_json3")

    # def test_manifest_versus_tar(self):
    #     tars_dir = "gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b/dataset_tars_jul_28_wav_new_join_no_space_dropped_duplicated_data"
    #     manifest_dir = "gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b/dataset_manifest_jul_28_wav_new_join_no_space_dropped_duplicated_data"

    #     spark = self.spark
    #     # tars_df = spark.read.format("tar").load(tars_dir)
    #     manifest_df = spark.read.format("json").load(manifest_dir)

    #     # tars_df = tars_df.where(F.col("key").contains("07282016HFUUforum"))
    #     manifest_df = manifest_df.where(F.col("identifier").contains("07282016HFUUforum"))

    #     pdf = manifest_df.toPandas()
    #     from IPython import embed; embed()

    #     # tars_df.write.mode("overwrite").format("tar").save("my_tar_files4")
    #     tars_df = spark.read.format("tar").load("my_tar_files4")
    #     # results = tars_df.collect()
    #     # assert len(results) == 1, results
    #     # assert manifest_df.count() == 1

    #     tars_df = spark.createDataFrame(tars_df.rdd.map(count_seconds),
    #                                     schema=
    #                                     T.StructType([T.StructField("key",T.StringType()),
    #                                                   T.StructField("duration", T.LongType())]))

    #     manifest_df = manifest_df.select(F.explode(F.arrays_zip(F.col("training_data.name"),
    #                                                             F.col("training_data.duration_ms"))))
    #     manifest_df = F.broadcast(manifest_df)

    #     df = manifest_df.join(tars_df, F.col("col.0") == tars_df.key, 'inner')

    #     df.write.mode("overwrite").format("json").save("my_results_json4")

    def test_alignments_filter(self):
        work_dir = "gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b"
        alignments_dir = os.path.join(work_dir, "alignments_json_jul_28")
        spark = self.spark
        alignments_df = spark.read.json(alignments_dir)
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

        max_duration_ms = 20_000
        max_cer = 36.0
        min_duration_ms = 1_000

        alignments_df = alignments_df.withColumn(
            "alignments",
            F.filter(
                alignments_df.alignments,
                # Need to select this filter such that total number of
                # hours is 31,400
                lambda alignment: (alignment.duration_ms < max_duration_ms)
                & (alignment.cer < max_cer)
                & (alignment.duration_ms > min_duration_ms),
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
                # Is this the fix?
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
        abc = alignments_df.select(
            F.sum(F.expr("aggregate(alignments.duration_ms, 0L, (x, acc) -> acc + x)"))
            / 1000.0
            / 60.0
            / 60.0
        ).collect()
        print("GALVEZ:max_duration_ms=", max_duration_ms)
        print("GALVEZ:max_cer=", max_cer)
        print("GALVEZ:min_duration_ms=", min_duration_ms)
        print("GALVEZ:total number of hours=", abc)
        
        
# gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b/dataset_tars_jul_28_wav_new_join_no_space_dropped_duplicated_data
# gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b/dataset_manifest_jul_28_wav_new_join_no_space_dropped_duplicated_data
        
def count_seconds(row):
    with wave.open(io.BytesIO(row["value"])) as wav_fh:
        true_frames = len(wav_fh.readframes(wav_fh.getnframes())) / wav_fh.getsampwidth()
        return {"key": row["key"], "duration": int(1000.0 * (true_frames / wav_fh.getframerate()))}
        
if __name__ == "__main__":
    unittest.main()

# GALVEZ:max_duration_ms= 20000
# GALVEZ:max_cer= 30.0
# GALVEZ:min_duration_ms= 1000
# GALVEZ:total number of hours= [Row((((sum(aggregate(alignments.duration_ms, 0, lambdafunction((acc + x), x, acc))) / 1000.0) / 60.0) / 60.0)=28525.648870555557)]

# GALVEZ:max_duration_ms= 20000
# GALVEZ:max_cer= 31.0
# GALVEZ:min_duration_ms= 1000
# GALVEZ:total number of hours= [Row((((sum(aggregate(alignments.duration_ms, 0, lambdafunction((acc + x), x, acc))) / 1000.0) / 60.0) / 60.0)=29135.55706)]

# GALVEZ:max_duration_ms= 20000
# GALVEZ:max_cer= 32.0
# GALVEZ:min_duration_ms= 1000
# GALVEZ:total number of hours= [Row((((sum(aggregate(alignments.duration_ms, 0, lambdafunction((acc + x), x, acc))) / 1000.0) / 60.0) / 60.0)=29676.506351388885)]

# GALVEZ:max_duration_ms= 20000
# GALVEZ:max_cer= 33.0
# GALVEZ:min_duration_ms= 1000
# GALVEZ:total number of hours= [Row((((sum(aggregate(alignments.duration_ms, 0, lambdafunction((acc + x), x, acc))) / 1000.0) / 60.0) / 60.0)=30188.006468055555)]

# GALVEZ:max_duration_ms= 20000
# GALVEZ:max_cer= 34.0
# GALVEZ:min_duration_ms= 1000
# GALVEZ:total number of hours= [Row((((sum(aggregate(alignments.duration_ms, 0, lambdafunction((acc + x), x, acc))) / 1000.0) / 60.0) / 60.0)=30654.712304444445)]

# GALVEZ:max_duration_ms= 20000
# GALVEZ:max_cer= 35.0
# GALVEZ:min_duration_ms= 1000
# GALVEZ:total number of hours= [Row((((sum(aggregate(alignments.duration_ms, 0, lambdafunction((acc + x), x, acc))) / 1000.0) / 60.0) / 60.0)=31077.03023222222)]

# GALVEZ:max_duration_ms= 20000
# GALVEZ:max_cer= 36.0
# GALVEZ:min_duration_ms= 1000
# GALVEZ:total number of hours= [Row((((sum(aggregate(alignments.duration_ms, 0, lambdafunction((acc + x), x, acc))) / 1000.0) / 60.0) / 60.0)=31470.84820222222)]
