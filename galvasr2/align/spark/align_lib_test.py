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

import unittest
import warnings

import pyspark
import pyspark.sql.functions as F

from galvasr2.align.spark.align_lib import (
    file_exists_udf, load_audio_and_text_dfs, load_audio_id_text_id_mapping,
)

class AudioIdTextIdMappingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        conf = (pyspark.SparkConf().setMaster("local[1]").setAppName(__file__)
                .set("spark.sql.autoBroadcastJoinThreshold", "-1")
                .set("spark.sql.debug.maxToStringFields", "1000"))
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

    @unittest.skip("Very slow to run. Basically the same as test_load_catalogue(), except that it does the fuzzy matchinh. May enable later.")
    def test_load_audio_id_text_id_mapping(self):
        base_dir = "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS"
        catalogue_path = "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS.jsonl.gz"
        catalogue_df = load_audio_id_text_id_mapping(self.spark, catalogue_path, base_dir)

        def full_path(identifier_column, audio_or_text_id_column: pyspark.sql.column.Column):
            return F.concat(F.lit(base_dir), F.lit("/"), identifier_column, F.lit("/"), audio_or_text_id_column)
        audio_file_exists = file_exists_udf(full_path(catalogue_df.identifier,
                                                      catalogue_df.audio_document_id))
        text_file_exists = file_exists_udf(full_path(catalogue_df.identifier,
                                                     catalogue_df.text_document_id))
        files = catalogue_df.collect()
        
    def test_load_catalogue(self):
        base_dir = "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS"
        catalogue_path = "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/CC_BY_SA_EXPANDED_LICENSES_FILTERED_ACCESS.jsonl.gz"
        audio_df, text_df = load_audio_and_text_dfs(self.spark, catalogue_path)
        def full_path(identifier_column, audio_or_text_id_column: pyspark.sql.column.Column):
            return F.concat(F.lit(base_dir), F.lit("/"), identifier_column, F.lit("/"), audio_or_text_id_column)
        audio_file_exists = file_exists_udf(full_path(audio_df.identifier,
                                                      audio_df.audio_document_id))
        text_file_exists = file_exists_udf(full_path(text_df.identifier,
                                                     text_df.text_document_id))
        missing_audio_files = audio_df.filter(~audio_file_exists)
        missing_audio_files_rows = missing_audio_files.collect()
        assert len(missing_audio_files_rows) == 0, missing_audio_files_rows
        missing_text_files = text_df.filter(~text_file_exists)
        missing_text_files_rows = missing_text_files.collect()
        assert len(missing_text_files_rows) == 0, missing_text_files_rows

if __name__ == '__main__':
    unittest.main()
