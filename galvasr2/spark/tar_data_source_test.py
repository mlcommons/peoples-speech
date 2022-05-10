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

import os
import tempfile
import unittest
import warnings

import pyspark
import pyspark.sql.functions as F

from galvasr2.utils import find_runfiles


class TarDataSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tar_jar = os.path.join(
            find_runfiles(), "__main__/galvasr2/spark/tar_spark_datasource.jar"
        )
        cls.spark_events_dir = tempfile.TemporaryDirectory("spark-events")
        conf = (
            pyspark.SparkConf()
            .setMaster("local[1]")
            .setAppName(__file__)
            .set("spark.sql.debug.maxToStringFields", "1000")
            .set("spark.eventLog.dir", "file://" + cls.spark_events_dir.name)
            .set("spark.jars", tar_jar)
        )
        cls.sc = pyspark.SparkContext(conf=conf)
        cls.spark = pyspark.sql.SparkSession(cls.sc)
        # pylint: enable=line-too-long
        # This is done to disable these warnings:
        # I have no idea what the cause is. When I search it, there is no known
        # SPARK JIRA issue for this warning.
        # /install/miniconda3/envs/100k-hours-lingvo-3/lib/python3.7/site-packages/pyspark/python/lib/pyspark.zip/pyspark/daemon.py:186: ResourceWarning: unclosed file <_io.BufferedReader name=4> pylint: disable=line-too-long
        warnings.simplefilter("ignore", ResourceWarning)

    def setUp(self):
        # The line in setupClass doesn't appear to work sometimes. No idea why...
        warnings.simplefilter("ignore", ResourceWarning)

    @classmethod
    def tearDownClass(cls):
        cls.sc.stop()
        cls.spark_events_dir.cleanup()

    def test_load_tar(self):
        # tar_path = "gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b/repartitioned_dataset_tars_jul_28_flac/part-01023-cea535b2-8139-49bf-912a-2fa585ff57e4-c000.tar"
        tar_path = "gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b/repartitioned_dataset_tars_jul_28_flac/part-00000-cea535b2-8139-49bf-912a-2fa585ff57e4-c000.tar"

        tars_df = self.spark.read.format("tar").load(tar_path)

        tars_df.write.format("json").mode("overwrite").save(
            "/home/ext_dt_galvez_gmail_com/json_test"
        )

        tars_df.select(tars_df.key).write.format("json").mode("overwrite").save(
            "/home/ext_dt_galvez_gmail_com/json_test2"
        )

        a = tars_df.select(tars_df.key).head()
        json_df = self.spark.read.format("json").load(
            "/home/ext_dt_galvez_gmail_com/json_test"
        )
        b = json_df.select(json_df.key).head()
        json2_df = self.spark.read.format("json").load(
            "/home/ext_dt_galvez_gmail_com/json_test2"
        )
        c = json2_df.head()
        from IPython import embed

        embed()


if __name__ == "__main__":
    unittest.main()
