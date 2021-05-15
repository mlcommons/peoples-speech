from dump_cc_by_licenses import create_dump_license_data
from pyspark.sql import SparkSession


def test_create_dump_license():
    spark = SparkSession.builder.appName("CC-BY-license").getOrCreate()
    input_catalogue_path = "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/EXPANDED_LICENSES_FILTERED_ACCESS.jsonl.gz"
    data_license = create_dump_license_data(spark, input_catalogue_path)
    assert len(data_license.columns) == 4
    assert data_license.columns == ["licenseurl", "creator", "title", "credits"]
