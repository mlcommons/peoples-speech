#!/usr/bin/env python
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from absl import app
from absl import flags
from align.spark.schemas import ARCHIVE_ORG_SCHEMA

flags.DEFINE_string(
    "input_catalogue_path",
    "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/EXPANDED_LICENSES_FILTERED_ACCESS.jsonl.gz",
    "Ubication of the path with the licence metadata",
)
flags.DEFINE_string("save_as", "csv", "Format to save the file")

FLAGS = flags.FLAGS


def my_concat(*cols):
    """Generate a format that allows import a Spark df as a one column txt

    Parameters
    ----------
    *cols : list
        columns
    Returns
    -------
    Spark data_license
        Data in the format needed to save as a txt
    """
    concat_columns = []
    for column in cols[:-1]:
        concat_columns.append(F.coalesce(column, F.lit("*")))
        concat_columns.append(F.lit(" "))
    concat_columns.append(F.coalesce(cols[-1], F.lit("*")))
    return F.concat(*concat_columns)


def create_dump_license_data(
    spark: SparkSession,
    input_catalogue_path: str = "gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/EXPANDED_LICENSES_FILTERED_ACCESS.jsonl.gz",
):
    """Function that takes the type of licenses to verify in which one needs to grant the necessary credits and deliver the dataframe with this data

    Parameters
    ----------
    spark : SparkSession
        Necessary SparkSession
    input_catalogue_path : str
        Ubication of the path
    save_as : str
        Format to save the file
    Returns
    -------
    Str
        Status of the file generation
    """
    data_license = (
        spark.read.format("json").schema(ARCHIVE_ORG_SCHEMA).load(input_catalogue_path)
    )
    ##Filter by necessary columns
    columns = [
        data_license.metadata.licenseurl,
        data_license.metadata.creator,
        data_license.metadata.title,
        data_license.metadata.credits,
    ]
    data_license = data_license.select(columns)
    ##Rename columns
    data_license = (
        data_license.withColumnRenamed("metadata.licenseurl", "licenseurl")
        .withColumnRenamed("metadata.creator", "creator")
        .withColumnRenamed("metadata.title", "title")
        .withColumnRenamed("metadata.credits", "credits")
    )
    ##There only 4 register without license at the moment. Without information in the rest of the data
    data_license = data_license.dropna(subset=["licenseurl"])
    ##Regex filter to search any kind of "by" license
    # Not enough. We need to detect CC-BY-SA data as well.
    regexp = r"(http|https)://creativecommons.org/licenses/(by|by-sa)/(1[.]0|2[.]0|2[.]5|3[.]0|4[.]0)"
    data_license = data_license.filter(data_license["licenseurl"].rlike(regexp))
    return data_license


def save_dump_license_data(data_license, save_as: str = "csv"):
    """Function that takes the table with the neccesary credits of licenses and deliver the file (cvs, txt, etc)

    Parameters
    ----------
    data_license : SparkSession
        Data with the neccesary credicts by CC-BY licenses
    -------
    Str
        Status of the file generation
    """
    if save_as == "csv":
        data_license.write.mode("overwrite").format("csv").option("header", "true").save("cc_by_licenses.csv")
    elif save_as == "txt":
        data_license = data_license.withColumn(
            "credits", my_concat(*data_license.columns)
        ).select("credits")
        data_license.coalesce(1).write.format("text").option("header", "false").mode(
            "append"
        ).save("credits.txt")
    else:
        return "This format to save is not allowed", 0
    return "save file successful"


def main(argv):
    spark = SparkSession.builder.appName("CC-BY-license").getOrCreate()
    data_license = create_dump_license_data(
        spark, FLAGS.input_catalogue_path
    )
    save_dump_license_data(data_license, FLAGS.save_as)


if __name__ == "__main__":
    app.run(main)
