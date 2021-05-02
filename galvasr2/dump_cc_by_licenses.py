#!/usr/bin/env python

from align.spark.schemas import ARCHIVE_ORG_SCHEMA
from pyspark.sql import SparkSession
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('input_catalogue_path',
                   'gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/EXPANDED_LICENSES_FILTERED_ACCESS.jsonl.gz',
                   'Ubication of the path with the licence metadata')
flags.DEFINE_string('save_as',
                   'csv',
                   'Format to save the file')

def myConcat(*cols):
    """Generate a format that allows import a Spark df as a one column txt
    
        Parameters
        ----------
        *cols : list
            columns
        Returns
        -------
        Spark df
            Data in the format needed to save as a txt
    """
    concat_columns = []
    for c in cols[:-1]:
        concat_columns.append(F.coalesce(c, F.lit("*")))
        concat_columns.append(F.lit(" "))  
    concat_columns.append(F.coalesce(cols[-1], F.lit("*")))
    return F.concat(*concat_columns)

def create_dump_license(spark:SparkSession, input_catalogue_path:str="gs://the-peoples-speech-west-europe/archive_org/Mar_7_2021/EXPANDED_LICENSES_FILTERED_ACCESS.jsonl.gz", save_as:str='csv'):
    """Function that takes the type of licenses to verify in which one needs to grant the necessary credits and deliver the file (cvs, txt, etc) with this data
    
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
    df = spark.read.format('json').schema(ARCHIVE_ORG_SCHEMA).load(input_catalogue_path)
    ##Filter by necessary columns 
    columns = [df.metadata.licenseurl, df.metadata.creator, df.metadata.title, df.metadata.credits]
    df = df.select(columns)
    ##Rename columns
    df = (df.withColumnRenamed('metadata.licenseurl','licenseurl').withColumnRenamed('metadata.creator', 'creator')
         .withColumnRenamed('metadata.title', 'title').withColumnRenamed('metadata.credits', 'credits'))
    ##There only 4 register without license at the moment. Without information in the rest of the data
    df = df.dropna(subset=['licenseurl'])
    ##Regex filter to search any kind of "by" license
    regexp = r'(http|https)://creativecommons.org/licenses/by/(1[.]0|2[.]0|2[.]5|3[.]0|4[.]0)'
    df = df.filter(df['licenseurl'].rlike(regexp))
    if save_as == 'csv':
        df.write.csv('cc_by_licenses.csv')
    elif save_as == 'txt':
        df = df.withColumn("credits", myConcat(*df.columns)).select("credits")
        df.coalesce(1).write.format("text").option("header", "false").mode("append").save("credits.txt")
    else:
        return 'This format to save is not allowed'
    return 'save file successful'

def main():
    spark = SparkSession.builder.appName('CC-BY-license').getOrCreate()
    create_dump_license(spark, FLAGS.input_catalogue_path, FLAGS.save_as)

if __name__ == '__main__':
    main()
        
        
