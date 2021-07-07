from argparse import ArgumentParser
import logging
import csv
import json
import os
import boto3
import botocore
import urllib.parse

from pydub import AudioSegment

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("This program converts the DSAlign "
        "data format to the default CSV audio file format.")

    parser.add_argument("-i", "--input-path", default = "",
        help = "The S3 path to the DSAligned dataset.")
    parser.add_argument("--max-count", default = 1e9,
        help = "The maximum number of audio samples to extract.")
    parser.add_argument("--cache-directory", default = "data",
        help = "The local path to cache.")
    parser.add_argument("-o", "--output-path", default = "",
        help = "The output path to save the dataset.")
    parser.add_argument("-v", "--verbose", default = False,
        action="store_true",
        help = "Set the log level to debug, "
            "printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    arguments["system"] = {"cache-directory" : arguments["cache_directory"]}

    setup_logger(arguments)

    convert_dsalign_to_csv(arguments)


def convert_dsalign_to_csv(arguments):

    directory = arguments["output_path"]

    logger.debug("Checking directory: " + directory)
    if not os.path.exists(directory):
        logger.debug("Making directory: " + directory)
        os.makedirs(directory)

    with open(os.path.join(arguments["output_path"], "data.csv"), "w", newline="") as output_csv_file, \
        open(os.path.join(arguments["output_path"], "metadata.csv"), "w", newline="") as metadata_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=',', quotechar='"')
        metadata_writer = csv.writer(metadata_csv_file, delimiter=',', quotechar='"')
        update_csv(arguments, csv_writer, metadata_writer)

def update_csv(arguments, csv_writer, metadata_writer):

    total_count = 0

    for bucket_name, file_name in get_all_object_paths(arguments):
        if not is_audio(file_name):
            continue

        aligned_file_name = get_corresponding_align_file_name(file_name)

        if not exists(bucket_name, aligned_file_name):
            continue

        logger.debug("Extracting alignments from " + str(aligned_file_name) + ", " + str(file_name))

        alignment = load_alignment(bucket_name, aligned_file_name, arguments)

        audio = load_audio(bucket_name, file_name, arguments)

        for entry in alignment:
            start = entry["start"]
            end = entry["end"]

            text = entry["aligned"]

            audio_segment = extract_audio(audio, start, end)

            save_training_sample(csv_writer, metadata_writer, audio_segment, text, entry, arguments, total_count)

            total_count += 1

            if total_count >= int(arguments["max_count"]):
                return

def is_audio(path):
    return os.path.splitext(path)[1] == '.mp3'

def get_corresponding_align_file_name(path):
    return os.path.splitext(path)[0] + ".aligned"

def exists(bucket_name, path):

    client = boto3.resource('s3')
    try:
        client.Object(bucket_name, path).load()
    except botocore.exceptions.ClientError as e:
        return False

    return True

def load_alignment(bucket_name, path, arguments):
    local_cache = LocalFileCache(arguments, "s3://" + os.path.join(bucket_name, path)).get_path()
    with open(local_cache) as json_file:
        return json.load(json_file)

def load_audio(bucket_name, path, arguments):
    local_cache = LocalFileCache(arguments, "s3://" + os.path.join(bucket_name, path)).get_path()

    return AudioSegment.from_mp3(local_cache)

def extract_audio(audio, start, end):
    return audio[start:end]

def save_training_sample(csv_writer, metadata_writer, audio_segment, text, entry, arguments, total_count):
    path = get_output_path(arguments, total_count)

    directory = os.path.dirname(path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    logger.debug("Saving sample: " + path)

    audio_segment.export(path, format="mp3")

    csv_writer.writerow([path, text])
    metadata_writer.writerow([path, json.dumps(entry)])

def get_output_path(arguments, total_count):
    return os.path.join(arguments["output_path"], "data", str(total_count) + ".mp3")

def get_all_object_paths(arguments):

    client = boto3.client('s3')
    paginator = client.get_paginator('list_objects_v2')

    bucket_name, prefix_name = get_bucket_and_prefix_name(arguments["input_path"])

    page_iterator = paginator.paginate(Bucket = bucket_name, Prefix = prefix_name)
    for page in page_iterator:
        logger.debug("Iterating through page with " + str(page["KeyCount"]) + " objects")

        for page_object in page['Contents']:
            yield bucket_name, page_object['Key']

def get_bucket_and_prefix_name(path):
    result = urllib.parse.urlparse(path, allow_fragments=False)

    return result.netloc, result.path.lstrip("/")

def setup_logger(arguments):

    if arguments["verbose"]:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s -'
        ' %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)




class LocalFileCache:
    """ Supports caching.  Currently it supports read-only access to S3.

        TODO: match the MD5 hash of the object against S3 before downloading again
    """

    def __init__(self, config, remote_path):
        self.config = config
        self.remote_path = remote_path

        self.download_if_remote()

    def get_path(self):
        return self.local_path

    def download_if_remote(self):
        if not self.is_remote_path(self.remote_path):
            self.local_path = self.remote_path
            return

        self.local_path = self.compute_local_path()

        self.download()

    def download(self):
        if os.path.exists(self.get_path()):
            logger.info(" using cached file '" + self.get_path() + "'")
            return

        directory = os.path.dirname(self.get_path())

        os.makedirs(directory, exist_ok=True)

        s3 = boto3.client("s3")

        bucket, key = self.get_bucket_and_key()

        logger.info(
            "Downloading '" + self.remote_path + "' to '" + self.get_path() + "'"
        )

        s3.download_file(bucket, key, self.get_path())

    def is_remote_path(self, path):
        return path.find("s3:") == 0

    def compute_local_path(self):
        bucket, key = self.get_bucket_and_key()
        if key.endswith(".engine") or key.endswith(".mp3"):
            return self.compute_s3_resource_local_path(key)
        return os.path.join(self.config["system"]["cache-directory"], key)

    def compute_s3_resource_local_path(self, key):
        repo_dir = "/".join(os.path.abspath(__file__).split("/")[:-3])
        path = os.path.join(repo_dir, "models", key)
        return path

    def get_bucket_and_key(self):
        return self.remote_path.split("/", 2)[-1].split("/", 1)



main()













