from argparse import ArgumentParser
import logging
import csv
import json
import os
from google.cloud import storage
from smart_open import open
import concurrent.futures

from pydub import AudioSegment

logger = logging.getLogger(__name__)

storage_client = storage.Client()

def main():
    parser = ArgumentParser("Convert all of the audio files in a csv to the specified format, update the csv to point at the new files.")

    parser.add_argument("-i", "--input-path", default = "gs://the-peoples-speech-aws-import/common-voice/train.csv",
        help = "The path to the dataset stored in csv format.")
    parser.add_argument("--max-count", default = 1e9,
        help = "The maximum number of audio samples to extract.")
    parser.add_argument("--cache-directory", default = "data",
        help = "The local path to cache.")
    parser.add_argument("-o", "--output-path", default = "gs://the-peoples-speech-aws-import/common-voice/train-flac.csv",
        help = "The output path to save the dataset.")
    parser.add_argument("--worker-count", default = 8,
        help = "The number of worker threads.")
    parser.add_argument("--sampling-rate", default = 16000,
        help = "The sampling rate for the audio.")
    parser.add_argument("-f", "--format", default = "flac",
        help = "The audio format to convert to.")
    parser.add_argument("-b", "--batch-size", default = 256,
        help = "The number of audio files to process at one time.")
    parser.add_argument("-v", "--verbose", default = False,
        action="store_true",
        help = "Set the log level to debug, "
            "printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    arguments["system"] = {"cache-directory" : arguments["cache_directory"]}

    setup_logger(arguments)

    convert_audio_format(arguments)


def convert_audio_format(arguments):
    with open(arguments["output_path"], "w", newline="") as output_csv_file, \
         open(arguments["input_path"]) as input_csv_file:
        csv_reader = csv.reader(input_csv_file, delimiter=',', quotechar='"')
        csv_writer = csv.writer(output_csv_file, delimiter=',', quotechar='"')

        converter = AudioConverter(arguments, csv_reader, csv_writer)

        converter.run()

class AudioConverter:
    def __init__(self, arguments, csv_reader, csv_writer):
        self.arguments = arguments
        self.csv_reader = csv_reader
        self.csv_writer = csv_writer

    def run(self):
        total_bytes = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(self.arguments["worker_count"])) as executor:
            while True:
                sample_batch = self.get_next_batch()

                if len(sample_batch) == 0:
                    break

                future_to_data = {executor.submit(convert_file, self.arguments, path, updated_path): (updated_path, path, transcript, metadata) for
                    updated_path, path, transcript, metadata in sample_batch}

                for future in concurrent.futures.as_completed(future_to_data):
                    updated_path, path, transcript, metadata = future_to_data[future]
                    try:
                        byte_count, ratio = future.result()
                        total_bytes += byte_count
                        self.csv_writer.writerow([updated_path, transcript, metadata])

                        logger.debug("converted %s (%sx) / %s bytes from %s " % (sizeof_fmt(byte_count), ratio, sizeof_fmt(total_bytes), path))
                    except Exception as exc:
                        print('%r generated an exception: %s' % (path, exc))

    def get_next_batch(self):
        batch = []

        for i in range(int(self.arguments["batch_size"])):
            try:
                row = next(self.csv_reader)

                path = row[0]
                transcript = row[1]
                metadata = ""

                if len(row) > 2:
                    metadata = row[2]

                updated_path = update_path(path, self.arguments["format"])

                batch.append((updated_path, path, transcript, metadata))
            except StopIteration:
                break


        return batch

def sizeof_fmt(num, suffix='B'):
    num = float(num)
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def convert_file(config, path, updated_path):
    if blob_exists(updated_path):
        logger.debug("Skipping existing file '" + updated_path + "'")
        return 1, 1

    local_path = LocalFileCache(config, path).get_path()
    updated_local_path = update_path(local_path, get_format(updated_path))

    logger.debug("Converting from " + local_path + " to " + updated_local_path + " ( " + updated_path + " )")

    audio = AudioSegment.from_file(local_path, os.path.splitext(local_path)[1][1:])
    audio.set_frame_rate(int(config["sampling_rate"]))

    audio.export(updated_local_path, format=get_format(updated_path), parameters=["-compression_level", "4", "-ac", "1"])

    # upload the file
    bucket_name, key = get_bucket_and_prefix(updated_path)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(key)
    blob.upload_from_filename(local_path)

    original_size = os.path.getsize(local_path)
    updated_size = os.path.getsize(updated_local_path)

    os.remove(local_path)
    os.remove(updated_local_path)

    return updated_size, updated_size / original_size

def blob_exists(path):
    bucket_name, prefix = get_bucket_and_prefix(path)
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(prefix)
        return blob.exists()
    except Exception as e:
        logger.debug(path + " does not exist: exception: " + str(e))
        return False

    return True

def get_format(path):
    pre, ext = os.path.splitext(path)

    return ext[1:]

def update_path(path, format_name):
    pre, ext = os.path.splitext(path)

    return pre + "." + format_name

def get_bucket_and_prefix(path):
    parts = split_all(path[5:])

    return parts[0], os.path.join(*parts[1:])

def split_all(path):
    allparts = []
    while True:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts

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
    """ Supports caching.  Currently it supports read-only access to GCS.
    """

    def __init__(self, config, remote_path, create=True):
        self.config = config
        self.remote_path = remote_path

        self.local_path = self.compute_local_path()

        if create:
            self.download_if_remote()

    def get_path(self):
        return self.local_path

    def download_if_remote(self):
        if not self.is_remote_path(self.remote_path):
            return

        self.download()

    def download(self):
        if os.path.exists(self.get_path()):
            logger.info(" using cached file '" + self.get_path() + "'")
            return

        directory = os.path.dirname(self.get_path())

        os.makedirs(directory, exist_ok=True)

        bucket, key = get_bucket_and_prefix(self.remote_path)

        logger.info(
            "Downloading '" + self.remote_path + "' to '" + self.get_path() + "'"
        )

        bucket = storage_client.get_bucket(bucket)
        blob = bucket.get_blob(key)

        blob.download_to_filename(self.local_path)

    def is_remote_path(self, path):
        return path.find("gs:") == 0

    def compute_local_path(self):
        if not self.is_remote_path(self.remote_path):
            return self.remote_path
        bucket, key = get_bucket_and_prefix(self.remote_path)
        return os.path.join(self.config["system"]["cache-directory"], key)




main()














