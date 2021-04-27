import concurrent.futures
from argparse import ArgumentParser
import logging
import csv
import os
import json
from google.cloud import storage
from smart_open import open
import audiofile

logger = logging.getLogger(__name__)
storage_client = storage.Client()

def main():
    parser = ArgumentParser("Add duration metadata to data.")

    parser.add_argument("-i", "--input-path",
        default = "gs://the-peoples-speech-aws-import/librispeech-formatted/train-clean-100-clean-metadata.csv",
        help = "The output path to load the dataset from.")
    parser.add_argument("-o", "--output-path",
        default = "gs://the-peoples-speech-aws-import/librispeech-formatted/train-clean-100-clean-metadata-durations.csv",
        help = "The output path to save dataset with metadata.")
    parser.add_argument("--worker-count", default = 8,
        help = "The number of worker threads.")
    parser.add_argument("--cache-directory", default = "data",
        help = "The local path to cache.")
    parser.add_argument("-b", "--batch-size", default = 256,
        help = "The number of audio files to process at one time.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    arguments["system"] = {"cache-directory" : arguments["cache_directory"]}

    add_duration_metadata(arguments)

def add_duration_metadata(arguments):
    samples = load_csv(arguments["input_path"])

    with open(arguments["output_path"], "w", newline="") as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=',', quotechar='"')
        converter = AudioConverter(arguments, samples, csv_writer)

        converter.run()

def load_csv(csv_path):
    new_samples = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')

        for row in reader:
            path, caption = row[0], row[1]

            metadata = {}
            if len(row) >= 3:
                if len(row[2]) > 0:
                    metadata = json.loads(row[2])

            yield {"path" : path, "caption" : caption, "metadata" : metadata}

class AudioConverter:
    def __init__(self, arguments, samples, csv_writer):
        self.arguments = arguments
        self.samples = samples
        self.csv_writer = csv_writer

    def run(self):
        total_duration = 0.0
        total_bytes = 0.0
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(self.arguments["worker_count"])) as executor:
            while True:
                sample_batch = self.get_next_batch()

                if len(sample_batch) == 0:
                    break

                future_to_data = {executor.submit(add_duration, self.arguments, path, metadata): (path, transcript, metadata) for
                    path, transcript, metadata in sample_batch}

                for future in concurrent.futures.as_completed(future_to_data):
                    path, transcript, metadata = future_to_data[future]
                    try:
                        new_metadata = future.result()
                        self.csv_writer.writerow([path, transcript, json.dumps(new_metadata)])

                        duration = new_metadata["duration_seconds"]
                        total_duration += duration

                        byte_count = new_metadata["size_in_bytes"]
                        total_bytes += new_metadata["size_in_bytes"]

                        logger.debug(" duration is %s seconds out of total %s (%s / %s bytes) with audio %s" %
                            (sizeof_fmt(duration), sizeof_fmt(total_duration / 3600.0, suffix='h'),
                             sizeof_fmt(byte_count, suffix='B'),
                             sizeof_fmt(total_bytes, suffix='B'), path))
                    except Exception as exc:
                        print('%r generated an exception: %s' % (path, exc))

    def get_next_batch(self):
        batch = []

        for i in range(int(self.arguments["batch_size"])):
            try:
                row = next(self.samples)

                path = row["path"]
                transcript = row["caption"]
                metadata = row["metadata"]

                batch.append((path, transcript, metadata))
            except StopIteration:
                break


        return batch

def add_duration(arguments, path, metadata):
    local_path = LocalFileCache(arguments, path).get_path()
    audio_duration = audiofile.duration(local_path)
    metadata["duration_seconds"] = audio_duration
    metadata["size_in_bytes"] = os.path.getsize(local_path)

    os.remove(local_path)
    return metadata

def sizeof_fmt(num, suffix='s'):
    num = float(num)
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def setup_logger(arguments):

   if arguments["verbose"]:
       logger.setLevel(logging.DEBUG)
   else:
       logger.setLevel(logging.INFO)

   ch = logging.StreamHandler()
   ch.setLevel(logging.DEBUG)

   # create formatter
   formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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

if __name__ == "__main__":
    main()





