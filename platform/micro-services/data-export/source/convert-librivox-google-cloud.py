from argparse import ArgumentParser
import logging
import csv
import json
import os
import queue
import threading
import hashlib
import gc
from google.cloud import storage
from smart_open import open

from pydub import AudioSegment

logger = logging.getLogger(__name__)

storage_client = storage.Client()

def main():
    parser = ArgumentParser("This program converts the DSAlign "
        "data format to the default CSV audio file format.")

    parser.add_argument("-i", "--input-path", default = "gs://the-peoples-speech-aws-import/librivox-aligned/en",
        help = "The GCP path to the DSAligned dataset.")
    parser.add_argument("--max-count", default = 1e9,
        help = "The maximum number of audio samples to extract.")
    parser.add_argument("--cache-directory", default = "data",
        help = "The local path to cache.")
    parser.add_argument("-o", "--output-path", default = "gs://the-peoples-speech-aws-import/librivox-v0.2-1B",
        help = "The output path to save the dataset.")
    parser.add_argument("--worker-count", default = 4,
        help = "The number of worker threads.")
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

    with open(os.path.join(arguments["output_path"], "data.csv"), "w", newline="") as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=',', quotechar='"')
        update_csv(arguments, csv_writer)

def update_csv(arguments, csv_writer):
    mp3_files = dict(get_mp3_files(arguments["input_path"], arguments), **get_mp3_files(arguments["output_path"], arguments))

    total_count = 0

    file_uploader = FileUploader(arguments)

    for bucket_name, file_name in get_all_object_paths(arguments):
        if not is_audio(file_name):
            continue

        mp3_path = os.path.join("gs://" + bucket_name, file_name)

        aligned_file_name = get_corresponding_align_file_name(file_name)

        mp3_size = get_blob_size(mp3_path)

        if mp3_size > 250e6:
            logger.debug("Skipping mp3 from " + mp3_path + " with " + str(mp3_size / 1e6) + "MB which is too big")
            delete_cached_file(arguments, "gs://" + os.path.join(bucket_name, aligned_file_name))
            continue

        logger.debug("Extracting alignments from " + str(aligned_file_name) + ", " + str(file_name))

        alignments = load_alignments(bucket_name, aligned_file_name, arguments)

        if all_alignments_exist(arguments, mp3_path, alignments, mp3_files):
            logger.debug("Skipping mp3 from " + mp3_path + " which is already fully converted.")
            delete_cached_file(arguments, "gs://" + os.path.join(bucket_name, aligned_file_name))
            continue

        mp3 = load_audio(mp3_path, arguments)

        for entry in alignments:
            start = entry["start"]
            end = entry["end"]

            text = entry["aligned"]

            save_training_sample(mp3_files, file_uploader, csv_writer, mp3, mp3_path, start, end, text, entry, arguments, total_count)

            total_count += 1

            if total_count >= int(arguments["max_count"]):
                break

        del alignments

        delete_audio(mp3, mp3_path, aligned_file_name, arguments)

        if total_count >= int(arguments["max_count"]):
            break

    file_uploader.join()

def all_alignments_exist(arguments, mp3_path, alignments, mp3_files):
    for entry in alignments:
        start = entry["start"]
        end = entry["end"]

        path = get_output_path(arguments, mp3_path, start, end)

        if not blob_exists(mp3_files, path):
            return False

    return True

def get_mp3_files(audio_path, arguments):
    logger.debug("Getting MP3 files under " + audio_path)

    # Note: Client.list_blobs requires at least package version 1.17.0.
    bucket_name, prefix = get_bucket_and_prefix(audio_path)
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    mp3_files = {}

    for blob in blobs:
        if is_audio(blob.name):
            path = os.path.join("gs://" + bucket_name, blob.name)
            mp3_files[path] = get_key(blob.name)
            if len(mp3_files) > int(arguments["max_count"]):
                break

    logger.debug(" Found " + str(len(mp3_files)) + " mp3 files")

    return mp3_files

def get_key(path):
    parts = split_all(path)
    return os.path.splitext(parts[-2] + "-" + parts[-1])[0]

def is_audio(path):
    return os.path.splitext(path)[1] == '.mp3' or os.path.splitext(path)[1] == '.wav'

def get_corresponding_align_file_name(path):
    return os.path.splitext(path)[0] + ".aligned"

def blob_exists(paths, path):
    return path in paths

def get_blob_size(path):
    bucket_name, prefix = get_bucket_and_prefix(path)
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.get_blob(prefix)
        return blob.size
    except:
        return 1e9

def load_alignments(bucket_name, path, arguments):
    local_cache = LocalFileCache(arguments, "gs://" + os.path.join(bucket_name, path)).get_path()
    with open(local_cache) as json_file:
        return json.load(json_file)

def load_audio(path, arguments):
    return MP3File(path, arguments)

class MP3File:
    def __init__(self, path, arguments):
        self.path = path
        self.arguments = arguments
        self.mp3 = None

    def get(self):
        if self.mp3 is None:
            local_cache = LocalFileCache(self.arguments, self.path).get_path()

            self.mp3 = AudioSegment.from_mp3(local_cache)

        return self.mp3

def delete_audio(mp3, mp3_path, alignment_file_name, arguments):
    del mp3
    gc.collect()
    delete_cached_file(arguments, mp3_path)
    delete_cached_file(arguments, alignment_file_name)

def delete_cached_file(arguments, path):
    local_cache = LocalFileCache(arguments, path, create=False).get_path()
    logger.debug("Deleting cache file " + local_cache)
    if os.path.exists(local_cache):
        os.remove(local_cache)

def extract_audio(audio, start, end):
    return audio.get()[start:end]

def save_training_sample(mp3s, file_uploader, csv_writer, audio, input_path, start, end, text, entry, arguments, total_count):
    path = get_output_path(arguments, input_path, start, end)

    if not blob_exists(mp3s, path):
        local_path = get_local_path(arguments, input_path, start, end)

        directory = os.path.dirname(local_path)

        if not os.path.exists(directory):
            os.makedirs(directory)

        logger.debug("Saving sample: " + path + " at local path " + local_path)

        audio_segment = extract_audio(audio, start, end)

        audio_segment.export(local_path, format="wav")

        file_uploader.upload(path, local_path)

    csv_writer.writerow([path, text, json.dumps(entry)])

def get_output_path(arguments, input_path, start, end):
    return os.path.join(arguments["output_path"], "data", "audio-" + hash_function(input_path + "-" + str(start) + "-" + str(end)) + ".wav")

def hash_function(data):
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def get_local_path(arguments, input_path, start, end):
    bucket, key = get_bucket_and_prefix(get_output_path(arguments, input_path, start, end))
    return os.path.join(arguments["system"]["cache-directory"], key)

def get_all_object_paths(arguments):

    bucket_name, prefix = get_bucket_and_prefix(arguments["input_path"])
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    for blob in blobs:
        yield bucket_name, blob.name

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

class FileUploader:
    def __init__(self, arguments):
        self.queue = queue.Queue(maxsize=512)
        self.threads = []

        for i in range(int(arguments["worker_count"])):
            thread = threading.Thread(target=upload_files_worker, args=(self.queue,))
            thread.start()
            self.threads.append(thread)

    def upload(self, path, local_path):
        self.queue.put((path, local_path, False))

    def join(self):

        for thread in self.threads:
            self.queue.put((None, None, True))

        for thread in self.threads:
            thread.join()

def upload_files_worker(queue):
    while True:
        path, local_path, is_finished = queue.get()

        if is_finished:
            break

        logger.debug("Uploading " + local_path + " to " + path)

        try:
            bucket_name, key = get_bucket_and_prefix(path)
            bucket = storage_client.get_bucket(bucket_name)
            blob = bucket.blob(key)
            blob.upload_from_filename(local_path)
        except:
            pass

        os.remove(local_path)

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













