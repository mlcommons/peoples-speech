import concurrent.futures
from google.cloud import storage
from argparse import ArgumentParser
import logging
import csv
import os
import json
import random
from pydub import AudioSegment

logger = logging.getLogger(__name__)

from smart_open import open

def main():
    parser = ArgumentParser("Creates CSV file for voicery samples.")

    parser.add_argument("--input-path", default = "gs://the-peoples-speech-aws-import/voicery",
        help = "The input path to search for data.")
    parser.add_argument("--output-path", default = "gs://the-peoples-speech-aws-import/voicery/data.csv",
        help = "The output path to save the training dataset.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    make_csv(arguments)

def make_csv(arguments):
    samples = []

    get_voicery_samples(samples, arguments["input_path"])

    save_samples(samples, arguments["output_path"])

def get_voicery_samples(samples, input_path):
    mp3_files = get_mp3_files(input_path)

    new_samples = []

    # We can use a with statement to ensure threads are cleaned up promptly
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        # Start the load operations and mark each future with its URL
        future_to_transcript = {executor.submit(get_voicery_transcript, path): (path, name) for path, name in mp3_files.items()}
        for future in concurrent.futures.as_completed(future_to_transcript):
            path, name = future_to_transcript[future]
            try:
                transcript = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (path, exc))
            #else:
            #    logger.debug("transcript is " + transcript)

            new_samples.append({"path" : path, "caption" : transcript, "metadata": {"speaker_id" : "voicery_" + name}})

            if len(new_samples) % 1000 == 0:
                logger.debug(" loaded " + str(len(new_samples)) + " transcripts")

    samples.extend(new_samples)

storage_client = storage.Client()

def get_voicery_transcript(path):
    base = os.path.splitext(path)[0]

    normalized_path = base + ".normalized.txt"

    bucket_name, prefix = get_bucket_and_prefix(normalized_path)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.get_blob(prefix)

    return blob.download_as_text().strip()

def get_mp3_files(audio_path):
    logger.debug("Getting MP3 files under " + audio_path)

    # Note: Client.list_blobs requires at least package version 1.17.0.
    bucket_name, prefix = get_bucket_and_prefix(audio_path)
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    mp3_files = {}

    for blob in blobs:
        if is_mp3(blob.name):
            path = os.path.join("gs://" + bucket_name, blob.name)
            mp3_files[path] = get_key(blob.name)

    logger.debug(" Found " + str(len(mp3_files)) + " mp3 files")

    return mp3_files

def get_key(path):
    parts = split_all(path)
    return os.path.splitext(parts[-2] + "-" + parts[-1])[0]

def is_mp3(path):
    return os.path.splitext(path)[1] == ".mp3" or os.path.splitext(path)[1] == ".wav"

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

def save_samples(samples, path):
    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"')

        for sample in samples:
            writer.writerow([sample["path"], sample["caption"], json.dumps(sample["metadata"])])

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

if __name__ == "__main__":
    main()




