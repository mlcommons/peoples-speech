import concurrent.futures
from argparse import ArgumentParser
import logging
import csv
import os
import json
import random
from pydub import AudioSegment

logger = logging.getLogger(__name__)

from smart_open import open

def make_splits(arguments):
    both_samples = []
    train_samples = []
    test_samples = []

    get_voicery_samples(both_samples)
    get_common_voice_train_samples(train_samples)
    get_common_voice_test_samples(test_samples)
    get_librispeech_train_samples(train_samples)
    get_librispeech_test_samples(test_samples)
    get_librivox_samples(both_samples)
    get_cc_search_samples(both_samples)

    train, test, development = split_samples(arguments,
        both_samples, train_samples, test_samples)

    save_samples(train, arguments["train_path"])
    save_samples(test, arguments["test_path"])
    save_samples(development, arguments["development_path"])

def get_common_voice_train_samples(samples):
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/common-voice/train-flac-clean.csv")

def get_common_voice_test_samples(samples):
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/common-voice/test-flac.csv")

def load_csv_samples(samples, csv_path):
    logger.debug("Loading samples from " + csv_path)
    new_samples = []
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=',', quotechar='"')

        for row in reader:
            path, caption = row[0], row[1]

            metadata = {}
            if len(row) >= 3:
                if len(row[2]) > 0:
                    metadata = json.loads(row[2])

            new_samples.append({"path" : path, "caption" : caption, "metadata" : metadata})

    logger.info("Loaded " + str(len(new_samples)) + " from " + csv_path)

    samples.extend(new_samples)

def get_librispeech_test_samples(samples):
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/dev-clean.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/dev-other.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/test-clean.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/test-other.csv")

def get_librispeech_train_samples(samples):
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/train-clean-100-clean-metadata.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/train-clean-360-clean-metadata.csv")
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librispeech-formatted/train-other-500-clean-metadata.csv")

def get_librivox_samples(samples):
    # 800 hours
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/librivox-v0.3-1B/data-flac.csv")

def get_voicery_samples(samples):
    #
    load_csv_samples(samples, "gs://the-peoples-speech-aws-import/voicery/data-clean-durations-flac.csv")

def get_cc_search_samples(samples):
    #
    load_csv_samples(samples, "gs://the-peoples-speech-west-europe/archive_org/v0.2/data-clean-duration-flac.csv")

def split_samples(arguments, both_samples, train_samples, test_samples):

    # id -> sample
    both_ids = group_samples_by_id(both_samples)
    train_ids = group_samples_by_id(train_samples)
    test_ids = group_samples_by_id(test_samples)

    # id -> sample
    test_and_both_ids = join_ids(both_ids, test_ids)

    test_set_size = min(len(test_and_both_ids), int(arguments["test_set_size"]))

    # id -> sample
    test = extract_samples(test_and_both_ids, test_set_size)
    development = extract_samples(test_and_both_ids, test_set_size)

    remaining_both_ids = remove_samples(both_ids, test)
    remaining_both_ids = remove_samples(remaining_both_ids, development)

    all_train_ids = join_ids(remaining_both_ids, train_ids)

    train = extract_samples(all_train_ids, len(all_train_ids))

    return drop_id(train), drop_id(test), drop_id(development)

def drop_id(dataset):
    return [value for key, value in dataset]

def join_ids(left, right):
    return left + right

def remove_samples(left, right):
    ids = set([key for key, value in right])

    return [(key, value) for key, value in left if not key in ids]

def get_id_for_sample(id_map, sample):
    return sample["path"]

def group_samples_by_id(samples):
    id_map = {}

    for sample in samples:
        sample_id = get_id_for_sample(id_map, sample)

        if not sample_id in id_map:
            id_map[sample_id] = []

        id_map[sample_id].append(sample)

    ids = stable_shuffle(id_map)

    return ids

def stable_shuffle(id_map):
    id_list = [(key, value) for key, value in id_map.items()]

    generator = random.Random(42)

    generator.shuffle(id_list)

    return flatten_id_list(id_list)

def flatten_id_list(id_list):
    new_samples = []

    for index, samples in id_list:
        for sample in samples:
            new_samples.append((index, sample))

    return new_samples

def extract_samples(ids, count):
    sample_count = min(len(ids), count)

    new_samples = ids[:sample_count].copy()

    del ids[:sample_count]

    return new_samples

def save_samples(samples, path):
    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='"')

        for sample in samples:
            writer.writerow([sample["path"], sample["caption"], json.dumps(sample["metadata"])])

def main():
    parser = ArgumentParser("Creates people's speech train, test, "
        "development splits.")

    parser.add_argument("--train-path", default = "gs://the-peoples-speech-west-europe/peoples-speech-v0.9/train.csv",
        help = "The output path to save the training dataset.")
    parser.add_argument("--test-path", default = "gs://the-peoples-speech-west-europe/peoples-speech-v0.9/test.csv",
        help = "The output path to save the test dataset.")
    parser.add_argument("--development-path", default = "gs://the-peoples-speech-west-europe/peoples-speech-v0.9/development.csv",
        help = "The output path to save the development dataset.")
    parser.add_argument("--test-set-size", default = 3000,
        help = "The number of samples to include in the test set.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    make_splits(arguments)

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



