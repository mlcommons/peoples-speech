from argparse import ArgumentParser

import logging
import csv
import json
import os
import random

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("This program makes a training and validation dataset "
        "based on speaker id from the processed output of DSAlign.")

    parser.add_argument("-i", "--input-path", default = "",
        help = "The input path to the metadata file.")
    parser.add_argument("-c", "--speaker-count", default = 100,
        help = "The number of speakers to put in the test set.")
    parser.add_argument("--cache-directory", default = "data",
        help = "The local path to cache.")
    parser.add_argument("-ot", "--output-training-path", default = "",
        help = "The output path to save the training csv.")
    parser.add_argument("-ov", "--output-validation-path", default = "",
        help = "The output path to save the validation csv.")
    parser.add_argument("-v", "--verbose", default = False,
        action="store_true",
        help = "Set the log level to debug, "
            "printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    arguments["system"] = {"cache-directory" : arguments["cache_directory"]}

    setup_logger(arguments)

    make_training_and_test_set(arguments)


def make_training_and_test_set(arguments):

    data = load_data(arguments)

    speakers = group_by_speaker(data)

    generator = random.Random(42)

    generator.shuffle(speakers)

    test_speaker_count = min(int(arguments["speaker_count"]), len(speakers) - 1)

    training_speakers = speakers[test_speaker_count:]
    test_speakers = speakers[:test_speaker_count]

    with open(arguments["output_training_path"], "w", newline="") as output_training_csv_file:
        training_csv_writer = csv.writer(output_training_csv_file, delimiter=',', quotechar='"')
        for data in training_speakers:
            for path, item in data:
                training_csv_writer.writerow([path, item["aligned"]])

    with open(arguments["output_validation_path"], "w", newline="") as output_validation_csv_file:
        validation_csv_writer = csv.writer(output_validation_csv_file, delimiter=',', quotechar='"')
        for data in test_speakers:
            for path, item in data:
                validation_csv_writer.writerow([path, item["aligned"]])

def group_by_speaker(data):
    speakers = {}

    for path, metadata in data:
        speaker_id = metadata["meta"]["speaker"]

        if not speaker_id in speakers:
            speakers[speaker_id] = []

        speakers[speaker_id].append((path, metadata))

    return [speakers[speaker_id] for speaker_id in speakers.keys()]

def load_data(arguments):
    samples = []
    with open(arguments["input_path"]) as input_file:
        csv_reader = csv.reader(input_file, delimiter=',', quotechar='"')

        for path, data in csv_reader:
            samples.append((path, json.loads(data)))

    return samples

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

main()














