
from argparse import ArgumentParser
import logging
import csv
import json
import os
import shutil

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("This program converts the LibriSpeech "
        "data format to the default CSV audio file format.")

    parser.add_argument("-i", "--input-path", default = "",
        help = "The path to the LibriSpeech dataset.")
    parser.add_argument("-o", "--output-path", default = "",
        help = "The output path to save the dataset.")
    parser.add_argument("-v", "--verbose", default = False,
        action="store_true",
        help = "Set the log level to debug, "
            "printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    convert_librispeech_to_csv(arguments)

def convert_librispeech_to_csv(arguments):

    convert_to_csv("dev-clean", arguments)
    convert_to_csv("dev-other", arguments)
    convert_to_csv("test-clean", arguments)
    convert_to_csv("test-other", arguments)
    convert_to_csv("train-clean-100", arguments)
    convert_to_csv("train-clean-360", arguments)
    convert_to_csv("train-other-500", arguments)

def convert_to_csv(name, arguments):

    path = os.path.join(arguments["input_path"], name)

    all_files = get_all_files(path)

    if not os.path.exists(arguments["output_path"]):
        os.makedirs(arguments["output_path"])

    csv_path = os.path.join(arguments["output_path"], name + ".csv")

    with open(csv_path, "w") as output_file:
        writer = csv.writer(output_file, delimiter=',', quotechar='"')

        for filename in all_files:
            base_directory = os.path.dirname(filename)

            with open(filename, "r") as label_file:
                for line in label_file:
                    tokens = line.split(" ")
                    audio_file = os.path.join(base_directory, tokens[0] + ".flac")
                    label = " ".join(tokens[1:]).strip()

                    writer.writerow([audio_file, label])

def is_txt(filename):
    return os.path.splitext(filename)[1] == ".txt"

def get_all_files(path):
    all_files = []

    for root, directories, files in os.walk(path):
        all_files += [os.path.join(root, f) for f in files if is_txt(f)]

    return sorted(all_files)

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












