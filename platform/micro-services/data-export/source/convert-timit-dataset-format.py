
from argparse import ArgumentParser
import logging
import csv
import json
import os
import shutil

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("This program converts the TIMIT data format to "
        "the default CSV audio file format.")

    parser.add_argument("-i", "--input-path", default = "",
        help = "The path to the TIMIT dataset.")
    parser.add_argument("-o", "--output-path", default = "",
        help = "The output path to save the dataset.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    convert_timit_to_csv(arguments)

def convert_timit_to_csv(arguments):
    all_files = get_all_files(arguments)

    if not os.path.exists(arguments["output_path"]):
        os.makedirs(arguments["output_path"])

    train_csv_path = os.path.join(arguments["output_path"], "train.csv")
    test_csv_path = os.path.join(arguments["output_path"], "test.csv")

    with open(train_csv_path, "w") as output_train_file, \
         open(test_csv_path,  "w") as output_test_file:
        train_writer = csv.writer(output_train_file, delimiter=',', quotechar='"')
        test_writer = csv.writer(output_test_file, delimiter=',', quotechar='"')

        for index, filename in enumerate(all_files):
            label_filename = os.path.splitext(filename)[0] + '.TXT'

            new_filename = os.path.join(arguments["output_path"],
                str(index) + ".wav")

            shutil.copyfile(filename, new_filename)
            logger.debug("Copying " + filename + " to " + new_filename)

            label = get_label(label_filename)

            if is_test(label_filename):
                test_writer.writerow([new_filename, label])
            else:
                train_writer.writerow([new_filename, label])

def is_test(filename):
    return filename.find("TEST") != -1

def get_label(label_filename):

    with open(label_filename) as label_file:
        tokens = label_file.read().split(" ")

    return " ".join(tokens[2:]).strip()

def is_wav(filename):
    return os.path.splitext(filename)[1] == ".WAV"

def get_all_files(arguments):
    all_files = []

    for root, directories, files in os.walk(arguments["input_path"]):
        all_files += [os.path.join(root, f) for f in files if is_wav(f)]

    return sorted(all_files)

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

main()











