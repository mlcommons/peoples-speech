
from argparse import ArgumentParser
import logging
import csv
import json
import os
import shutil

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("This program converts the COMMON-VOICE data format to "
        "the default CSV audio file format.")

    parser.add_argument("-i", "--input-path", default = "",
        help = "The path to the COMMON-VOICE dataset.")
    parser.add_argument("-o", "--output-path", default = "",
        help = "The output path to save the dataset.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    convert_common_voice_to_csv(arguments)

def convert_common_voice_to_csv(arguments):
    all_files = get_all_files(arguments)

    if not os.path.exists(arguments["output_path"]):
        os.makedirs(arguments["output_path"])
        os.makedirs(os.path.join(arguments["output_path"], "clips"))

    train_csv_path = os.path.join(arguments["output_path"], "train.csv")
    test_csv_path  = os.path.join(arguments["output_path"], "test.csv")
    dev_csv_path   = os.path.join(arguments["output_path"], "dev.csv")

    with open(train_csv_path, "w") as output_train_file, \
         open(test_csv_path,  "w") as output_test_file, \
         open(dev_csv_path,  "w") as output_dev_file:
        train_writer = csv.writer(output_train_file, delimiter=',', quotechar='"')
        test_writer = csv.writer(output_test_file, delimiter=',', quotechar='"')
        dev_writer = csv.writer(output_test_file, delimiter=',', quotechar='"')

        for index, filename in enumerate(all_files):
            with open(filename) as input_file:
                reader = csv.reader(input_file, delimiter='\t', quotechar='"')

                next(reader)

                for row in reader:
                    label = row[2]
                    path = row[1]

                    base = os.path.split(filename)[0]

                    utterance_path = os.path.join(base, "clips", path)

                    new_filename = os.path.join(arguments["output_path"], "clips", path)

                    logger.debug("Copying from " + utterance_path + " to " + new_filename)
                    shutil.copy(utterance_path, new_filename)

                    if is_test(filename):
                        test_writer.writerow([new_filename, label])
                    elif is_dev(filename):
                        dev_writer.writerow([new_filename, label])
                    elif is_train(filename):
                        train_writer.writerow([new_filename, label])

def is_test(filename):
    return filename.find("test") != -1

def is_dev(filename):
    return filename.find("dev") != -1

def is_train(filename):
    return filename.find("train") != -1

def is_tsv(filename):
    return os.path.splitext(filename)[1] == ".tsv"

def get_all_files(arguments):
    all_files = []

    for root, directories, files in os.walk(arguments["input_path"]):
        all_files += [os.path.join(root, f) for f in files if is_tsv(f)]

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











