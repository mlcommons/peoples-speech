
from argparse import ArgumentParser
import logging
import csv
import json
import os
from datetime import datetime

from pydub import AudioSegment

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("This program converts the DiPCo data format to "
        "the default CSV audio file format.")

    parser.add_argument("-i", "--input-path", default = "",
        help = "The path to the DiPCo dataset.")
    parser.add_argument("-o", "--output-path", default = "",
        help = "The output path to save the dataset.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out "
               "detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    convert_dipco_to_csv(arguments)

def convert_dipco_to_csv(arguments):
    all_files = get_all_files(arguments)

    if not os.path.exists(arguments["output_path"]):
        os.makedirs(arguments["output_path"])

    train_csv_path = os.path.join(arguments["output_path"], "train.csv")
    test_csv_path = os.path.join(arguments["output_path"], "test.csv")

    with open(train_csv_path, "w") as output_train_file, \
         open(test_csv_path,  "w") as output_test_file:
        train_writer = csv.writer(output_train_file, delimiter=',', quotechar='"')
        test_writer  = csv.writer(output_test_file,  delimiter=',', quotechar='"')

        for index, filename in enumerate(all_files):
            rows = json.load(open(filename))

            for row in rows:
                items = expand_row(row, filename)

                for item in items:
                    new_filename, label = copy_data(item, arguments)

                    if is_test(filename):
                        test_writer.writerow([new_filename, label])
                    else:
                        train_writer.writerow([new_filename, label])

def expand_row(row, filename):
    results = []

    label = row["words"]

    for key, start_time in row["start_time"].items():
        end_time = row["end_time"][key]

        if key == 'close-talk':
            results.append((filename, label, row["speaker_id"], start_time, end_time))
        else:
            for i in range(1,8):
                results.append((filename, label, key + ".CH" + str(i), start_time, end_time))

    return results

def copy_data(item, arguments):
    filename, label, key, start_time, end_time = item

    source_filename = get_source_filename(filename, key)

    target_filename = get_target_filename(arguments, filename, key, start_time)

    extract_wav(target_filename, source_filename, start_time, end_time)

    return (target_filename, label)

def get_source_filename(filename, key):
    s_id = get_s_id(filename)
    modifier = get_modifier(key)

    base = get_base(filename)

    return os.path.join(base, s_id + modifier + ".wav")

def get_s_id(filename):
    return os.path.splitext(os.path.basename(filename))[0]

def get_modifier(key):
    return "_" + key

def get_base(filename):
    directory_switch = "eval" if is_test(filename) else "dev"

    base, tail = os.path.split(filename)

    while tail != 'transcriptions' and base != '':
        base, tail = os.path.split(base)

    return os.path.join(base, "audio", directory_switch)

def get_target_filename(arguments, filename, key, start_time):
    directory_switch = "eval" if is_test(filename) else "dev"
    base = os.path.join(arguments["output_path"], directory_switch)

    if not os.path.exists(base):
        os.makedirs(base)

    s_id = get_s_id(filename)
    modifier = get_modifier(key)
    timestamp = start_time.replace(":", "_")
    return os.path.join(base, s_id + modifier + timestamp + ".wav")

def extract_wav(target_filename, source_filename, start_time, end_time):
    logger.debug("extracting data from " + source_filename +
        " [" + start_time + ", " + end_time + "]")

    wav = AudioSegment.from_wav(source_filename)

    start_milliseconds = get_milliseconds(start_time)
    end_milliseconds   = get_milliseconds(  end_time)

    wav_slice = wav[start_milliseconds:end_milliseconds]

    wav_slice.export(target_filename, format='wav')

def get_milliseconds(time):
    split_time = time.split(":")

    seconds = float(split_time[0]) * 3600 + float(split_time[1]) * 60 + float(split_time[2])

    return seconds * 1000

def is_test(filename):
    return filename.find("eval") != -1

def get_all_files(arguments):
    all_files = []

    for root, directories, files in os.walk(arguments["input_path"]):
        all_files += [os.path.join(root, f) for f in files if is_json(f)]

    return sorted(all_files)

def is_json(filename):
    return os.path.splitext(filename)[1] == ".json"

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












