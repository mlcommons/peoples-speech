from argparse import ArgumentParser
import logging
import csv
import os
import json
from smart_open import open

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("Add metadata to librispeech data.")

    parser.add_argument("-i", "--input-path", default = "gs://the-peoples-speech-aws-import/librispeech-formatted/train-clean-100.csv",
        help = "The output path to load the dataset from.")
    parser.add_argument("-m", "--metadata-path", default = "gs://the-peoples-speech-aws-import/librispeech-formatted/SPEAKERS.txt",
        help = "The path to load the metadata from.")
    parser.add_argument("-o", "--output-path", default = "gs://the-peoples-speech-aws-import/librispeech-formatted/train-clean-100-metadata.csv",
        help = "The output path to save dataset with metadata.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    add_metadata(arguments)

def add_metadata(arguments):
    samples = load_csv(arguments["input_path"])
    metadata = load_metadata(arguments["metadata_path"])

    updated_samples = update_samples(samples, metadata)

    with open(arguments["output_path"], "w", newline="") as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=',', quotechar='"')
        for sample in updated_samples:
            csv_writer.writerow(sample)

def decomment(csvfile):
    for row in csvfile:
        raw = row.split(';')[0].strip()
        if raw: yield raw

def load_metadata(speakers_path):
    metadata = {}
    with open(speakers_path) as speakers_file:
        csv_reader = csv.reader(decomment(speakers_file), delimiter='|', quotechar='"')

        #14   | F | train-clean-360  | 25.03 | Kristin LeMoine
        #60   | M | train-clean-100  | 20.18 | |CBW|Simon
        for row in csv_reader:
            if len(row) > 5:
                row[4] = "|".join(row[4:])

            speaker_id = row[0].strip()
            gender = row[1].strip()
            dataset_name = row[2].strip()
            hours = row[3].strip()
            name = row[4].strip()

            metadata[speaker_id] = {"data_source" : "librispeech",
                "speaker_id" : speaker_id, "gender" : gender,
                "librispeech_split_name" : dataset_name,
                "hours_per_speaker" : hours, "speaker_name" : name}

            logger.debug("Found metadata for speaker id: '" + str(speaker_id) + "': " + str(metadata[speaker_id]))

    return metadata

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

def update_samples(samples, metadata):
    for sample in samples:
        name = get_speaker_id_for_sample(sample["path"])
        logger.debug("For " + sample["path"])
        sample_metadata = metadata[name]
        logger.debug("Added metadata " + str(sample_metadata))
        yield (sample["path"], sample["caption"], json.dumps(sample_metadata))

def get_speaker_id_for_sample(path):
    basename = os.path.basename(path)
    return basename.split("-")[0]

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




