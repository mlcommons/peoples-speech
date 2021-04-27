from argparse import ArgumentParser
import logging
import csv
import os
import json
from smart_open import open
from cleantext import clean

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("Clean text.")

    parser.add_argument("-i", "--input-path", default = "gs://the-peoples-speech-west-europe/peoples-speech-v0.6/test.csv",
        help = "The output path to load the dataset from.")
    parser.add_argument("-o", "--output-path", default = "gs://the-peoples-speech-west-europe/peoples-speech-v0.6/test-cleaned-transcripts.csv",
        help = "The output path tosave cleaned transcripts.")
    parser.add_argument("--to-lower", default = False, action="store_true",
        help = "Lower case.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    clean_transcripts(arguments)

def clean_transcripts(arguments):
    samples = load_csv(arguments["input_path"])

    updated_samples = update_samples(samples, arguments)

    with open(arguments["output_path"], "w", newline="") as output_csv_file:
        csv_writer = csv.writer(output_csv_file, delimiter=',', quotechar='"')
        for sample in updated_samples:
            csv_writer.writerow(sample)

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

def update_samples(samples, arguments):
    for sample in samples:
        cleaned_caption = clean(sample["caption"], lower=arguments["to_lower"], no_line_breaks=True).strip()
        metadata = sample["metadata"]
        logger.debug("For " + sample["path"])
        metadata["pre-cleaned-transcript"] = sample["caption"]
        logger.debug("Cleaned transcript from '" + sample["caption"] + "' to '" + cleaned_caption + "'")
        yield (sample["path"], cleaned_caption, json.dumps(metadata))

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



