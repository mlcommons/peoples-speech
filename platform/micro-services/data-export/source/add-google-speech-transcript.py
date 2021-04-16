
from argparse import ArgumentParser
import logging
import csv
import os
import json
import tarfile
import io
from smart_open import open
from google.cloud import speech_v1 as speech

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser(".")

    parser.add_argument("-i", "--input-path", default = "gs://the-peoples-speech-west-europe/peoples-speech-v0.6/test.csv",
        help = "The output path to load the dataset from.")
    parser.add_argument("-o", "--output-path", default = "gs://the-peoples-speech-west-europe/peoples-speech-v0.6/test-with-transcripts.csv",
        help = "The output path to save new transcripts.")
    parser.add_argument("-v", "--verbose", default = False, action="store_true",
        help = "Set the log level to debug, printing out detailed messages during execution.")

    arguments = vars(parser.parse_args())

    setup_logger(arguments)

    add_transcripts(arguments)

def add_transcripts(arguments):
    samples = load_csv(arguments["input_path"])

    updated_samples = update_samples(samples)

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

def update_samples(samples):
    for sample in samples:
        metadata = sample["metadata"]
        logger.debug("For " + sample["path"])
        metadata["google-speech-api-transcript"] = get_transcript(sample["path"])
        logger.debug("Transcribed '" + sample["caption"] + "' to '" + metadata["google-speech-api-transcript"] + "'")
        yield (sample["path"], sample["caption"], json.dumps(metadata))

speech_client = speech.SpeechClient()

def get_transcript(path):

    # The language of the supplied audio
    language_code = "en-US"

    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = 16000

    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.
    #encoding = speech.RecognitionConfig.AudioEncoding.MP3
    config = {
        "language_code": language_code,
        #"enable_separate_recognition_per_channel": True,
        #"audio_channel_count": 2,
        "sample_rate_hertz": sample_rate_hertz,
    #    "encoding": encoding,
    }
    audio = {"uri": path}

    response = speech_client.recognize(config=config, audio=audio)

    transcript = ""

    for result in response.results:
        # First alternative is the most probable result
        alternative = result.alternatives[0]
        transcript += alternative.transcript

    return transcript



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


