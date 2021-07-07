
import os

from argparse import ArgumentParser

import config
import jsonlines
import random
from smart_open import open

import logging

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("Randomly sample audio and transcripts from a json lines file.")

    subparsers = parser.add_subparsers()

    setup_cli_parser(subparsers)

    args = parser.parse_args()

    args.func(args)

def setup_cli_parser(subparsers):

    parser = subparsers.add_parser('dataset')

    parser.add_argument("-i", "--dataset-path",
        default="gs://the-peoples-speech-west-europe/forced-aligner/cuda-forced-aligner/output_work_dir_5b/output_work_dir_5b/dataset_manifest_mp3_956_all.json",
        help="Path to json lines file describing the dataset.")
    parser.add_argument("-o", "--output-dataset-path", default="data/dataset.json", help="The path to save the new dataset.")
    parser.add_argument("-c", "--config-file-path", default="", help="The path to the config file.")
    parser.add_argument("-m", "--maximum-samples", default=100, help="How many samples to download.")
    parser.add_argument("--maximum-dataset-size", default=1000, help="How many samples to scan.")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Print out debug messages.")
    parser.add_argument("-vi", "--verbose-info", default=False, action="store_true", help="Print out info messages.")

    parser.set_defaults(func=dispatch)

def dispatch(args):
    arguments = vars(args)

    config = setup_config(arguments)

    setup_logging(config)

    logger.debug("Full config: " + str(config))

    samples = load_samples(config)

    filtered_samples = filter_samples(samples, config)

    save_samples(filtered_samples, config)

def load_samples(config):
    samples = []

    with open(config["dataset_path"]) as dataset_file:

        dataset_reader = jsonlines.Reader(dataset_file)

        for line in dataset_reader:
            for path, label in zip(line["training_data"]["labels"], line["training_data"]["output_paths"]):
                samples.append(path, label)

            if len(samples) > int(config["maximum_dataset_size"]):
                break

    return samples

def filter_samples(samples, config):
    random.seed(42)
    random.shuffle(samples)

    limit = min(len(samples), config["maximum_samples"])

    return samples[:limit]

def save_samples(samples, config):

    output_directory = os.path.dirname(config["output_dataset_path"])
    os.mkdirs(output_directory, exist_ok=True)

    with open(config["output_dataset_path"]) as dataset_file:
        dataset_writer = jsonlines.Writer(dataset_file)

        for path, label in samples:
            local_path = download(path, output_directory)
            dataset_writer({"training_datset" : { "output_paths": [local_path], "labels" : [label] }})

def setup_config(dictionary):
    return config.ConfigurationSet(
        config.config_from_env(prefix="MLCOMMONS"),
        config.config_from_yaml(config_path(), read_from_file=True),
        config.config_from_dict(dictionary),
    )

def config_path():
    home = os.path.expanduser("~")
    home_config_path = os.path.join(home, ".mlcommons", "config.yaml")
    if os.path.exists(home_config_path):
        return home_config_path

    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "default.yaml")

def setup_logging(arguments):

    logging_format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"

    if arguments["verbose"]:
        logging.basicConfig(level=logging.DEBUG, format=logging_format)
    elif arguments["verbose_info"]:
        logging.basicConfig(level=logging.INFO, format=logging_format)
    else:
        logging.basicConfig(level=logging.WARNING, format=logging_format)

    root_logger = logging.getLogger()

    if arguments["verbose"]:
        root_logger.setLevel(logging.DEBUG)
    elif arguments["verbose_info"]:
        root_logger.setLevel(logging.INFO)
    else:
        root_logger.setLevel(logging.WARNING)

if __name__ == "__main__":
    main()


