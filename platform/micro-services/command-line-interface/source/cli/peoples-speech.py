
import os

from argparse import ArgumentParser

import peoples_speech.data_book
import peoples_speech.data_export
import peoples_speech.task_manager

import config

import logging

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("The MLCommons data engineering framework.")

    parser.add_argument("-i", "--data-book-path", default=sample_databook_path(), help="Path to data book to generate a dataset for.")
    parser.add_argument("-o", "--output-dataset-path", default="", help="The path to save the new dataset.")
    parser.add_argument("-c", "--config-file-path", default=".csv", help="The path to save the new dataset.")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Print out debug messages.")
    parser.add_argument("-vi", "--verbose-info", default=False, action="store_true", help="Print out info messages.")

    arguments = vars(parser.parse_args())

    config = setup_config(arguments)

    setup_logging(config)

    logger.debug("Full config: " + str(config))

    data_book = peoples_speech.data_book.load_databook(config["data_book_path"])

    dataset = peoples_speech.task_manager.make_dataset(config, data_book)

    peoples_speech.data_export.save_dataset(config, dataset)

def sample_databook_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "search", "test", "marathi-databook.csv")

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

