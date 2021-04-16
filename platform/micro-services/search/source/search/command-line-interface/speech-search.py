
from search.api import search_for_data_book
from search.util import save_dataset
from search.util import load_databook

import os

from argparse import ArgumentParser

import logging

logger = logging.getLogger(__name__)

def main():
    parser = ArgumentParser("Search API exposed as a command line utility.")

    parser.add_argument("-i", "--data-book-path", default=sample_databook_path(), help="Path to data book to search for.")
    parser.add_argument("--dataset-path", default=sample_dataset_path(), help="Path to dataset to search.")
    parser.add_argument("--engine", default="ExistingDatasetSearchEngine", help="Which search engine to use.")
    parser.add_argument("--samples-per-page", default=4, help="How many search results per databook page.")
    parser.add_argument("-o", "--output-path", default="searched-dataset.csv", help="The path to save the new dataset.")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Print out debug messages.")
    parser.add_argument("-vi", "--verbose-info", default=False, action="store_true", help="Print out info messages.")

    arguments = vars(parser.parse_args())

    setup_logging(arguments)

    data_book = load_databook(arguments["data_book_path"])

    dataset = search_for_data_book(arguments, data_book)

    save_dataset(arguments, dataset)

def sample_databook_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "test", "marathi-databook.csv")

def sample_dataset_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "test", "test-dataset.csv")

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





